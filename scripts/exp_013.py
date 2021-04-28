import os
import sys
sys.path.append(f'{os.path.dirname(__file__)}/..')
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tftxt
import tensorflow_addons as tfa
import albumentations as A

from glob import glob
from tqdm import tqdm
from functools import partial
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
from transformers import AutoTokenizer, TFAutoModel

from src.utils import seed_everything, f1score, ShopeeF1Score
from src.layers import CircleLossCL

AUTOTUNE = tf.data.experimental.AUTOTUNE
seed_everything()


'''
exp 9:
- Efficient data operation
- Image feature: EfficientNet-b0
- Text feature: USE and TfIdf
- CircleLossCL
- output both embedding and logit (only logit in ~007)
- multiscale image features
'''


def build_df(datadir):
    train_image_dir = f'{datadir}/train_images'
    test_image_dir = f'{datadir}/test_images'

    df_train = pd.read_csv(f'{datadir}/train.csv')
    df_test = pd.read_csv(f'{datadir}/test.csv')

    unique_label_groups = df_train.label_group.unique()
    unique_label_groups = sorted(unique_label_groups)
    label_map = {g: i for i, g in enumerate(unique_label_groups)}
    n_classes = len(label_map)

    df_train['label_group_id'] = df_train.label_group.map(label_map)
    df_train['image_path'] = df_train.image.apply(lambda x: f'{train_image_dir}/{x}')

    df_test['label_group_id'] = -1
    df_test['image_path'] = df_test.image.apply(lambda x: f'{test_image_dir}/{x}')
    return df_train, df_test, label_map, n_classes


def read_image(path, label_id, target_size):
    image = tf.io.decode_jpeg(tf.io.read_file(path))
    
    image_small = tf.image.resize_with_pad(image, *target_size.small)
    image_medium = tf.image.resize_with_pad(image, *target_size.medium)
    image_large = tf.image.resize_with_pad(image, *target_size.large)
    images = (image_small, image_medium, image_large)
    return images, label_id


def convert_transform(transform):
    def f(image):
        return transform(image=image)['image']
    
    def transform_tf(images, label_id):
        image_s, image_m, image_l = images
        image_s = tf.numpy_function(func=f, inp=[image_s], Tout=tf.float32)
        image_m = tf.numpy_function(func=f, inp=[image_m], Tout=tf.float32)
        image_l = tf.numpy_function(func=f, inp=[image_l], Tout=tf.float32)
        images = (image_s, image_m, image_l)
        return images, label_id
    return transform_tf


def set_shape(images, label_id, image_size):
    image_s, image_m, image_l = images
    image_s.set_shape([*image_size.small, 3])
    image_m.set_shape([*image_size.medium, 3])
    image_l.set_shape([*image_size.large, 3])
    images = (image_s, image_m, image_l)    
    label_id.set_shape([])
    return images, label_id


def to_tfidf_tensor(tfidf, dense_shape):
    # Convert to coodinates format
    tfidf_coo = tfidf.tocoo()
    # Convert to SparseTensor
    indices = np.stack([tfidf_coo.row, tfidf_coo.col], axis=1)
    tfidf_tensor = tf.SparseTensor(
        indices=indices,
        values=tfidf_coo.data,
        dense_shape=dense_shape)
    tfidf_tensor = tf.sparse.reorder(tfidf_tensor)
    return tfidf_tensor


def to_dense(tfidf):
    tfidf = tf.sparse.to_dense(tfidf)
    return tfidf


def rearrange(image_inputs, tfidf, embed):
    images, label_id = image_inputs
    return (*images, tfidf, embed, label_id), (label_id, 0)


class DatasetFactory:
    def __init__(self,
                 label_map,
                 n_classes,
                 target_size):
        self.label_map = label_map
        self.n_classes = n_classes
        self.target_size = target_size

    def build(self, df, tfidf_feats, title_embeds,
              image_aug=None, batch_size=4, is_training=False):
        # Build image iterator
        ds_image = tf.data.Dataset.from_tensor_slices((
            df.image_path, df.label_group_id
            ))
        read_fn = partial(read_image, target_size=self.target_size)
        ds_image = ds_image.map(read_fn, num_parallel_calls=AUTOTUNE)

        if image_aug:
            image_aug_fn = convert_transform(image_aug)
            ds_image = ds_image.map(image_aug_fn, num_parallel_calls=AUTOTUNE)
            set_shape_fn = partial(set_shape, image_size=self.target_size)
            ds_image = ds_image.map(set_shape_fn, num_parallel_calls=AUTOTUNE)

        ds_tfidf = tf.data.Dataset.from_tensor_slices(tfidf_feats)
        ds_tfidf = ds_tfidf.map(to_dense, num_parallel_calls=AUTOTUNE)
        
        ds_embed = tf.data.Dataset.from_tensor_slices(title_embeds)

        # Concat image and title iterator
        ds = tf.data.Dataset.zip((ds_image, ds_tfidf, ds_embed))
        ds = ds.map(rearrange, num_parallel_calls=AUTOTUNE)

        if is_training:
            ds = ds.shuffle(buffer_size=len(df),
                            reshuffle_each_iteration=True)

        ds = ds.batch(batch_size).prefetch(buffer_size=batch_size)
        return ds


class FeatureFactory:
    def __init__(self, tfidf_config, encoder_weights):
        self.tfidf_dim = tfidf_config.max_features
        self.tfidf_model = TfidfVectorizer(**tfidf_config)
        self.sentence_encoder = hub.KerasLayer(encoder_weights)

    def build(self, df):
        # Build Tfidf feature
        if not hasattr(self.tfidf_model, 'vocabulary_'):
            tfidf = self.tfidf_model.fit_transform(df.title)
        else:
            tfidf = self.tfidf_model.transform(df.title)
        tfidf_tensor = to_tfidf_tensor(
            tfidf, dense_shape=(len(df), self.tfidf_dim))

        # Build USE feature
        n_batches = np.ceil(len(df)/1000).astype(int)
        title_embeds = []
        for i in tqdm(range(n_batches), desc='Embed text'):
            embeds = self.sentence_encoder(df.title[i*1000:(i+1)*1000])
            title_embeds.append(embeds)
        title_embeds = tf.concat(title_embeds, axis=0)

        return tfidf_tensor, title_embeds

    def save(self, savedir):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        with open(f'{savedir}/tfidf.pkl', 'wb') as f:
            pickle.dump(self.tfidf_model, f)


def build_model(image_size,
                tfidf_dim,
                final_feature_dim,
                effnet_weights,
                circle_loss_params,
                n_classes) -> tf.keras.Model:
    # Image feature
    image_small = layers.Input([*image_size.small, 3],
                               dtype=tf.float32, name='image_small')
    image_medium = layers.Input([*image_size.medium, 3],
                                dtype=tf.float32, name='image_medium')
    image_large = layers.Input([*image_size.large, 3],
                               dtype=tf.float32, name='image_large')
    images = [image_small, image_medium, image_large]
    xs = tf.keras.applications.efficientnet.preprocess_input(image_small)
    xm = tf.keras.applications.efficientnet.preprocess_input(image_medium)
    xl = tf.keras.applications.efficientnet.preprocess_input(image_large)

    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,                
        weights=effnet_weights,
        pooling='avg')
    image_feat_s = effnet(xs)
    image_feat_m = effnet(xm)
    image_feat_l = effnet(xl)
    image_feat = tf.concat([image_feat_s, image_feat_m, image_feat_l], axis=-1)
    image_feat = layers.Dropout(0.5)(image_feat)
    image_feat = layers.Dense(1024, activation='relu')(image_feat)

    # Text feature
    tfidf = layers.Input([tfidf_dim], dtype=tf.float32, name='tfidf')
    tfidf_feat = layers.Dropout(0.5)(tfidf)
    tfidf_feat = layers.Dense(512, activation='relu')(tfidf_feat)

    # Universal sentence encoder embedding
    embed = layers.Input([512], dtype=tf.float32, name='embed')

    feat = tf.concat([image_feat, tfidf_feat, embed], axis=-1)
    feat = layers.Dropout(0.5)(feat)
    feat = layers.Dense(final_feature_dim)(feat)

    label = tf.keras.layers.Input([], dtype=tf.int32)
    label_onehot = tf.one_hot(label, depth=n_classes)

    similarity = CircleLossCL(n_classes, **circle_loss_params)
    logits = similarity([feat, label_onehot])

    model = tf.keras.Model(inputs=[images, tfidf, embed, label],
                           outputs=[logits, feat])
    return model


def dummy_loss(y_true, y_pred):
    ''' Dummy loss to ignore feature embedding in the training '''
    return 0


class ShopeeF1V2(ShopeeF1Score):
    def on_epoch_end(self, epoch, logs={}):
        labels, preds = [], []
        for inp, (lbl, _) in self.ds:
            logit, _ = self.model(inp, training=False)
            preds.append(tf.argmax(logit, axis=-1).numpy())
            labels.append(lbl.numpy())
            
        preds = np.concatenate(preds, axis=0).astype(int)
        labels = np.concatenate(labels, axis=0).astype(int)
        current = f1score(self.posting_ids, labels, preds)

        if current > self.best: # larger is better
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)


def train(config, logdir):
    # Load settings
    datadir = config.datadir
    preprocess_config = config.preprocess
    model_config = config.model
    train_config = config.train

    # Build dataframe
    df_train, df_test, label_map, n_classes = build_df(datadir)

    # Load training parameters
    target_size = preprocess_config.target_size
    final_feature_dim = model_config.final_feature_dim
    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate
    patience = train_config.patience

    # Build CV-shared objects
    image_aug = A.Compose([
        A.ShiftScaleRotate(scale_limit=0.2),
        A.ColorJitter(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Cutout(num_holes=8, max_h_size=12, max_w_size=12)
    ])

    ds_factory = DatasetFactory(
        label_map=label_map,
        n_classes=n_classes,
        target_size=target_size,
    )

    # Prepare training result placeholders
    feats_oof = np.zeros((len(df_train), final_feature_dim))
    logits_oof = np.zeros((len(df_train), n_classes))

    skf = StratifiedKFold(n_splits=train_config.n_splits)
    splits = skf.split(df_train, df_train.label_group_id)
    for cv, (train_idx, val_idx) in enumerate(splits):
        df_t = df_train.iloc[train_idx]
        df_v = df_train.iloc[val_idx]

        # Build text features
        feat_factory = FeatureFactory(
            tfidf_config=preprocess_config.tfidf,
            encoder_weights=preprocess_config.encoder_weights
        )
        tfidf_feats_t, title_embeds_t = feat_factory.build(df_t)
        tfidf_feats_v, title_embeds_v = feat_factory.build(df_v)
        tfidf_feats_test, title_embeds_test = feat_factory.build(df_test)
        feat_factory.save(savedir=f'{logdir}/features/cv{cv}')

        # Build dataset
        ds_train = ds_factory.build(
            df_t, tfidf_feats_t, title_embeds_t,
            image_aug=image_aug,
            batch_size=batch_size,
            is_training=True
        )
        ds_val = ds_factory.build(
            df_v, tfidf_feats_v, title_embeds_v,
            batch_size=batch_size
        )

        # Build model
        model = build_model(n_classes=n_classes, **model_config)
        print(model.summary())

        model.compile(
            optimizer=tfa.optimizers.RAdam(learning_rate),
            loss=[losses.SparseCategoricalCrossentropy(from_logits=True),
                  dummy_loss],
            metrics=[[metrics.SparseCategoricalAccuracy(name='acc')], []]
            )

        recorder = ShopeeF1V2(ds_val, df_v.posting_id, patience=patience)
        model_dir = f'{logdir}/models'
        model.fit(ds_train,
                  epochs=train_config.epochs,
                  callbacks=[
                      recorder,
                      callbacks.TensorBoard(log_dir=f'{logdir}/cv{cv}')],
                  validation_data=ds_val,
                  steps_per_epoch=len(ds_train),
                  validation_steps=len(ds_val))

        # Save the best weights
        best_score = recorder.best
        print(f'Best f1score: {best_score:.4f}')
        model.set_weights(recorder.best_weights)
        model.save_weights(f'{model_dir}/cv{cv}-score{best_score:.4f}')

        logits_val, feats_val = model.predict(ds_val)
        feats_oof[val_idx] = feats_val
        logits_oof[val_idx] = logits_val

    preds_oof = np.argmax(logits_oof, axis=-1)
    cv_score = f1score(df_train.posting_id,
                       labels=df_train.label_group_id,
                       preds=preds_oof)
    print(f'Overall f1score: {cv_score:.4f}')

    np.save(f'{logdir}/feats_oof.npy', feats_oof)


def run():
    # Get filename: 'exp_abc'
    exec_file = os.path.basename(__file__).split('.')[0]
    
    # Load setting from yaml and CLI
    base_config = OmegaConf.load(f'configs/{exec_file}.yaml')
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)

    # Create log directory
    time_str = time.strftime('%Y-%m-%dT%H-%M-%S')
    logdir = f'{config.logdir}/{exec_file}/{time_str}'
    print(f'logdir: {logdir}')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Save actual and CLI-specified parameters
    OmegaConf.save(config, f'{logdir}/config.yaml')
    OmegaConf.save(cli_config, f'{logdir}/override.yaml')
    
    train(config, logdir)


if __name__ == '__main__':
    run()
