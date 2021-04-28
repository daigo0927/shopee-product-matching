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
from typing import List
from functools import partial
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics

from src.utils import seed_everything
from src.layers import CircleLossCL

AUTOTUNE = tf.data.experimental.AUTOTUNE
seed_everything()


'''
exp 12:
- Efficient data operation
- Image feature: allow any type of EfficientNet (b0-7)
- Text feature: USE and TfIdf
- CircleLossCL
- output both embedding and logit
- Train on all dataset, No CV, save weights every 5 epochs
- Check F1score for all training data for each weights at last
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
    image = tf.image.resize_with_pad(image, *target_size)
    image = tf.cast(image, dtype=tf.uint8)
    return image, label_id


def convert_transform(transform):
    def f(image):
        return transform(image=image)['image']
    
    def transform_tf(image, label_id):
        image = tf.numpy_function(func=f, inp=[image], Tout=tf.uint8)
        return image, label_id
    return transform_tf


def set_shape(image, label_id, image_size):
    image.set_shape([*image_size, 3])
    label_id.set_shape([])
    return image, label_id


def to_dense(tfidf):
    tfidf = tf.sparse.to_dense(tfidf)
    return tfidf


def rearrange(image_inputs, tfidf, embed):
    image, label_id = image_inputs
    return (image, tfidf, embed, label_id), (label_id, 0)


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
        # Convert to coodinates format
        tfidf_coo = tfidf.tocoo()
        # Convert to SparseTensor
        indices = np.stack([tfidf_coo.row, tfidf_coo.col], axis=1)
        tfidf_tensor = tf.SparseTensor(
            indices=indices,
            values=tfidf_coo.data,
            dense_shape=(len(df), self.tfidf_dim))
        tfidf_tensor = tf.sparse.reorder(tfidf_tensor)

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
                effnet_type,
                effnet_weights,
                circle_loss_params,
                n_classes) -> tf.keras.Model:
    # Image feature
    image = layers.Input([*image_size, 3], dtype=tf.uint8, name='image')
    x = tf.cast(image, dtype=tf.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    EfficientNet = getattr(tf.keras.applications, effnet_type)
    effnet = EfficientNet(include_top=False,
                          weights=effnet_weights,
                          pooling='avg')
    image_feat = effnet(x)

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

    model = tf.keras.Model(inputs=[image, tfidf, embed, label],
                           outputs=[logits, feat])
    return model


def dummy_loss(y_true, y_pred):
    ''' Dummy loss to ignore feature embedding in the training '''
    return 0


def f1score(labels, preds):
    def f1(row):
        n = len(np.intersect1d(row.target, row.matches))
        return 2*n / (len(row.target)+len(row.matches))
    
    df = pd.DataFrame({
        'target': labels,
        'matches': preds
    })
    df['f1'] = df.apply(f1, axis=1)
    return df.f1.mean()


def neighbor_f1score(distances, indices, df, threshold):
    preds = []
    # indices: [0, n_samples), preds: array of posting_id
    for dists, ids in zip(distances, indices):
        neighbor_ids = ids[dists < threshold]
        pred = df.posting_id.iloc[neighbor_ids].values
        preds.append(pred)

    # True identical products
    map_fn = df.groupby('label_group').posting_id.agg('unique').to_dict()
    labels = df.label_group.map(map_fn)
    f1 = f1score(labels=labels, preds=preds)
    return f1


class FixedStepCheckpoint(callbacks.Callback):
    def __init__(self,
                 ds: tf.data.Dataset,
                 df: pd.DataFrame,
                 save_step: int,
                 savedir: str, **kwargs):
        super().__init__(**kwargs)
        self.ds = ds
        self.df = df
        self.save_step = save_step
        self.savedir = savedir

        if not os.path.exists(savedir):
            os.makedirs(savedir)

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.save_step != 0:
            return

        self.model.save_weights(f'{self.savedir}/weights_{epoch+1:02}epoch')

        _, features = self.model.predict(self.ds)
        knn = NearestNeighbors(n_neighbors=50, metric='cosine', n_jobs=-1)
        knn.fit(features)
        distances, indices = knn.kneighbors(features)

        scores = []
        for thr in np.arange(0.1, 1.0, 0.1):
            f1 = neighbor_f1score(distances, indices, self.df, thr)
            scores.append([thr, f1])
        df = pd.DataFrame(scores, columns=['threshold', 'f1score'])
        df.to_csv(f'{self.savedir}/f1_{epoch+1:02}epoch.csv', index=False)


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
    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate

    # Build image augmentation
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

    # Build text features
    feat_factory = FeatureFactory(
        tfidf_config=preprocess_config.tfidf,
        encoder_weights=preprocess_config.encoder_weights
        )
    tfidf_feats_train, title_embeds_train = feat_factory.build(df_train)
    tfidf_feats_test, title_embeds_test = feat_factory.build(df_test)
    feat_factory.save(savedir=f'{logdir}/features')

    # Build dataset
    ds_train = ds_factory.build(
        df_train, tfidf_feats_train, title_embeds_train,
        image_aug=image_aug,
        batch_size=batch_size,
        is_training=True
    )
    ds_val = ds_factory.build(
        df_train, tfidf_feats_train, title_embeds_train,
        batch_size=batch_size
    )

    # Build model
    model = build_model(n_classes=n_classes, **model_config)
    print(model.summary())

    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate),
        loss=[losses.SparseCategoricalCrossentropy(from_logits=True),
              dummy_loss],
        metrics=[[metrics.SparseCategoricalAccuracy(name='acc')], []]
    )

    model.fit(ds_train,
              epochs=train_config.epochs,
              callbacks=[
                  callbacks.TensorBoard(log_dir=logdir),
                  FixedStepCheckpoint(ds_val, df_train,
                                      savedir=f'{logdir}/models',
                                      **train_config.checkpoint)],
              steps_per_epoch=len(ds_train))


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
