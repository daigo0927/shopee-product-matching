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
from src.layers import ArcFace, CircleLossCL

AUTOTUNE = tf.data.experimental.AUTOTUNE
seed_everything()


'''
exp 8:
- Efficient data operation
- Image feature: EfficientNet-b0
- Text feature: BERT and USE
- CircleLoss (class-wise version)
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


def rearrange(image_inputs, tokens, embed):
    image, label_id = image_inputs
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    tokens = [input_ids, attention_mask]
    return (image, *tokens, embed, label_id), label_id


class DatasetFactory:
    def __init__(self,
                 label_map,
                 n_classes,
                 target_size):
        self.label_map = label_map
        self.n_classes = n_classes
        self.target_size = target_size

    def build(self, df, tokens, title_embeds,
              image_aug=None, epochs=1, batch_size=4, is_training=False):
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

        ds_token = tf.data.Dataset.from_tensor_slices(tokens)
        ds_embed = tf.data.Dataset.from_tensor_slices(title_embeds)

        # Concat image and title iterator
        ds = tf.data.Dataset.zip((ds_image, ds_token, ds_embed))
        ds = ds.map(rearrange, num_parallel_calls=AUTOTUNE)

        if is_training:
            ds = ds.shuffle(buffer_size=len(df),
                            reshuffle_each_iteration=True)

        ds = ds.batch(batch_size).prefetch(buffer_size=batch_size)
        return ds


class FeatureFactory:
    def __init__(self, transformer_name, tokenize_params, encoder_weights):
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.tokenize_params = tokenize_params
        self.sentence_encoder = hub.KerasLayer(encoder_weights)

    def build(self, df):
        tokens = self.tokenizer(df.title.to_list(),
                                padding='max_length',
                                truncation=True,
                                return_tensors='tf',
                                **self.tokenize_params)

        # Build USE feature
        n_batches = np.ceil(len(df)/1000).astype(int)
        title_embeds = []
        for i in tqdm(range(n_batches), desc='Embed text'):
            embeds = self.sentence_encoder(df.title[i*1000:(i+1)*1000])
            title_embeds.append(embeds)
        title_embeds = tf.concat(title_embeds, axis=0)

        return tokens, title_embeds

    def save(self, savedir):
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        self.tokenizer.save_pretrained(savedir)


def build_model(image_size,
                token_length,
                effnet_weights,
                transformer_name,
                arcface_params,
                n_classes) -> tf.keras.Model:
    # Image feature
    image = layers.Input([*image_size, 3], dtype=tf.uint8, name='image')
    x = tf.cast(image, dtype=tf.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,                
        weights=effnet_weights,
        pooling='avg')
    image_feat = effnet(x)

    # Transformer feature
    input_ids = layers.Input([token_length], dtype=tf.int32,
                             name='input_ids')
    attention_mask = layers.Input([token_length], dtype=tf.int32,
                                  name='attn_mask')
    tokens = [input_ids, attention_mask]

    bert = TFAutoModel.from_pretrained(transformer_name)
    text_outputs = bert(tokens)
    # (batch_size, sequence_length, hidden_size)
    last_hidden_state = text_outputs.last_hidden_state
    text_feat = tf.reduce_mean(last_hidden_state, axis=1)
    
    # Universal sentence encoder embedding
    embed = layers.Input([512], dtype=tf.float32, name='embed')

    feat = tf.concat([image_feat, text_feat, embed], axis=-1)
    feat = layers.Dropout(0.5)(feat)

    label = tf.keras.layers.Input([], dtype=tf.int32)
    label_onehot = tf.one_hot(label, depth=n_classes)

    arcface = ArcFace(n_classes, **arcface_params)
    logit = arcface([feat, label_onehot])

    model = tf.keras.Model(inputs=[image, *tokens, embed, label],
                           outputs=logit)
    return model


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
    n_splits = train_config.n_splits
    epochs = train_config.epochs
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
    logits_oof = np.zeros((len(df_train), n_classes))
    logits_test = np.zeros((len(df_test), n_classes))

    skf = StratifiedKFold(n_splits=train_config.n_splits)
    splits = skf.split(df_train, df_train.label_group_id)
    for cv, (train_idx, val_idx) in enumerate(splits):
        df_t = df_train.iloc[train_idx]
        df_v = df_train.iloc[val_idx]

        # Build text features
        feat_factory = FeatureFactory(
            transformer_name=model_config.transformer_name,
            tokenize_params=preprocess_config.tokenize,
            encoder_weights=preprocess_config.encoder_weights
        )
        tokens_t, title_embeds_t = feat_factory.build(df_t)
        tokens_v, title_embeds_v = feat_factory.build(df_v)
        tokens_test, title_embeds_test = feat_factory.build(df_test)
        feat_factory.save(savedir=f'{logdir}/features/cv{cv}')

        # Build dataset
        ds_train = ds_factory.build(
            df_t, tokens_t, title_embeds_t,
            image_aug=image_aug,
            epochs=epochs,
            batch_size=batch_size,
            is_training=True
        )
        ds_val = ds_factory.build(
            df_v, tokens_v, title_embeds_v,
            batch_size=batch_size
        )
        ds_test = ds_factory.build(
            df_test, tokens_test, title_embeds_test,
            batch_size=batch_size
        )

        # Build model
        model = build_model(n_classes=n_classes, **model_config)
        print(model.summary())

        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics.SparseCategoricalAccuracy(name='acc')
            )

        recorder = ShopeeF1Score(ds_val, df_v.posting_id, patience=patience)
        model_dir = f'{logdir}/models'
        model.fit(ds_train,
                  epochs=epochs,
                  callbacks=[recorder,
                             callbacks.TensorBoard(log_dir=f'{logdir}/cv{cv}'),
                             callbacks.ModelCheckpoint(
                                 filepath=f'{model_dir}/cv{cv}',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min'
                             )],
                  validation_data=ds_val,
                  steps_per_epoch=len(ds_train),
                  validation_steps=len(ds_val))
        
        best_score = recorder.best
        print(f'Best f1score: {best_score:.4f}')
        model.set_weights(recorder.best_weights)
        model.save_weights(f'{model_dir}/cv{cv}-score{best_score:.4f}')
        logits_oof[val_idx] = model.predict(ds_val)
        logits_test += model.predict(ds_test)/n_splits

    preds_oof = np.argmax(logits_oof, axis=-1)
    cv_score = f1score(df_train.posting_id,
                       labels=df_train.label_group_id,
                       preds=preds_oof)
    print(f'Overall f1score: {cv_score:.4f}')

    np.save(f'{logdir}/logits_oof.npy', logits_oof)
    np.save(f'{logdir}/logits_test.npy', logits_test)

    preds_test = np.argmax(logits_test, axis=-1)
    df_concat = pd.concat([pd.DataFrame({'posting_id': df_train.posting_id,
                                         'pred': preds_oof,
                                         'test': False}),
                           pd.DataFrame({'posting_id': df_test.posting_id,
                                         'pred': preds_test,
                                         'test': True})],
                          ignore_index=True)
    pred_map = df_concat.groupby('pred').posting_id.agg('unique').to_dict()
    df_concat['matches'] = df_concat.pred.map(pred_map)
    df_test = df_concat[df_concat.test]
    df_test['matches'] = df_test.matches.apply(lambda x: ' '.join(x))
    df_test[['posting_id', 'matches']].to_csv(f'{logdir}/submission.csv',
                                              index=False)


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
