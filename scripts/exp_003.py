import os
import sys
sys.path.append(f'{os.path.dirname(__file__)}/..')
import cv2
import time
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
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
from transformers import AutoTokenizer, TFAutoModel

from src.utils import seed_everything, f1score, ShopeeF1Score
from src.layers import ArcFace, CircleLossCL

AUTOTUNE = tf.data.experimental.AUTOTUNE
seed_everything()


'''
exp3:
- Efficient data operation
- BERT and EffNet-based feature
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


def rearrange(image_inputs, tokens):
    image, label_id = image_inputs
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    token_type_ids = tokens['token_type_ids']
    return (image, input_ids, attention_mask, token_type_ids), label_id


class DatasetFactory:
    def __init__(self,
                 label_map,
                 n_classes,
                 tokenizer,
                 target_size,
                 max_token_length):
        self.label_map = label_map
        self.n_classes = n_classes
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.max_token_length = max_token_length

    def build(self, df, image_aug=None, text_aug=None,
              epochs=1, batch_size=4, is_training=False):
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

        # Build title iterator
        tokens = self.tokenizer(df.title.to_list(),
                                padding='max_length',
                                truncation=True,
                                return_tensors='tf',
                                max_length=self.max_token_length)
        ds_title = tf.data.Dataset.from_tensor_slices(tokens)

        # Concat image and title iterator
        ds = tf.data.Dataset.zip((ds_image, ds_title))
        ds = ds.map(rearrange, num_parallel_calls=AUTOTUNE)

        if is_training:
            ds = ds.shuffle(buffer_size=len(df),
                            reshuffle_each_iteration=True)

        ds = ds.batch(batch_size).prefetch(buffer_size=batch_size)
        return ds        


def build_model(image_size,
                token_length,
                effnet_weights,
                transformer_name,
                output_dim=1024) -> tf.keras.Model:
    # Image feature
    image = layers.Input([*image_size, 3], dtype=tf.uint8, name='image')
    x = tf.cast(image, dtype=tf.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,                
        weights=effnet_weights,
        pooling='avg')
    image_feat = effnet(x)

    # Text feature
    input_ids = layers.Input([token_length], dtype=tf.int32,
                             name='input_ids')
    attention_mask = layers.Input([token_length], dtype=tf.int32,
                                  name='attn_mask')
    token_type_ids = layers.Input([token_length], dtype=tf.int32,
                                  name='type_ids')
    tokens = [input_ids, attention_mask, token_type_ids]
    
    bert = TFAutoModel.from_pretrained(transformer_name)
    text_outputs = bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
    # (batch_size, sequence_length, hidden_size)
    last_hidden_state = text_outputs.last_hidden_state
    text_feat = tf.reduce_mean(last_hidden_state, axis=1)
    
    feat = tf.concat([image_feat, text_feat], axis=-1)
    feat = layers.Dropout(0.5)(feat)
    logits = layers.Dense(output_dim)(feat)

    model = tf.keras.Model(inputs=[image, *tokens], outputs=logits)
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
    max_token_length = preprocess_config.max_token_length
    n_splits = train_config.n_splits
    epochs = train_config.epochs
    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate
    patience = train_config.patience

    # Build CV-shared objects
    image_aug = A.Compose([
        A.Rotate(limit=180),
        A.ColorJitter(),
        A.HorizontalFlip(),
        A.VerticalFlip()
    ])
    tokenizer = AutoTokenizer.from_pretrained(model_config.transformer_name)

    ds_factory = DatasetFactory(
        label_map=label_map,
        n_classes=n_classes,
        tokenizer=tokenizer,
        target_size=target_size,
        max_token_length=max_token_length
    )
    ds_test = ds_factory.build(df_test, batch_size=batch_size)

    # Prepare training result placeholders
    logits_oof = np.zeros((len(df_train), n_classes))
    logits_test = np.zeros((len(df_test), n_classes))

    skf = StratifiedKFold(n_splits=train_config.n_splits)
    splits = skf.split(df_train, df_train.label_group_id)
    for cv, (train_idx, val_idx) in enumerate(splits):
        df_t = df_train.iloc[train_idx]
        df_v = df_train.iloc[val_idx]

        # Build dataset
        ds_train = ds_factory.build(
            df_t,
            image_aug=image_aug,
            epochs=epochs,
            batch_size=batch_size,
            is_training=True
        )
        ds_val = ds_factory.build(df_v, batch_size=batch_size)

        # Build model
        model = build_model(**model_config)

        # TODO: set appropriate loss function
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
    # Load setting from yaml and CLI
    base_config = OmegaConf.load('configs/exp003.yaml')
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)

    # Create log directory
    exec_file = os.path.basename(__file__).split('.')[0]
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
