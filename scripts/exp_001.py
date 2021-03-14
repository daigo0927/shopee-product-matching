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
import albumentations as A

from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, callbacks, optimizers, losses, metrics
from transformers import BertConfig, TFBertModel, BertTokenizerFast

from utils import seed_everything, ShopeeDataset, f1score, ShopeeF1Score

seed_everything()



def as_supervised(image, title, label_id):
    return (image, title), label_id


def build_model(image_size, n_classes, effnet_weights, use_weights):
    # Image feature
    input_image = layers.Input([*image_size, 3], dtype=tf.uint8)
    x = tf.cast(input_image, dtype=tf.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    net = tf.keras.applications.EfficientNetB0(
        include_top=False,                
        weights=effnet_weights,
        pooling='avg')
    image_feat = net(x)

    # Text feature
    input_text = layers.Input([], dtype=tf.string)
    encoder = hub.KerasLayer(use_weights)
    text_feat = encoder(input_text)
    
    feat = tf.concat([image_feat, text_feat], axis=-1)
    feat = layers.Dropout(0.5)(feat)
    logits = layers.Dense(n_classes)(feat)

    model = tf.keras.Model(inputs=[input_image, input_text], outputs=logits)
    return model


def train(config, logdir):
    datadir = config.datadir
    train_image_dir = f'{datadir}/train_images'
    train_image_paths = glob(f'{train_image_dir}/*.jpg')
    test_image_dir = f'{datadir}/test_images'
    test_image_paths = glob(f'{test_image_dir}/*.jpg')

    df_train = pd.read_csv(f'{datadir}/train.csv')
    df_test = pd.read_csv(f'{datadir}/test.csv')
    df_sub = pd.read_csv(f'{datadir}/sample_submission.csv')

    unique_label_groups = df_train.label_group.unique()
    unique_label_groups = sorted(unique_label_groups)
    label_map = {g: i for i, g in enumerate(unique_label_groups)}
    n_classes = len(label_map)

    df_train['label_group_id'] = df_train.label_group.map(label_map)
    df_train['image_path'] = df_train.image.apply(lambda x: f'{train_image_dir}/{x}')

    df_test['label_group_id'] = -1
    df_test['image_path'] = df_test.image.apply(lambda x: f'{test_image_dir}/{x}')

    image_aug = A.Compose([
        A.Rotate(limit=90),
        A.ColorJitter(),
        A.HorizontalFlip(),
        A.VerticalFlip()
    ])

    target_size = config.preprocess.target_size    
    signatures = (
        tf.TensorSpec(shape=(*target_size, 3), dtype=tf.uint8), 
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32)
        )

    logits_oof = np.zeros((len(df_train), n_classes))
    logits_test = np.zeros((len(df_test), n_classes))

    n_splits = config.train.n_splits
    epochs = config.train.epochs
    batch_size = config.train.batch_size
    learning_rate = config.train.learning_rate
    patience = config.train.patience

    skf = StratifiedKFold(n_splits=n_splits)
    for cv, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train.label_group_id)):
        df_t = df_train.iloc[train_idx]
        df_v = df_train.iloc[val_idx]

        ds_train = ShopeeDataset(df_t, target_size, shuffle=True, image_aug=image_aug)
        ds_val = ShopeeDataset(df_v, target_size)
        ds_test = ShopeeDataset(df_test, target_size)

        ds_train = tf.data.Dataset.from_generator(ds_train, output_signature=signatures)
        ds_val = tf.data.Dataset.from_generator(ds_val, output_signature=signatures)
        ds_test = tf.data.Dataset.from_generator(ds_test, output_signature=signatures)

        ds_train = ds_train.repeat(epochs)

        ds_train = ds_train.map(as_supervised)
        ds_val = ds_val.map(as_supervised)
        ds_test = ds_test.map(as_supervised)

        ds_train = ds_train.batch(batch_size).prefetch(buffer_size=batch_size)
        ds_val = ds_val.batch(batch_size).prefetch(buffer_size=batch_size)
        ds_test = ds_test.batch(batch_size).prefetch(buffer_size=batch_size)

        model = build_model(n_classes=n_classes, **config.model)
        print(model.summary())

        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics.SparseCategoricalAccuracy(name='acc')
            )

        train_steps = np.ceil(len(df_t)/batch_size)
        val_steps = np.ceil(len(df_v)/batch_size)
        recorder = ShopeeF1Score(ds_val, df_v.posting_id, patience=patience)
        model.fit(ds_train,
                  epochs=epochs,
                  callbacks=[recorder, callbacks.TensorBoard(log_dir=logdir)],
                  validation_data=ds_val,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)
        print(f'Best f1score: {recorder.best:.4f}')
        model.set_weights(recorder.best_weights)
        model.save_weights(f'{logdir}/models/ckpt-cv{cv}')

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
    df_concat = pd.concat([pd.DataFrame({'posting_id':df_train.posting_id,
                                         'pred':preds_oof,
                                         'test': False}),
                           pd.DataFrame({'posting_id': df_test.posting_id,
                                         'pred':preds_test,
                                         'test': True})],
                          ignore_index=True)
    pred_map = df_concat.groupby('pred').posting_id.agg('unique').to_dict()
    df_concat['matches'] = df_concat.pred.map(pred_map)
    df_test = df_concat[df_concat.test]
    df_test['matches'] = df_test.matches.apply(lambda x: ' '.join(x))
    df_test[['posting_id','matches']].to_csv(f'{logdir}/submission.csv',index=False)


def run():
    base_config = OmegaConf.load('configs/exp001.yaml')
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(base_config, cli_config)

    time_str = time.strftime('%Y-%m-%dT%H-%M-%S')
    logdir = f'{config.logdir}/{time_str}'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    OmegaConf.save(config, f'{logdir}/config.yaml')
    OmegaConf.save(cli_config, f'{logdir}/override.yaml')
    
    train(config, logdir)


if __name__ == '__main__':
    run()
