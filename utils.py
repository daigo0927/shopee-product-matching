import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks


def seed_everything(seed: int = 42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ShopeeDataset:
    def __init__(self,
                 df,
                 target_size=(512, 512),
                 shuffle=False,
                 image_aug=None,
                 text_aug=None):
        self.df = df
        self.values = df[['image_path', 'title', 'label_group_id']].values
        self.shuffle = shuffle
        self.target_size = target_size
        self.image_aug = image_aug
        self.text_aug = text_aug
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, i):
        path, title, label_id = self.values[i]
        image = cv2.imread(path)[:,:,::-1]
        
        if self.image_aug:
            image = self.image_aug(image=image)['image']
        if self.text_aug:
            title = self.text_aug(title)
        
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize_with_pad(image, *self.target_size)
        title = tf.convert_to_tensor(title, dtype=tf.string)
        label_id = tf.convert_to_tensor(label_id, dtype=tf.int32)
        return image, title, label_id
    
    def __call__(self):
        if self.shuffle:
            np.random.shuffle(self.values)
            
        for sample in self:
            yield sample    


def f1score(posting_ids, labels, preds):
    def f1(row):
        n = len(np.intersect1d(row.target, row.matches))
        return 2*n / (len(row.target)+len(row.matches))
    
    df = pd.DataFrame({
        'posting_id': posting_ids,
        'pred': preds,
        'label': labels
    })
    true_map = df.groupby('label').posting_id.agg('unique').to_dict()
    df['target'] = df.label.map(true_map)
    pred_map = df.groupby('pred').posting_id.agg('unique').to_dict()
    df['matches'] = df.pred.map(pred_map)
    df['f1'] = df.apply(f1, axis=1)
    return df.f1.mean()


class ShopeeF1Score(callbacks.Callback):
    def __init__(self, ds, posting_ids, patience=5):
        super().__init__()
        self.ds = ds
        self.posting_ids = posting_ids
        self.patience = patience
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs={}):
        labels, preds = [], []
        for inp, lbl in self.ds:
            logit = self.model(inp, training=False)
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

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))    
