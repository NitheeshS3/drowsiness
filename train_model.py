#!/usr/bin/env python3
"""
train_model.py — FAST TRAIN VERSION (no MediaPipe)
⚡ Trains CNN + LSTM drowsiness model quickly using pre-cropped frames
⚡ Uses folders:
    data/Alert/*.jpg, *.jpeg, *.png
    data/Drowsy/*.jpg, *.jpeg, *.png
Saves:
    model/my_model.h5 (best)
    model/my_model_final.h5 (final)
"""

import os, sys, glob, random
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.model_selection import train_test_split

# ==== Config ====
DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 145
SEQ_LEN = 10
BATCH_SIZE = 8
EPOCHS = 18
RANDOM_SEED = 42


# ==== Dataset prep ====
def collect_image_sequences(data_dir):
    classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    if not classes:
        return [], []
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        paths = []
        # Accept all common image extensions
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(data_dir, c, ext)))
        paths = sorted(paths)
        if not paths:
            print(f"⚠️ No images found in {c} folder.")
        for i in range(0, max(1, len(paths) - SEQ_LEN + 1), 3):  # skip frames to speed up
            seq = paths[i:i + SEQ_LEN]
            if len(seq) == SEQ_LEN:
                samples.append((class_to_idx[c], seq))
    return samples, classes


def preprocess_image_cv2(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def seq_generator(samples):
    for label_idx, seq_paths in samples:
        frames = [preprocess_image_cv2(p) for p in seq_paths]
        arr = np.stack(frames, axis=0)
        yield arr, label_idx


def make_tf_dataset(sample_list, batch_size):
    out_shapes = ((SEQ_LEN, IMG_SIZE, IMG_SIZE, 3), ())
    out_types = (tf.float32, tf.int32)

    def gen():
        for x, y in seq_generator(sample_list):
            yield x, y

    ds = tf.data.Dataset.from_generator(gen, output_types=out_types, output_shapes=out_shapes)
    ds = ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ==== Model ====
def build_model():
    input_layer = layers.Input(shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3))
    x = input_layer

    def td_conv_block(x, filters):
        x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.Conv2D(filters, 3, padding='same', activation='relu'))(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
        return x

    x = td_conv_block(x, 16)
    x = td_conv_block(x, 32)
    x = td_conv_block(x, 64)
    x = td_conv_block(x, 128)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=out)
    return model


# ==== Train ====
def main():
    samples, classes = collect_image_sequences(DATA_DIR)
    if not samples:
        print("❌ No images found. Check your dataset structure.")
        sys.exit(1)

    print(f"✅ Found {len(samples)} sequences across {classes}")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    labels = [s[0] for s in samples]
    train_s, val_s = train_test_split(samples, test_size=0.15, random_state=RANDOM_SEED, stratify=labels)

    train_ds = make_tf_dataset(train_s, BATCH_SIZE)
    val_ds = make_tf_dataset(val_s, BATCH_SIZE)

    model = build_model()
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    model.summary()

    cb = [
        callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'my_model.h5'), save_best_only=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)
    model.save(os.path.join(MODEL_DIR, 'my_model_final.h5'))
    print("✅ Training complete. Models saved in /model/")


if __name__ == "__main__":
    main()
