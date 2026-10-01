from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


def train(data_dir: Path, out_dir: Path, epochs_head: int = 10, epochs_ft: int = 10) -> None:
    img_size = (224, 224)
    batch_size = 32

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Expected train/ and val/ folders under data_dir")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, label_mode="int", shuffle=True, seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, label_mode="int", shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, label_mode="int", shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
        ]
    )

    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = augment(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-6, monitor="val_loss"),
    ]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_head, callbacks=callbacks)

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_ft, callbacks=callbacks)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "disease_model.h5")
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    (out_dir / "metrics.json").write_text(json.dumps({"test_accuracy": float(test_acc)}, indent=2), encoding="utf-8")

    print(f"Saved: {out_dir / 'disease_model.h5'}")
    print(f"Saved: {out_dir / 'class_names.json'}")
    print(f"Test accuracy: {test_acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing train/val/test")
    ap.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "model"))
    ap.add_argument("--epochs_head", type=int, default=10)
    ap.add_argument("--epochs_ft", type=int, default=10)
    args = ap.parse_args()

    train(Path(args.data_dir), Path(args.out_dir), epochs_head=args.epochs_head, epochs_ft=args.epochs_ft)


if __name__ == "__main__":
    main()

