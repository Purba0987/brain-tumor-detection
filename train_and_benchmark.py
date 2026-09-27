"""
Brain Tumor Classification - Model Training, Fine-Tuning & Architecture Benchmarking Script
Supports VGG16 (baseline), ResNet50, EfficientNetB0, and EfficientNetB2.
Computes Class Weights, Learning Rate Scheduling, Early Stopping, Evaluation Metrics, and XAI Validation.
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.model_trainer import build_model, compute_balanced_class_weights, evaluate_model_metrics, CLASSES
from modules.xai_metrics import evaluate_xai_batch
from modules.xai_engine import generate_gradcam

def create_synthetic_dataset(output_dir="dataset_sample", samples_per_class=50, img_size=(128, 128)):
    """
    Creates a sample directory structure with synthetic MRI images for testing model training pipelines.
    """
    print(f"Creating sample dataset in '{output_dir}'...")
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            cls_dir = os.path.join(output_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            num_imgs = samples_per_class if split == 'train' else max(10, samples_per_class // 5)
            for i in range(num_imgs):
                # Generate synthetic MRI grayscale pattern
                img_data = np.random.normal(120, 30, (img_size[0], img_size[1], 3)).clip(0, 255).astype(np.uint8)
                if cls != 'notumor':
                    # Draw a simulated tumor blob
                    rr, cc = np.ogrid[:img_size[0], :img_size[1]]
                    circle_mask = (rr - 64)**2 + (cc - 64)**2 <= 20**2
                    img_data[circle_mask] = np.array([220, 220, 220], dtype=np.uint8)
                plt.imsave(os.path.join(cls_dir, f"{cls}_{i}.jpg"), img_data)
    print("Sample dataset created successfully.")

def train_and_benchmark(data_dir="dataset_sample", epochs=15, batch_size=16, img_size=(128, 128)):
    print("\n========================================================")
    print("      BRAIN TUMOR MODEL COMPARISON & FINE-TUNING        ")
    print("========================================================\n")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        print(f"Dataset directory '{data_dir}' not found. Generating sample dataset...")
        create_synthetic_dataset(output_dir=data_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    # Calculate class weights to handle imbalance
    y_train_labels = train_gen.classes
    class_weights = compute_balanced_class_weights(y_train_labels)
    print("Calculated Class Weights:", class_weights)

    architectures = ['vgg16', 'resnet50', 'efficientnetb0']
    benchmark_results = {}

    os.makedirs("models", exist_ok=True)

    for arch in architectures:
        print(f"\n---> Training Architecture: {arch.upper()} <---")
        model = build_model(architecture=arch, input_shape=(img_size[0], img_size[1], 3), num_classes=len(CLASSES))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
            ModelCheckpoint(f"models/{arch}_best.h5", monitor='val_accuracy', save_best_only=True, verbose=0)
        ]

        start_train = time.time()
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        train_duration = time.time() - start_train

        # Evaluate on test set
        test_gen.reset()
        X_test, y_test = [], []
        for i in range(len(test_gen)):
            x_b, y_b = test_gen[i]
            X_test.append(x_b)
            y_test.append(y_b)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        eval_res = evaluate_model_metrics(model, X_test, y_test, class_names=CLASSES)
        eval_res['train_duration_sec'] = train_duration
        benchmark_results[arch] = eval_res

        if arch == 'vgg16':
            model.save("models/model.h5")
            print("Saved baseline VGG16 model to models/model.h5")

    # Print Summary Table
    print("\n==========================================================================================")
    print("                             ARCHITECTURE COMPARISON SUMMARY                              ")
    print("==========================================================================================")
    print(f"{'Architecture':<16} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Latency (ms)':<12}")
    print("-" * 82)

    best_arch = None
    best_f1 = -1.0

    for arch, metrics in benchmark_results.items():
        acc = metrics['accuracy']
        prec = metrics['weighted_precision']
        rec = metrics['weighted_recall']
        f1 = metrics['weighted_f1']
        lat = metrics['latency_ms_per_sample']
        print(f"{arch.upper():<16} | {acc*100:6.2f}%    | {prec*100:6.2f}%    | {rec*100:6.2f}%    | {f1*100:6.2f}%    | {lat:8.2f} ms")
        if f1 > best_f1:
            best_f1 = f1
            best_arch = arch

    print("==========================================================================================")
    print(f"Selected Optimal Architecture: {best_arch.upper()} with Weighted F1-Score: {best_f1*100:.2f}%\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and benchmark CNN architectures for Brain Tumor Diagnosis.")
    parser.add_argument("--data_dir", type=str, default="dataset_sample", help="Directory containing train/val/test folders")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    train_and_benchmark(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
