import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, EfficientNetB2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Input
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']

def build_model(architecture='vgg16', input_shape=(128, 128, 3), num_classes=4, dropout_rate=0.3):
    """
    Builds and returns a specified deep learning architecture with a custom classification head.
    Supported architectures: 'vgg16', 'resnet50', 'efficientnetb0', 'efficientnetb2'
    """
    arch = architecture.lower().strip()
    inputs = Input(shape=input_shape)

    if arch == 'vgg16':
        base = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        x = Flatten(name='flatten')(base.output)
        x = Dropout(dropout_rate, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        outputs = Dense(num_classes, activation='softmax', name='dense_output')(x)
        model = Model(inputs=base.input, outputs=outputs, name='vgg16_brain_tumor')

    elif arch == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        x = GlobalAveragePooling2D(name='avg_pool')(base.output)
        x = Dropout(dropout_rate, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        outputs = Dense(num_classes, activation='softmax', name='dense_output')(x)
        model = Model(inputs=base.input, outputs=outputs, name='resnet50_brain_tumor')

    elif arch == 'efficientnetb0':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=inputs)
        x = GlobalAveragePooling2D(name='avg_pool')(base.output)
        x = Dropout(dropout_rate, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        outputs = Dense(num_classes, activation='softmax', name='dense_output')(x)
        model = Model(inputs=base.input, outputs=outputs, name='efficientnetb0_brain_tumor')

    elif arch == 'efficientnetb2':
        base = EfficientNetB2(weights='imagenet', include_top=False, input_tensor=inputs)
        x = GlobalAveragePooling2D(name='avg_pool')(base.output)
        x = Dropout(dropout_rate, name='dropout_1')(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        outputs = Dense(num_classes, activation='softmax', name='dense_output')(x)
        model = Model(inputs=base.input, outputs=outputs, name='efficientnetb2_brain_tumor')

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model

def get_target_conv_layer(model, architecture='vgg16'):
    """
    Returns the target final convolutional layer name for Grad-CAM / Grad-CAM++ calculation.
    """
    arch = architecture.lower().strip()
    if arch == 'vgg16':
        return 'block5_conv3'
    elif arch == 'resnet50':
        return 'conv5_block3_out'
    elif 'efficientnet' in arch:
        return 'top_conv'
    
    # Fallback: search for last Conv2D layer in model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError(f"Could not find a convolutional layer in model for architecture {architecture}")

def compute_balanced_class_weights(y_train):
    """
    Computes class weights to handle class imbalance in training set.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

def evaluate_model_metrics(model, X_test, y_test, class_names=CLASSES):
    """
    Evaluates model performance: Accuracy, Precision, Recall, F1-Score, and Inference Latency (ms).
    """
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time_ms = ((time.time() - start_time) / len(X_test)) * 1000.0

    y_pred = np.argmax(y_pred_probs, axis=1)
    if y_test.ndim > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    return {
        'accuracy': float(acc),
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1': float(f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'latency_ms_per_sample': float(inference_time_ms),
        'classification_report': report
    }
