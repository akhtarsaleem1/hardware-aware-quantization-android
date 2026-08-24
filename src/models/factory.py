"""Comparable ImageNet-initialized Keras classifiers with embedded preprocessing."""

from __future__ import annotations


SUPPORTED_MODELS = ("mobilenet_v2", "mobilenet_v3_small", "efficientnet_b0")


def build_classifier(tf, model_name: str, class_count: int = 9, image_size: int = 224):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")
    inputs = tf.keras.Input((image_size, image_size, 3), dtype=tf.float32, name="image")
    if model_name == "mobilenet_v2":
        prepared = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )
    elif model_name == "mobilenet_v3_small":
        prepared = inputs
        backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling="avg",
            include_preprocessing=True,
        )
    else:
        prepared = inputs
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3),
            pooling="avg",
        )
    features = backbone(prepared, training=False)
    features = tf.keras.layers.Dropout(0.2, name="head_dropout")(features)
    logits = tf.keras.layers.Dense(class_count, name="logits")(features)
    outputs = tf.keras.layers.Activation("softmax", dtype="float32", name="probabilities")(logits)
    model = tf.keras.Model(inputs, outputs, name=f"deepweeds_{model_name}")
    return model, backbone
