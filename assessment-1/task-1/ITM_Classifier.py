#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import os
from vit_keras import vit

from ITM_Classifier_baseline import ITM_Classifier


class ITM_Classifier_ConvNeXt(ITM_Classifier):
    def __init__(self):
        super().__init__()

    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate
    ):
        """
        return learnt feature representations of input data (images)
        """
        # Create the base model from the pre-trained model MobileNet V2
        # preprocess_input = tf.keras.applications.ConvNeXtXLarge.preprocess_input
        base_model = tf.keras.applications.ConvNeXtBase(
            input_shape=self.IMAGE_SHAPE, include_top=False, weights="imagenet"
        )
        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")

        # x = preprocess_input(img_input)
        x = img_input
        x = base_model(x, training=False)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Flatten()(x)
        outputs = self.project_embeddings(
            x, num_projection_layers, projection_dims, dropout_rate
        )

        return img_input, outputs


INPUT_SHAPE = (224, 224, 3)
IMAGE_SIZE = 72
PATCH_SIZE = 9
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

TRANSFORMER_LAYERS = 4
NUM_HEADS = 8
PROJECTION_DIM = 64
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]
MLP_HEAD_UNITS = [256, 128]
LAYER_NORM_EPS = 1e-6

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"


class Patches(keras.layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(keras.layers.Layer):
    def __init__(self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = keras.layers.Dense(units=projection_dim)
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def create_classifier_model():
    inputs = keras.layers.Input(shape=INPUT_SHAPE)
    resized_inputs = keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(inputs)
    patches = Patches()(resized_inputs)
    encoded_patches = PatchEncoder()(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = keras.layers.Add()([attention_output, encoded_patches])
        x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        encoded_patches = keras.layers.Add()([x3, x2])

    representation = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = keras.layers.Flatten()(representation)
    representation = keras.layers.Dropout(0.5)(representation)

    model = keras.Model(inputs=inputs, outputs=representation)

    return model


class BiTModel(keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = hub.load("https://tfhub.dev/google/bit/m-r50x1/1")

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


class ITM_Classifier_ViT(ITM_Classifier):
    def __init__(self, BiT=False, ViT=False):
        self.BiT = BiT
        self.ViT = ViT
        super().__init__()

    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate
    ):
        """
        return learnt feature representations of input data (images)
        """
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")

        if self.BiT:
            module = hub.KerasLayer("https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r101x1/versions/1")
            x = module(img_input)

        elif self.ViT:
            vit_model = vit.vit_b32(
                image_size = INPUT_SHAPE[0],
                activation = 'softmax',
                pretrained = True,
                include_top = False,
                pretrained_top = False,
                classes = 2
            )
            # resized_inputs = keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(img_input)
            x = vit_model(img_input, training=False)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Flatten()(x)

        else:
            base_model = create_classifier_model()

            x = img_input
            x = base_model(x, training=True)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Flatten()(x)

        outputs = self.project_embeddings(
            x, num_projection_layers, projection_dims, dropout_rate
        )

        return img_input, outputs


if __name__ == "__main__":
    # classifier = ITM_Classifier_ConvNeXt()
    classifier = ITM_Classifier_ViT(ViT=True)
