#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import os
from vit_keras import vit

from ITM_Classifier_baseline import ITM_Classifier
from TextTransformerClassifier_Vanilla import (
    TransformerBlock,
    TokenAndPositionEmbedding,
)
from TextTransformerClassifier_BERT import map_name_to_handle, map_model_to_preprocess

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


class ITM_Classifier_Advanced(ITM_Classifier):
    def __init__(
        self,
        image_encoder,
        trainable=False,
        epochs=10,
        text_encoder="direct_embedding",
        load_weights=None,
    ):
        self.image_encoder = image_encoder.lower()
        self.text_encoder = text_encoder.lower()
        self.trainable = trainable
        self.epochs = epochs
        self.classifier_model_name = f"ITM_Classifier_Advanced_{image_encoder}_{text_encoder}_trainable_{trainable}_epochs_{epochs}"
        self.epochs = epochs
        super().__init__(load_weights)

    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate, img_input=None
    ):
        """
        return learnt feature representations of input data (images)
        """
        if img_input is None:
            img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")

        if self.image_encoder == "bit":
            module = hub.KerasLayer(
                "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r101x1/versions/1"
            )
            module.trainable = self.trainable
            x = module(img_input)

        elif self.image_encoder == "vit":
            vit_model = vit.vit_b32(
                image_size=INPUT_SHAPE[0],
                activation="softmax",
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=2,
            )
            vit_model.trainable = self.trainable
            # resized_inputs = keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)(img_input)
            x = vit_model(img_input, training=self.trainable)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Flatten()(x)

        elif self.image_encoder == "convnext":
            image_encoder = tf.keras.applications.ConvNeXtBase(
                input_shape=self.IMAGE_SHAPE, include_top=False, weights="imagenet"
            )
            image_encoder.trainable = self.trainable

            x = image_encoder(img_input, training=self.trainable)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Flatten()(x)

        elif self.image_encoder == "vanilla_vit":
            image_encoder = create_classifier_model()

            x = image_encoder(img_input, training=True)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Flatten()(x)

        elif self.image_encoder == "baseline":
            return super().create_vision_encoder(
                num_projection_layers, projection_dims, dropout_rate, img_input
            )

        else:
            raise ValueError(
                "image_encoder must be one of 'bit', 'vit', 'convnext', 'vanilla_vit' or 'baseline'"
            )

        outputs = self.project_embeddings(
            x, num_projection_layers, projection_dims, dropout_rate
        )

        return img_input, outputs

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        if self.text_encoder == "direct_embedding":
            return super().create_text_encoder(
                num_projection_layers, projection_dims, dropout_rate
            )
        elif self.text_encoder == "vanilla_transformer":
            embed_dim = 64  # Embedding size for each token
            num_heads = 2  # Number of attention heads
            ff_dim = 32  # Hidden layer size in feed forward network inside transformer
            vocab_size = 20000
            maxlen = 50

            text_input = layers.Input(shape=(maxlen,), name="sentence_vector")
            embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
            x = embedding_layer(text_input)
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
            x = transformer_block(x)
            x = layers.GlobalAveragePooling1D()(x)

            outputs = self.project_embeddings(
                x, num_projection_layers, projection_dims, dropout_rate
            )

            print("outputs shape: ", outputs.shape)
            return text_input, outputs
        elif self.text_encoder == "bert":
            bert_model_name = "small_bert/bert_en_uncased_L-4_H-512_A-8"
            tfhub_handle_encoder = map_name_to_handle[bert_model_name]
            tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

            text_input = keras.layers.Input(shape=(), dtype=tf.string, name="caption")
            encoder_inputs = hub.KerasLayer(
                tfhub_handle_preprocess, name="preprocessing"
            )(text_input)
            encoder = hub.KerasLayer(
                tfhub_handle_encoder, trainable=self.trainable, name="BERT_encoder"
            )
            outputs = encoder(encoder_inputs)
            net = outputs["pooled_output"]  # take the pooled output of the BERT model
            net = layers.Dropout(0.1)(net)
            outputs = self.project_embeddings(
                net, num_projection_layers, projection_dims, dropout_rate
            )

            return text_input, outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_encoder",
        type=str,
        default="bit",
        help="base model to use for the image encoder",
        choices=["bit", "vit", "convnext", "vanilla_vit", "baseline"],
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="direct_embedding",
        help="base model to use for the text encoder",
        choices=["vanilla_transformer", "direct_embedding", "bert"],
    )
    parser.add_argument(
        "--trainable",
        action="store_true",
        help="whether to train the base model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train for"
    )
    parser.add_argument(
        "--load_weights",
        type=str,
        default=None,
        help="directory to load the model weights",
    )

    args = parser.parse_args()

    classifier = ITM_Classifier_Advanced(
        image_encoder=args.image_encoder,
        trainable=args.trainable,
        epochs=args.epochs,
        text_encoder=args.text_encoder,
        load_weights=args.load_weights,
    )
