#! /usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ITM_Classifier_baseline import ITM_Classifier



class ITM_Classifier_ConvNeXt(ITM_Classifier):
    def __init__(self):
        super().__init__()

    # return learnt feature representations of input data (images)
    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate
    ):
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

if __name__ == "__main__":
    classifier = ITM_Classifier_ConvNeXt()