#! /usr/bin/env python
# -*- coding: utf-8 -*-
#####################################################
# Image-Text Matching Classifier: baseline system
#
# This program has been adapted and rewriten from sources such as:
# https://www.tensorflow.org/tutorials/images/cnn
# https://keras.io/api/layers/merging_layers/concatenate/
# https://pypi.org/project/sentence-transformers/
#
# If you are new to Tensorflow, read the following brief tutorial:
# https://www.tensorflow.org/tutorials/quickstart/beginner
#
# As you develop your experience and skills, you may want to check
# details of particular aspects of the Tensorflow API:
# https://www.tensorflow.org/api_docs/python/tf/all_symbols
#
# This is a binary classifier for image-text matching, where the inputs
# are images and text-based features, and the outputs (denoted as
# match=[1,0] and nomatch=[0,1]) correspond to (predicted) answers. This
# baseline classifier makes use of two strands of features. The first
# are produced by a CNN-classifier, and the second are derived offline
# from a sentence embedding generator. The latter have the advantage of
# being generated once, which can accelerate training due to being
# pre-trained and loaded at runtime. Those two strands of features are
# concatenated at trianing time to form a multimodal set of features,
# combining learnt image features and pre-trained sentence features.

# This program has been tested using an Anaconda environment with Python 3.9
# and 3.10 on Windows 11 and Linux Ubuntu 22. The easiest way to run this
# baseline at Uni is by booting your PC with Windows and using the following steps.
#
# Step 1=> Make sure that your downloaded data and baseline system are
#          extracted in the Downloads folder.
#          Note. Your path should start with /mnt/c/Users/Computing/Downloads
#
# Step 2=> Open a terminal and select Ubuntu from the little arrow pointing down
#          Note. Your program will be executed under a Linux environment.
#
# Step 3=> Install the following dependencies:
# pip install tf-models-official
# pip install tensorflow-text
# pip install einops
#
# Step 4=> Edit file ITM_Classifier-baseline.py and make sure that variable
# IMAGES_PATH points to the right folder containing the data.
#
# Step 5=> Run the program using a command such as
# $ python ITM_Classifier-baseline.py
#
#
# The code above can also be run from Visual Studio Code, to access it using the
# Linux envinment type "code ." in the Ubuntu terminal. From VSCode, click View, Terminal,
# type your command (example: python ITM_Classifier-baseline.py) and Enter.
#
# Running this baseline takes about 5 minutes minutes with a GPU-enabled Uni PC.
# WARNING: Running this code without a GPU is too slow and not recommended.
#
# In your own PC you can use Anaconda to run this code. From a conda terminal
# for example. If you want GPU-enabled execution, it is recommended that you
# install the following versions of software:
# CUDA 11.8
# CuDNN 8.6
# Tensorflow 2.10
#
# Feel free to use and/or modify this program as part of your CMP9137 assignment.
# You are invited to use the knowledge acquired during lectures, workshops
# and beyond to propose and evaluate alternative solutions to this baseline.
#
# Version 1.0, main functionality tested with COCO data
# Version 1.2, extended functionality for Flickr data
# Contact: {hcuayahuitl, lzhang, friaz}@lincoln.ac.uk
#####################################################


# Let's import the dependencies

import sys, re
import os
import time

# import einops
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt
from collections import Counter
from TextTransformerClassifier_Vanilla import TransformerBlock
import datetime

# Class for loading image and text data


class ITM_DataLoader:
    BATCH_SIZE = 16
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    SENTENCE_EMBEDDING_SHAPE = 384
    AUTOTUNE = tf.data.AUTOTUNE
    IMAGES_PATH = "flickr8k-resised"
    train_data_file = IMAGES_PATH + "/../flickr8k.TrainImages.txt"
    dev_data_file = IMAGES_PATH + "/../flickr8k.DevImages.txt"
    test_data_file = IMAGES_PATH + "/../flickr8k.TestImages.txt"
    sentence_embeddings_file = (
        IMAGES_PATH + "/../flickr8k.cmp9137.sentence_transformers.pkl"
    )
    sentence_embeddings = {}
    train_ds = None
    val_ds = None
    test_ds = None
    VOCAB_SIZE = 20000
    MAXLEN = 50

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.words, self.indexes = self.get_dictionary(
            self.train_data_file, self.VOCAB_SIZE
        )
        self.train_ds = self.load_classifier_data(self.train_data_file)
        self.val_ds = self.load_classifier_data(self.dev_data_file)
        self.test_ds = self.load_classifier_data(self.test_data_file)
        print("done loading data...")

    def get_dictionary(self, file_path, vocab_size):
        print("EXTRACTING dictionary from " + str(file_path))
        # step 1: generate a dictionary of words with their frequencies
        word_freqs = Counter()
        with open(file_path, "r") as f:
            for line in f:
                line = line.split("	")[1]
                line = re.sub(r"[^\w\s]", "", line)  # remove punctuation
                for word in line.split():
                    word_freqs[word] += 1

        # step 2: generate a set of word-index (and index-word) pairs -- the most popular words
        words = {}
        indexes = {}
        num_words = 1
        sorted_word_freqs = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_word_freqs:
            words[word] = num_words
            indexes[num_words] = word
            num_words += 1
            if num_words > vocab_size:
                break
        print("|words|=%s |indexes|=%s" % (len(words), len(indexes)))
        return words, indexes

    # Sentence embeddings are dense vectors representing text data, one vector per sentence.
    # Sentences with similar vectors would mean sentences with equivalent meaning.
    # They are useful here to provide text-based features of questions in the data.
    # Note: sentence embeddings don't include label info, they are solely based on captions.
    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, "rb") as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                # print("*sentence=",sentence)
                sentence_embeddings[sentence] = dense_vector
        print("Done reading sentence_embeddings!")
        return sentence_embeddings

    # In contrast to text-data based on pre-trained features, image data does not use
    # any form of pre-training in this program. Instead, it makes use of raw pixels.
    # Notes that input features to the classifier are only pixels and sentence embeddings.
    def process_input(self, img_path, dense_vector, text, label, sentence_vector):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        features = {}
        features["image_input"] = img
        features["text_embedding"] = dense_vector
        features["caption"] = text
        features["file_name"] = img_path
        features["sentence_vector"] = sentence_vector
        return features, label

    # This method loads the multimodal data, which comes from the following sources:
    # (1) image files in IMAGES_PATH, and (2) files with pattern flickr8k.*Images.txt
    # The data is stored in a tensorflow data structure to make it easy to use by
    # the tensorflow model during training, validation and test. This method was
    # carefully prepared to load the data rapidly, i.e., by loading already created
    # sentence embeddings (text features) rather than creating them at runtime.
    def load_classifier_data(self, data_files):
        print("LOADING data from " + str(data_files))
        print("=========================================")
        image_data = []
        text_data = []
        embeddings_data = []
        label_data = []
        sentence_vectors = []

        # get image, text, label of image_files
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("	")
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())
                image_data.append(img_name)

                # get binary labels from match/no-match answers
                label = [1, 0] if raw_label == "match" else [0, 1]
                label_data.append(label)
                # print("I=%s T=%s _L=%s L=%s" % (img_name, text, raw_label, label))

                # get sentence embeddings (of textual captions)
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)
                embeddings_data.append(text_sentence_embedding)

                sentence_vector = []
                text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
                words_in_text = text.split()
                for i in range(self.MAXLEN):
                    if i < len(words_in_text):
                        word = words_in_text[i]
                        index = self.words[word] if word in self.words else 0
                        sentence_vector.append(int(index))
                    else:
                        sentence_vector.append(0)  # zero-padding
                sentence_vectors.append(sentence_vector)

                text_data.append(text)

        print("|image_data|=" + str(len(image_data)))
        print("|text_data|=" + str(len(text_data)))
        print("|label_data|=" + str(len(label_data)))

        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_data, embeddings_data, text_data, label_data, sentence_vectors)
        )
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        self.print_data_samples(dataset)
        return dataset

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Caption: {features_batch["caption"].numpy()}')
                label = label_batch.numpy()[i]
                print(f"Label : {label}")
        print("-----------------------------------------")


# Main class for the Image-Text Matching (ITM) task


class ITM_Classifier(ITM_DataLoader):
    epochs = 10
    learning_rate = 3e-5
    class_names = {"match", "no-match"}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = "ITM_Classifier-flickr"
    simple_classifier = True
    attention_classifier = False

    def __init__(self, load_weights=None):
        super().__init__()
        self.build_classifier_model()
        if load_weights is not None:
            self.classifier_model.load_weights(load_weights)
        else:
            self.train_classifier_model()
        self.test_classifier_model()

    # return learnt feature representations of input data (images)
    def create_vision_encoder(
        self, num_projection_layers, projection_dims, dropout_rate, img_input=None
    ):
        if img_input is None:
            img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")

        cnn_layer = layers.Conv2D(16, 3, padding="same", activation="relu")(img_input)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(32, 3, padding="same", activation="relu")(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding="same", activation="relu")(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Dropout(dropout_rate)(cnn_layer)
        cnn_layer = layers.Flatten()(cnn_layer)
        outputs = self.project_embeddings(
            cnn_layer, num_projection_layers, projection_dims, dropout_rate
        )
        return img_input, outputs

    # return learnt feature representations based on dense layers, dropout, and layer normalisation
    def project_embeddings(
        self, embeddings, num_projection_layers, projection_dims, dropout_rate
    ):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    # return learnt feature representations of input data (text embeddings in the form of dense vectors)
    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        text_input = keras.Input(
            shape=self.SENTENCE_EMBEDDING_SHAPE, name="text_embedding"
        )
        outputs = self.project_embeddings(
            text_input, num_projection_layers, projection_dims, dropout_rate
        )
        return text_input, outputs

    # put together the feature representations above to create the image-text (multimodal) deep learning model
    def build_classifier_model(self):
        print(f"BUILDING model")

        data_augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(factor=0.02),
                keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            ],
            name="data_augmentation",
        )

        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        augmented_input = data_augmentation(img_input)

        _, vision_net = self.create_vision_encoder(
            num_projection_layers=1,
            projection_dims=128,
            dropout_rate=0.1,
            img_input=augmented_input,
        )

        text_input, text_net = self.create_text_encoder(
            num_projection_layers=1, projection_dims=128, dropout_rate=0.1
        )
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])

        if self.attention_classifier:
            ff_dim = 32
            embed_dim = 256
            rate = 0.1
            inputs = tf.expand_dims(net, -1)
            attn = layers.MultiHeadAttention(
                key_dim=embed_dim, num_heads=2, output_shape=256
            )

            net = attn(inputs, inputs, return_attention_scores=False)

            ffn = keras.Sequential(
                [
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ]
            )

            attn_output = layers.Dropout(rate)(net)
            out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
            ffn_output = ffn(out1)
            ffn_output = layers.Dropout(rate)(ffn_output)
            net = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
            # net = layers.GlobalAveragePooling1D(data_format="channels_first")(net)
            net = layers.Flatten()(net)

        net = tf.keras.layers.Dropout(0.1)(net)

        if not self.simple_classifier:
            net = tf.keras.layers.Dense(
                128, activation="relu", name="classifier_dense_1"
            )(net)
            net = tf.keras.layers.Dense(
                32, activation="relu", name="classifier_dense_2"
            )(net)
            net = tf.keras.layers.Dense(
                self.num_classes, activation="softmax", name="classifier_dense_3"
            )(net)
        else:
            net = tf.keras.layers.Dense(
                self.num_classes, activation="softmax", name=self.classifier_model_name
            )(net)

        self.classifier_model = tf.keras.Model(
            inputs=[img_input, text_input], outputs=net
        )
        self.classifier_model.summary()

    def train_classifier_model(self):
        print(f"TRAINING model")
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2 * num_train_steps)

        loss = keras.losses.KLDivergence()
        metrics = [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
        ]
        optimizer = optimization.create_optimizer(
            init_lr=self.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type="adamw",
        )

        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "logs/fit/" + self.classifier_model_name + "-" + timestamp
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        checkpoint_filepath = f"{log_dir}/epoch_{{epoch}}_checkpoint.model.keras"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            mode="max",
            save_best_only=False,
            save_freq="epoch",
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=11, restore_best_weights=True),
            tensorboard_callback,
            model_checkpoint_callback,
        ]

        self.history = self.classifier_model.fit(
            x=self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
        )
        print("model trained!")

    def test_classifier_model(self):
        print(
            "TESTING classifier model (showing a sample of image-text-matching predictions)..."
        )
        num_classifications = 0
        num_correct_predictions = 0

        # read test data for ITM classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            # captions = features["caption"].numpy()
            # file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_match = predicted_values[0]
                probability_nomatch = predicted_values[1]
                predicted_class = (
                    "[1 0]" if probability_match > probability_nomatch else "[0 1]"
                )
                if str(groundtruth[batch_index]) == predicted_class:
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions -- about 10% of all possible
                # if random.random() < 0.1:
                #     caption = captions[batch_index]
                #     file_name = file_names[batch_index].decode("utf-8")
                #     print(
                #         "ITM=%s PREDICTIONS: match=%s, no-match=%s \t -> \t %s"
                #         % (caption, probability_match, probability_nomatch, file_name)
                #     )

        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions / num_classifications
        print("TEST accuracy=%4f" % (accuracy))

        # reveal test performance using Tensorflow calculations
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f"Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}")


if __name__ == "__main__":
    # Let's create an instance of the main class
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    itm = ITM_Classifier()
