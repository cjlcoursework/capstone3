import importlib
import os

import tensorflow as tf
import matplotlib.pyplot as plt
# import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import pathlib
import pandas as pd
import os
import csv
import time
# import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Flatten, InputLayer, BatchNormalization, Dropout
# from google.colab import drive
# from google.colab import files
from datasets import load_dataset


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]

    input_mask_expanded = tf.cast(
        tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
        tf.float32
    )
    return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(
        tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)


class TransformerModel(tf.keras.Model):
    def __init__(self, model):
        super(TransformerModel, self).__init__()
        self.loss_metric = None
        self.loss_fn = None
        self.optimizer = None
        self.model = model
        self.dense = Dense(1, activation='sigmoid')

    def compile(self, optimizer, loss):
        super(TransformerModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss
        self.loss_metric = tf.keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, train_data):
        query = {'input_ids': train_data['input_ids_query'],
                 'token_type_ids': train_data['token_type_ids_query'],
                 'attention_mask': train_data['attention_mask_query']}

        product = {'input_ids': train_data['input_ids_product'],
                   'token_type_ids': train_data['token_type_ids_product'],
                   'attention_mask': train_data['attention_mask_product']}

        labels = train_data['label']

        with tf.GradientTape() as recorder:
            query_predictions = self.model(query)
            pred_query = mean_pooling(query_predictions, train_data['attention_mask_query'])

            product_predictions = self.model(product)
            pred_product = mean_pooling(product_predictions, train_data['attention_mask_product'])

            pred_concat = tf.concat([pred_query, pred_product, tf.abs(pred_query - pred_product)], axis=-1)

            predictions = self.dense(pred_concat)
            loss = self.loss_fn(labels, predictions)

        partial_derivatives = recorder.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(partial_derivatives, self.model.trainable_weights))

        self.loss_metric.update_state(loss)

        return {'loss': self.loss_metric.result()}
