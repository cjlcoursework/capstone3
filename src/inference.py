import os
import subprocess

import tensorflow as tf
import numpy as np
import sklearn
# import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import pathlib
import io
import pandas as pd
import os
import re
import csv
import string
import time
from numpy import random
# import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
# from google.colab import drive
# from google.colab import files
from datasets import load_dataset
from transformers import AutoTokenizer, create_optimizer, TFAutoModel
from aicrowd.dataset import download_dataset

from _secrets.secret_vars import get_secret
# from _secrets.secret_vars import get_secret
from config import get_config, get_directory
from src.jupyter_util import get_file_path


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]

    input_mask_expanded = tf.cast(
        tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
        tf.float32
    )
    return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(
        tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)


class ProductEmbedding:
    def __init__(self, tf_model, max_length=64):
        self.model_path = get_directory('model_path')
        self.query_embedding_file = os.path.join(self.model_path, 'embeddings.npz')
        self.product_embedding_file = os.path.join(self.model_path, 'product_titles.npz')
        self.product_file = 'product_catalogue-v0.3.csv.zip'
        self.INFERENCE_BATCH_SIZE = 640
        self.MAX_LENGTH = max_length
        self.model = tf_model

    def embedding_exists(self):
        return os.path.exists(self.query_embedding_file) and os.path.exists(self.product_embedding_file)

    def build_product_titles(self):
        if self.embedding_exists():
            print("Embedding files already exist, no need to remake them....")
            return

        print("Creating product embeddings")
        product_file = get_file_path('raw_data', self.product_file)
        df_catalogue = pd.read_csv(product_file, compression='zip')
        product_titles = [str(df_catalogue['product_title'][i]) for i in range(len(df_catalogue))]

        model_id = get_config('model_id')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        embeddings = []

        for i in range(len(product_titles) // self.INFERENCE_BATCH_SIZE):
            tokenized_output = tokenizer(
                product_titles[self.INFERENCE_BATCH_SIZE * i:self.INFERENCE_BATCH_SIZE * (i + 1)],
                max_length=self.MAX_LENGTH,
                padding='max_length', truncation=True, return_tensors="tf")
            model_output = self.model(tokenized_output)
            embedding = mean_pooling(model_output, tokenized_output['attention_mask'])
            embeddings.append(embedding)
            if i % 100 == 0:
                print(i)

        print("Saving product embeddings")
        np.savez_compressed(self.query_embedding_file, embeddings)
        np.savez_compressed(self.product_embedding_file, product_titles)
        return


class InferenceEngine:
    def __init__(self, tf_model, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.tf_model = tf_model
        self.model_path = get_directory('model_path')
        self.model_id = get_directory('model_id')

        # always check the product embeddings or build them
        self.loaded_titles_array = None
        self.loaded_embedding_array = None
        self.max_length = max_length

        self.query_embedding_file = os.path.join(self.model_path, 'embeddings.npz')
        self.product_embedding_file = os.path.join(self.model_path, 'product_titles.npz')

        api_key = get_secret('api_key')
        os.environ["AICROWD_API_KEY"] = api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # self.api_login()
        # self.load_model()
        self.load_embeddings()

    # def api_login(self):
    #     subprocess.run(['aicrowd', 'login'], check=True, text=True)
    #
    # def load_model(self):
    #     weights_file = os.path.join(self.model_path, 'model_weights.h5')
    #
    #     self.tf_model = TFAutoModel.from_pretrained(self.model_id)
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    #     self.tf_model.load_weights(weights_file)

    def load_embeddings(self):
        product_embedding = ProductEmbedding(tf_model=self.tf_model, max_length=self.max_length)
        product_embedding.build_product_titles()

        loaded_embedding = np.load(self.query_embedding_file)
        loaded_embedding_array = np.array(loaded_embedding['arr_0'])
        self.loaded_embedding_array = loaded_embedding_array.reshape(-1, loaded_embedding_array.shape[2])

        loaded_titles = np.load(self.product_embedding_file)
        self.loaded_titles_array = np.array(loaded_titles['arr_0'])

    def query(self, sentence, top=25):
        inputs = self.tokenizer([sentence], max_length=self.max_length, padding='max_length', truncation=True,
                                return_tensors="tf")

        logits = self.tf_model(**inputs)
        out_embedding = mean_pooling(logits, inputs['attention_mask'])
        print(out_embedding.shape)

        u_dot_v = np.matmul(self.loaded_embedding_array, np.array(out_embedding).T)
        v_magnitude = np.sqrt(np.sum(out_embedding * out_embedding, axis=-1))
        u_magnitude = np.sqrt(np.sum(self.loaded_embedding_array * self.loaded_embedding_array, axis=-1))

        cosine_similarity = u_dot_v.T / (u_magnitude * v_magnitude)
        sorted_indices = np.argsort(cosine_similarity, axis=-1)

        top_results = []
        for i in range(top):
            r = (i, self.loaded_titles_array[sorted_indices[:, len(sorted_indices[0]) - i - 1]])
            top_results.append(r)
        return top_results


if __name__ == '__main__':
    inference_engine = InferenceEngine()
    results = inference_engine.query("a blue hb pencil", top=5)
    for r in results:
        print(r)