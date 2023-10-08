import importlib
import os
import subprocess

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
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
from tensorflow.keras.layers import Dense, Flatten, InputLayer, BatchNormalization, Dropout, Input, LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy, TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
# from google.colab import drive
# from google.colab import files
from datasets import load_dataset
from transformers import AutoTokenizer, create_optimizer, TFAutoModel
from aicrowd.dataset import download_dataset

# from _secrets.secret_vars import get_secret
from config import get_config, get_directory




def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def conditional_download(file_path):
    data_dir = get_config('raw_data')
    full_path = os.path.join(data_dir, file_path)
    if os.path.exists(full_path):
        print(f"File exists:  {full_path}")
        return

    data_files = get_config('project_files')
    if file_path not in data_files:
        raise Exception(f"This project does not have a file with name {file_path}")

    id = data_files[file_path]
    download_dataset(challenge='esci-challenge-for-improving-product-search', output_dir=data_dir, jobs=2,
                     dataset_files=[id])


def get_file_path(dir_conf, file_name):
    data_dir = get_config(dir_conf)
    return os.path.join(data_dir, file_name)


