import os.path
import sys

if 'google.colab' in sys.modules:
    root_dir = '/content/drive/MyDrive/projects/capstone3'
else:
    root_dir = os.path.abspath('..')


# root_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
root_model = "sentence-transformers/all-MiniLM-L6-v2"

configs = {
    "model_id": root_model,
    "model_path": os.path.join(root_dir, 'data', 'model'),
    "data": os.path.join(root_dir, 'data'),
    "raw_data": os.path.join(root_dir, 'data', 'raw'),
    "prep_data": os.path.join(root_dir, 'data', 'prep'),
    "project_files": {
        'train-v0.3.csv.zip': 3,
        'test_public-v0.3.csv.zip': 2,
        'product_catalogue-v0.3.csv.zip': 0,
        'sample_submission_public-v0.3.csv.zip': 1
    }
}


def get_directory(key, mkdir=True):
    if key in configs:
        dir = configs[key]
        os.makedirs(dir, exist_ok=True)
        return dir
    else:
        raise Exception(f"no configuration found for key: {key}")


def get_config(key, default=None):
    if key in configs:
        return configs[key]
    else:
        print(f"no configuration found for key: {key}")
        return default
