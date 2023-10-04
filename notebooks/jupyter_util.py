import importlib
import os

from aicrowd.dataset import download_dataset

from notebooks.config import get_config


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
    download_dataset(challenge='esci-challenge-for-improving-product-search', output_dir=data_dir, jobs=2, dataset_files=[id])


def get_file_path(dir_conf, file_name):
    data_dir = get_config(dir_conf)
    return os.path.join(data_dir, file_name)


if __name__ == '__main__':
    conditional_download('train-v0.3.csv.zip')