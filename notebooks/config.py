import os.path

configs = {
    "raw_data": os.path.abspath('../data/raw'),
    "prep_data": os.path.abspath('../data/prep'),
    "project_files": {
        'train-v0.3.csv.zip': 3,
        'test_public-v0.3.csv.zip': 2,
        'product_catalogue-v0.3.csv.zip': 0,
        'sample_submission_public-v0.3.csv.zip': 1
    }
}


def get_directory(key):
    if key in configs:
        dir = configs[key]
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    else:
        raise Exception(f"no configuration found for key: {key}")


def get_config(key, default=None):
    if key in configs:
        return configs[key]
    else:
        print(f"no configuration found for key: {key}")
        return default
