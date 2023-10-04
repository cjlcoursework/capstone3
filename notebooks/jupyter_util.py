import importlib


def is_module_installed(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


