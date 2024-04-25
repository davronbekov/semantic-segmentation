import yaml


def read_yaml(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.load(f, yaml.Loader)
    except Exception as ex:
        print(f'Cant read config from {config_path}. Reason: {str(ex)}')

    return None
