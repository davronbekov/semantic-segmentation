from src.models.smp import Smp
from src.config_reader import read_yaml
import torch


class Model:
    def __init__(self, **kwargs):
        conf = read_yaml('configs/model.yaml')
        self.model = None
        if conf['adapter'] == 'smp':
            self.model = Smp(**conf, **kwargs)

        if not self.model:
            raise Exception(f'{conf["adapter"]} at Model is not found!')

        if conf['path']:
            self.model.load_state_dict(torch.load(conf['path'], map_location=conf['map_location']))

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(**kwargs)
        return instance.model

