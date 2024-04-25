import segmentation_models_pytorch as smp_pytorch


class Smp:
    def __init__(self, **kwargs):
        self.model = smp_pytorch.create_model(
            arch=kwargs['arch'],
            encoder_name=kwargs['encoder_name'],
            encoder_weights=kwargs['encoder_weights'],

            in_channels=kwargs['in_channels'],
            classes=kwargs['out_channels'],

            activation=kwargs['activation']
        )

    def __new__(cls, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(**kwargs)
        return instance.model

