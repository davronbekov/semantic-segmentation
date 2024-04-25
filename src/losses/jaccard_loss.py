from segmentation_models_pytorch.losses import BINARY_MODE as SMP_BINARY_MODE, JaccardLoss as SmpJaccardLoss


class JaccardLoss:
    def __init__(self, adapter: str):
        self.adapter = None
        if adapter == 'smp':
            self.adapter = SmpJaccardLoss(mode=SMP_BINARY_MODE, from_logits=True)

        if not self.adapter:
            raise Exception(f'{type} adapter at JaccardLoss not found!')

    def get_adapter(self):
        return self.adapter

    @staticmethod
    def get_instance(adapter: str):
        return JaccardLoss(adapter=adapter).get_adapter()


