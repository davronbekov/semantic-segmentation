from segmentation_models_pytorch.losses import BINARY_MODE as SMP_BINARY_MODE, DiceLoss as SmpDiceLoss


class DiceLoss:
    def __init__(self, adapter: str):
        self.adapter = None
        if adapter == 'smp':
            self.adapter = SmpDiceLoss(mode=SMP_BINARY_MODE, from_logits=True)

        if not self.adapter:
            raise Exception(f'{type} adapter at DiceLoss not found!')

    def get_adapter(self):
        return self.adapter

    @staticmethod
    def get_instance(adapter: str):
        return DiceLoss(adapter=adapter).get_adapter()


