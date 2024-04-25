from src.losses.dice_loss import DiceLoss
from src.losses.jaccard_loss import JaccardLoss


class Loss:
    def __init__(self, adapter: str, loss_fn: str):
        self.loss_fn = None
        if loss_fn == 'dice_loss':
            self.loss_fn = DiceLoss.get_instance(adapter=adapter)
        elif loss_fn == 'jaccard_loss':
            self.loss_fn = JaccardLoss.get_instance(adapter=adapter)

        if not self.loss_fn:
            raise Exception(f'{adapter} adapter at Loss not found!')

    def get_loss_fn(self):
        return self.loss_fn

    @staticmethod
    def get_instance(adapter: str, loss_fn: str):
        return Loss(adapter=adapter, loss_fn=loss_fn).get_loss_fn()
