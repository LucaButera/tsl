from torch_geometric.transforms import BaseTransform

from tsl.data import Data


class MaskInput(BaseTransform):
    """Whiten masked values in :attr:`input_key`."""
    def __init__(self, input_key: str = 'x', mask_key: str = 'input_mask'):
        self.input_key = input_key
        self.mask_key = mask_key

    def __call__(self, data: Data) -> Data:
        data[self.input_key] *= data[self.mask_key]
        return data