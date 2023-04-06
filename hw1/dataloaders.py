import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data so4urce is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        return FirstLastSamplerIterator(len(self.data_source))
        # ========================

    def __len__(self):
        return len(self.data_source)

class FirstLastSamplerIterator(Iterator[int]):
    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N
        self.current = -1
        
    def __next__(self):
        self.current += 1
        if self.current >= self.N:
            raise StopIteration
        if self.current % 2 == 0:
            return int(self.current / 2)
        else:
            return self.N - 1 - int(self.current / 2)

    
def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    num_samples = len(dataset)
    validation_size = int(num_samples * validation_ratio)
    train_size = num_samples - validation_size
    train_sampler = torch.utils.data.SubsetRandomSampler(range(0, train_size))
    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler,
    )
    valid_sampler = torch.utils.data.SubsetRandomSampler(range(train_size, num_samples))
    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        sampler=valid_sampler,
    )
    # ========================

    return dl_train, dl_valid
    
class ByIndexSampler(Sampler):
    def __init__(self, data_source: Sized ,indices_list: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.indices_list = indices_list

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices_list)
    
    
    