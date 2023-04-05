import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders
from .dataloaders import ByIndexSampler


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train_list = []
        y_train_list = []
        for samples, labels in dl_train:
            x_train_list.append(samples)
            y_train_list.append(labels)
        x_train = torch.cat(x_train_list, dim = 0)
        y_train = torch.cat(y_train_list, dim = 0)
        n_classes = len(y_train)
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            ith_col = dist_matrix[:, i]
            values, indices = torch.topk(ith_col, self.k, largest = False)
            nearest_n = self.y_train[indices]
            y_pred[i], count = torch.mode(nearest_n, 0)
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # ====== YOUR CODE: ======
    # row_mat = x1
    # column_mat = x2.transpose(0, 1)
    # print(row_mat[:, :, None].shape)
    # print(column_mat[None, : :].shape)
    # diff = row_mat[:, :, None] - column_mat[None, : :]
    # squared = diff ** 2
    # sum = torch.sum(squared, dim = 1)
    # dists = torch.sqrt(sum)
    dists = torch.cdist(x1, x2)
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    equal_cells = torch.eq(y, y_pred)
    equal_cells_float = equal_cells.float()
    accuracy = torch.mean(equal_cells_float).item()
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        fold_accuracy = []
        fold_size = len(ds_train) / num_folds
        
        
        for fold_index in range(num_folds):
            val_start = fold_index * fold_size
            val_end = val_start + fold_size
            val_indices = list(set(range(int(val_end))) - set(range(int(val_start))))
            train_indices = list(set(range(len(ds_train))) - set(val_indices))
            
            #sampler defined in dataloaders.py
            val_sampler = ByIndexSampler(ds_train, val_indices)
            train_sampler = ByIndexSampler(ds_train, train_indices)
            
            val_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=1024, num_workers = 2, sampler = val_sampler)
            train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=1024, num_workers = 2, sampler = train_sampler)
            
            model.train(train_dataloader)
            
            test_data = []
            for (batch, labels) in val_dataloader:
                test_data.append(batch)
            test_data = torch.cat(test_data, dim = 0)
            print(test_data.shape)
            y_pred = model.predict(test_data)
            x_val, y_val = dataloader_utils.flatten(val_dataloader)
            current_accuracy = accuracy(y_val, y_pred)
            fold_accuracy.append(current_accuracy)
            
        # ========================

    accuracies.append(fold_accuracy)
    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies


