import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.randn((self.n_features, self.n_classes), requires_grad=False)
        self.weights = weight_std * self.weights 
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        current_predictions = y == y_pred
        acc = torch.mean(current_predictions.float())
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            
            if epoch_idx == 0:
                train_samples_num = 0
                valid_samples_num = 0
            
            total_correct_valid = 0
            total_loss_valid = 0
            for idx, (X, y) in enumerate(dl_valid):
                y_hat, x_scores = self.predict(X)
                loss = loss_fn.loss(X, y, x_scores, y_hat)
                loss += weight_decay / 2 * torch.sum(torch.square(self.weights))
                grad = loss_fn.grad()
                
                total_correct_valid += torch.sum(y==y_hat)
                total_loss_valid += loss
                
                if epoch_idx == 0:
                    valid_samples_num += X.shape[0]

            valid_res.loss.append(total_loss_valid / valid_samples_num)
            valid_res.accuracy.append(total_correct_valid / valid_samples_num)
            
            total_correct_train = 0
            total_loss_train = 0
            for idx, (X, y) in enumerate(dl_train):
                if epoch_idx == 0:
                    train_samples_num += X.shape[0]
                y_hat, x_scores = self.predict(X)
                loss = loss_fn.loss(X, y, x_scores, y_hat)
                loss += weight_decay / 2 * torch.sum(torch.square(self.weights))
                grad = loss_fn.grad()
                
                self.weights = (1 - weight_decay) * self.weights
                self.weights = self.weights - learn_rate * grad
                
                total_correct_train += torch.sum(y==y_hat)
                total_loss_train += loss
                
            train_res.loss.append(total_loss_train / train_samples_num)
            train_res.accuracy.append(total_correct_train / train_samples_num)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        shape = tuple([self.n_classes] + list(img_shape))
        if has_bias:
            w_images = torch.reshape(self.weights[1:], shape)
        else:
            w_images = torch.reshape(self.weights , shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp["weight_std"] = 0.01
    hp["learn_rate"] = 0.003
    hp["weight_decay"] = 0.0001
    # ========================

    return hp
