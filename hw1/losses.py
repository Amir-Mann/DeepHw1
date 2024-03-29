import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        y=y.reshape(-1, 1)
        zeros_N_C = torch.zeros(size=x_scores.shape, dtype=torch.float32)
        zeros_N_1 = torch.zeros(size=y.shape, dtype=torch.float32)
        ones_N_1 = torch.ones_like(y, dtype=torch.float32)
        y_onehots = torch.scatter(zeros_N_C, dim=1, index=y, src=ones_N_1)
        truth_scores = torch.sum(x_scores * y_onehots, dim=1).reshape(-1, 1)
        M = x_scores - truth_scores + self.delta
        M = torch.scatter(M, dim=1, index=y, src=zeros_N_1)
        M = torch.maximum(M, zeros_N_C)
        loss = torch.sum(M)
        loss = loss / x.shape[0]
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx["M"] = M
        self.grad_ctx["y_onehots"] = y_onehots
        self.grad_ctx["x"] = x
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        G = self.grad_ctx["M"]
        G[self.grad_ctx["M"] != 0] = 1
        dLi_wrt_dWyj = - self.grad_ctx["y_onehots"] * torch.reshape(torch.sum(G, dim=1), (-1, 1))
        G = G + dLi_wrt_dWyj
        grad = torch.matmul(self.grad_ctx["x"].T, G)
        grad = grad / self.grad_ctx["x"].shape[0]
        # ========================

        return grad
