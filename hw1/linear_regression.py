import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.matmul(X, self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        N = len(y)
        K = np.matmul(X.T, X) + (self.reg_lambda * N * np.identity(X.shape[1]))
        K[0, 0] = K[0, 0] - N * self.reg_lambda
        w_opt = np.matmul(np.linalg.inv(K), np.matmul(X.T, y))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    if(feature_names):
        X = (df[feature_names]).to_numpy()
    else:
        X = (df.loc[ : , df.columns != target_name]).to_numpy()
    y = (df[target_name]).to_numpy()
    y_pred = model.fit_predict(X, y)
    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        ones = np.ones((X.shape[0], 1))
        xb = np.concatenate((ones, X), axis=1)
        # ========================

        return xb

class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        rm = X[:, 6]
        ptratio = X[:,11]
        b = X[:,12]
        nox = X[:,5]
        tax = X[:,10]
        lstat = X[:, 13]
        #print(X[:1,:])
        feature0 = X[:, 0]
        feature1 = np.power(lstat, -0.5)
        feature2 = np.power(rm, 2) * np.power(ptratio + 0.01, -1)
        feature3 = np.power(rm, 3) * np.log(b + 0.01)
        feature4 = np.power(nox * ptratio + 0.01, -1)
        feature5 = np.power(tax, -0.5)
        features = [feature1, feature2, feature3, feature4, feature5]
        for index1 in range(1, 14):
            for index2 in range(index1, 14):
                features.append(X[:, index1] * X[:, index2])
        for index1 in range(1, 14):
            for index2 in range(index1, 14):
                for index3 in range(index2, 14):
                    features.append(X[:, index1] * X[:, index2]  * X[:, index3])
        for i, feature in enumerate(features):
            features[i] = feature.reshape(-1, 1)
        features.append(X)
        features.append(np.power(X + 0.01, -1))
        features.append(np.log(X + 0.01))

        X_transformed = np.concatenate(features, axis = 1)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    y = df[target_feature]
    mu_y = y.mean()
    temp_y = y - mu_y
    temp_y = temp_y * temp_y
    std_y = temp_y.sum() ** 0.5
    
    pearsons_r_list = []
    features = []
    for feature in df:
        if feature == target_feature:
            continue
        x = df[feature]
        mu_x = x.mean()
        temp_x = x - mu_x
        temp_x = temp_x * temp_x
        std_x = temp_x.sum() ** 0.5
        
        temp_xy = (x - mu_x) * (y - mu_y)
        cov_xy = temp_xy.sum()
        pearsons_r = cov_xy / (std_x * std_y)
        pearsons_r_list.append(pearsons_r)
        features.append(feature)
    
    features = sorted(features, reverse=True ,key=lambda f: abs(pearsons_r_list[features.index(f)]))
    pearsons_r_list = sorted(pearsons_r_list, reverse=True, key=abs)
    
    top_n_features = features[:n]
    top_n_corr = pearsons_r_list[:n]
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """
    
    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    diffs = y - y_pred
    diffs = diffs * diffs
    mse = diffs.sum() / len(y)
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    residuls = y - y_pred
    residuls = residuls * residuls
    y_avg = sum(y) / len(y)
    avg_diff = y - y_avg
    avg_diff = avg_diff * avg_diff
    r2 = 1 - residuls.sum() / avg_diff.sum()
    # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return best_params
