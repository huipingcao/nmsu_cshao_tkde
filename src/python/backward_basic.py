import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def backward_basic(train_x, train_y, test_x, test_y):
    """
    This function implements the backward feature selection algorithm based on decision tree

    Input
    -----
    train_x: {2d numpy array matrix}, shape (n_samples, n_features)
        input data
    train_y: {1d numpy array vector}, shape (n_samples,)
        input class labels
    test_x: {2d numpy array matrix}, shape (n_samples, n_features)
        input data
    test_y: {1d numpy array vector}, shape (n_samples,)
        input class labels
    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = train_x.shape

    model = DecisionTreeClassifier()

    # selected feature set, initialized to contain all features
    F = range(n_features)
    count = n_features

    model.fit(train_x, train_y)
    y_predict = model.predict(test_x)
    acc_all = accuracy_score(test_y, y_predict)

    f_score = []

    for i in range(n_features):
        if i in F:
            F.remove(i)
            train_x_tmp = train_x[:, F]
            test_x_tmp = test_x[:, F]
            acc = 0

            model.fit(train_x_tmp, train_y)
            y_predict = model.predict(test_x_tmp)
            acc_tmp = accuracy_score(test_y, y_predict)
            f_score.append(acc_all - acc_tmp)
            F.append(i)

    return np.array(f_score)


