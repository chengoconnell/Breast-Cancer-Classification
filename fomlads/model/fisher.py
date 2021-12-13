import numpy as np

from fomlads.model.metrics import accuracy, recall, precision, ConfusionMatrix
from fomlads.plot.fisher_plots import histogram_projected_data


def fit_and_predict_fisher(x_crossval, y_crossval, inputs_test, targets_test, classes):
    """
    Using cross-validation, fits a Fisher's Linear Discriminant model and uses it to make predictions on test data.

    :param x_crossval: an array of training inputs data, split for cross validation
    :param y_crossval: an array of training targets data, split for cross validation
    :param inputs_test: a numpy array of test inputs, used for making predictions
    :param targets_test: a numpy array of test targets, only used for plotting and not in the predictions process
    :param classes: the classes of the dataset

    :return: a numpy array of predictions for the test inputs
    """
    # Initialize model values
    # Loop through cross-validation splits and find the optimal model parameters
    # (those which maximize the evaluation metric defined below)
    max_metric, threshold, weights = 0, None, None
    for i in range(len(x_crossval)):
        x_train = x_crossval[i]["train"]
        x_valid = x_crossval[i]["valid"]
        y_train = y_crossval[i]["train"]
        y_valid = y_crossval[i]["valid"]

        if len(classes) == 2:
            # Get projection weights, find the best threshold and generate predictions for this fold
            weights_fold = get_projection_weights_fisher(x_train, y_train)
            best_threshold = find_best_threshold(x_train, y_train, weights_fold)
            if best_threshold is None:
                continue
            y_pred = get_predictions(x_valid, weights_fold, best_threshold)

            # Calculate the evaluation metric, a weighted combination of accuracy, precision and recall
            # Find the fold which maximizes this metric, and save its weights and threshold
            cm = ConfusionMatrix(y_valid, y_pred)
            eval_metric = 0.5 * recall(cm) + 0.25 * accuracy(y_valid, y_pred) + 0.25 * precision(cm)

            if eval_metric > max_metric:
                max_metric = eval_metric
                threshold = best_threshold
                weights = weights_fold

    # Having found optimal weights and threshold by cross validation, use them to get predictions for the test data
    y_pred = get_predictions(inputs_test, weights, threshold)
    proj = project_data(inputs_test, weights)
    histogram_projected_data(proj, targets_test, classes, threshold)
    print("\nThreshold selected:", threshold)

    return y_pred


def find_best_threshold(inputs, targets, weights):
    """
    Given inputs, targets, and weights, finds the threshold value to separate 2 classes by maximising
    the specified criteria.

    :param inputs: numpy array of inputs
    :param targets: numpy array of targets
    :param weights: numpy array of weights used to project the inputs to 1 dimension

    :return: float value representing the best threshold
    """
    # get projected inputs, then consider values between the min and max as potential thresholds
    projected_inputs = project_data(inputs, weights)
    thresholds = np.linspace(min(projected_inputs), max(projected_inputs),
                             int((max(projected_inputs) - min(projected_inputs)) * 10))

    # for each threshold, calculate the criteria and find the threshold that maximizes it
    max_metric, best_threshold = 0, None
    for threshold in thresholds:
        predictions = get_predictions(inputs, weights, threshold)
        cm = ConfusionMatrix(targets.astype(int), predictions)
        eval_metric = 0.5 * recall(cm) + 0.25 * precision(cm) + 0.25 * accuracy(targets, predictions)

        if eval_metric >= max_metric:
            max_metric = eval_metric
            best_threshold = threshold

    return best_threshold


def get_predictions(inputs, weights, threshold):
    """
    Given inputs, weights and a classification threshold found using Fisher's Linear Discriminant, produces predictions
    for the target values.

    :param inputs: numpy array of inputs
    :param weights: numpy array of weights used to project the inputs to 1 dimension
    :param threshold: a float value, where all projected values are classified as class 1 if they are >= threshold,
                      and class 0 otherwise

    :return: numpy array of predictions
    """
    # Project inputs to 1 dimension using weights
    projected_inputs = project_data(inputs, weights)

    # Generate predictions using projected inputs and threshold
    targets_pred = projected_inputs >= threshold

    return targets_pred.astype(int)


def get_projection_weights_fisher(inputs, targets):
    """
    Takes input and target data for classification and projects this down onto 1 dimension using the Fisher method.

    :param inputs: a 2d input matrix (array-like), each row is a data-point
    :param targets: 1d target vector (array-like) -- can be at most 2 classes (0 and 1)

    :return weights: the projection vector
    """

    if len(np.unique(targets)) > 2:
        raise ValueError("This method only supports data with two classes")

    weights = fisher_linear_discriminant_projection(inputs, targets)

    return weights


def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector parameters.

    :param data: a 2d data matrix (shape NxD array-like)
    :param weights: a 1d weight vector (shape D array like)

    :return projected_data: 1d vector (shape N np.array)
    """
    N, D = data.shape
    data = np.matrix(data)
    weights = np.matrix(weights).reshape((D, 1))
    projected_data = np.array(data * weights).flatten()
    return projected_data


def fisher_linear_discriminant_projection(inputs, targets):
    """
    Finds the direction of best projection based on Fisher's linear discriminant parameters.

    :param inputs: a 2d input matrix (array-like), each row is a data-point
    :param targets: 1d target vector (array-like) -- can be at most 2 classes (0 and 1)

    :return weights: a normalised projection vector corresponding to Fisher's linear discriminant
    """
    # get the shape of the data
    N, D = inputs.shape

    # separate the classes
    inputs0 = inputs[targets == 0]
    inputs1 = inputs[targets == 1]

    # find maximum likelihood approximations to the two data-sets
    m0, S_0 = max_lik_mv_gaussian(inputs0)
    m1, S_1 = max_lik_mv_gaussian(inputs1)

    # convert the mean vectors to column vectors (type matrix)
    m0 = np.matrix(m0).reshape((D, 1))
    m1 = np.matrix(m1).reshape((D, 1))

    # calculate the total within-class covariance matrix (type matrix)
    S_W = np.matrix(S_0 + S_1)

    # calculate weights vector and normalise
    weights = np.array(np.linalg.inv(S_W) * (m1 - m0))
    weights = weights / np.sum(weights)

    # we want to make sure that the projection is in the right direction
    # i.e. giving larger projected values to class1 so:
    projected_m0 = np.mean(project_data(inputs0, weights))
    projected_m1 = np.mean(project_data(inputs1, weights))
    if projected_m0 > projected_m1:
        weights = -weights

    return weights


def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix
    """
    # the mean sample is the mean of the rows of data
    N, dim = data.shape
    mu = np.mean(data, 0)
    Sigma = np.zeros((dim, dim))
    # the covariance matrix requires us to sum the dyadic product of
    # each sample minus the mean.
    for x in data:
        # subtract mean from data point, and reshape to column vector
        # note that numpy.matrix is being used so that the * operator
        # in the next line performs the outer-product v * v.T
        x_minus_mu = np.matrix(x - mu).reshape((dim, 1))
        # the outer-product v * v.T of a k-dimensional vector v gives
        # a (k x k)-matrix as output. This is added to the running total.
        Sigma += x_minus_mu * x_minus_mu.T
    # Sigma is un-normalised, so we divide by the number of data points
    Sigma /= N
    # we convert Sigma matrix back to an array to avoid confusion later
    return mu, np.asarray(Sigma)
