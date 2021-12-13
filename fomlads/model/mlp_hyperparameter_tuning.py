# testing out mlp using the sklearn library
from sklearn.neural_network import MLPClassifier
from fomlads.model.metrics import accuracy, precision, recall, f1, ConfusionMatrix, decompose_confusion_matrix
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def train_and_predict_k_fold(inputs_kfolds, targets_kfolds, activation_func, solver, network_shape,
                             regularisation_param,
                             learning_rate):
    """
    Trains MLP using k-fold cross validation. Computes predictions and performance metrics per fo
    :param inputs_kfolds: nested list containing training and validation inputs
    :param targets_kfolds: nested list containing training and validation targets
    :param activation_func: activation function (string)
    :param solver: learning algorithm (string)
    :param network_shape: shape of network (tuple). Each item represents number of units in that layer. Only hidden
    layers need to be defined.
    :param regularisation_param: regularisation parameter (float)
    :param learning_rate: learning rate (float). Only relevant if the solver is 'sgd' or 'adam'
    :return: list: contains metrics for each fold in the k-fold. Metrics are: loss, mean accuracy, mean confusion
    matrix, mean true positives, mean true negatives, mean false positive, mean false negatives, mean precision,
    mean recall, mean f1 score.
    """
    # set up MLP classifier
    clf = MLPClassifier(activation=activation_func, solver=solver, hidden_layer_sizes=network_shape,
                        random_state=17, max_iter=1000, alpha=regularisation_param, learning_rate_init=learning_rate)

    # prepare lists to store metrics
    ave_accuracy = []
    ave_CM = []
    ave_tp = []
    ave_tn = []
    ave_fp = []
    ave_fn = []
    ave_prec = []
    ave_rec = []
    ave_f1 = []
    loss = []
    for i in range(len(inputs_kfolds)):
        # split data into train and test sets
        X_train = inputs_kfolds[i]["train"]
        X_test = inputs_kfolds[i]["valid"]
        y_train = targets_kfolds[i]["train"]
        y_test = targets_kfolds[i]["valid"]

        # fit the model
        clf.fit(X_train, y_train)

        # get the loss curve from this fold
        # no loss curve for lbfgs as it is not iterative - uses hessian instead
        try:
            loss.append(clf.loss_)  # gives final loss
        except AttributeError as err:
            pass

        # use the trained model to make predictions
        predictions = clf.predict(X_test)

        # append metrics from this fold
        ave_accuracy.append(accuracy(y_test, predictions))
        CM = ConfusionMatrix(y_test, predictions)
        ave_CM.append(CM)
        tp, tn, fp, fn = decompose_confusion_matrix(CM)
        ave_tp.append(tp)
        ave_tn.append(tn)
        ave_fp.append(fp)
        ave_fn.append(fn)
        ave_prec.append(precision(CM))
        ave_rec.append(recall(CM))
        ave_f1.append(f1(precision(CM), recall(CM)))
    # calculate mean metrics over all folds
    mean_accuracy = np.mean(ave_accuracy)
    mean_CM = np.mean(ave_CM, axis=0)
    mean_tp = np.mean(ave_tp)
    mean_tn = np.mean(ave_tn)
    mean_fp = np.mean(ave_fp)
    mean_fn = np.mean(ave_fn)
    mean_prec = np.mean(ave_prec)
    mean_rec = np.mean(ave_rec)
    mean_f1 = np.mean(ave_f1)

    return [loss, mean_accuracy, mean_CM, mean_tp, mean_tn, mean_fp, mean_fn, mean_prec, mean_rec, mean_f1]


def find_best_activation_function(inputs_kfolds, targets_kfolds, activation_functions, solver="lbfgs", shape=(5, 5),
                                  reg_param=2, learning_rate=1e-4):
    """Finds best activation function for a MLP, based on which activation function results in the smallest training
    error
    :param inputs_kfolds: array of inputs
    :param targets_kfolds: array of target values
    :param activation_functions: list containing activation functions that can be used with sklearn's MLP classifier
    :param solver: learning algorithm that can be used with sklearn's MLP classifier. Defaults to "sgd"
    :param shape: tuple of shape of network, defaults to (5,5)
    :param reg_param: regularisation parameter to be used for the L2 norm, defaults to 1
    :param learning_rate: learning rate used if the solver is "sgd" or "adam". Ignored otherwise. Defaults to 1e-3
    :return: accuracy (list), precision (list), recall (list), f1 score (list), best activation function (string),
    losses (list)
    """

    print("Finding best activation function, keeping all other parameters constant...")
    acc = []
    rec = []
    prec = []
    f1 = []
    losses = []
    best_func = {}
    for func in activation_functions:
        results = train_and_predict_k_fold(
            inputs_kfolds, targets_kfolds, func, solver, shape, reg_param, learning_rate)
        acc.append(results[1])
        prec.append(results[7])
        rec.append(results[8])
        f1.append(results[9])
        losses.append(results[0])
        best_func[func] = 0.25 * results[1] + 0.25 * results[7] + 0.5 * results[8]

    # if any values in best_shape are NaN, replace with 0
    for key in best_func.keys():
        best_func[key] = np.nan_to_num(best_func[key])

    return acc, prec, rec, f1, max(best_func, key=best_func.get), losses


def find_best_shape(inputs_kfolds, targets_kfolds, shapes, activation_func="identity", solver="lbfgs", reg_param=2,
                    learning_rate=1e-4):
    """
    Finds best shape (architecture) for a MLP, based on which shape results in the best weighted
    mean of the precision, recall, f1 score and accuracy in k-fold cross val
    :param inputs_kfolds: array of inputs
    :param targets_kfolds: array of target values
    :param shapes: list containing shape tuples that can be used with sklearn's MLP classifier
    :param activation_func: activation function that can be used with sklearn's MLP classifier. Defaults to "identity"
    :param solver: learning algorithm that can be used with sklearn's MLP classifier. Defaults to "sgd"
    :param reg_param: regularisation parameter to be used for the L2 norm. Defaults to 1
    :param learning_rate: learning rate used if the solver is "sgd" or "adam". Ignored otherwise. Defaults to 1e-3
    :return: accuracy (list), precision (list), recall (list), f1 score (list), best shape (tuple), losses (list)
    """

    print("Finding best network shape, keeping all other parameters constant...")
    acc = []
    rec = []
    prec = []
    f1 = []
    losses = []
    best_shape = {}
    for shape in shapes:
        results = train_and_predict_k_fold(
            inputs_kfolds, targets_kfolds, activation_func, solver, shape, reg_param, learning_rate)
        acc.append(results[1])
        prec.append(results[7])
        rec.append(results[8])
        f1.append(results[9])
        losses.append(results[0])
        best_shape[shape] = 0.25 * results[1] + 0.25 * results[7] + 0.5 * results[8]

    # if any values in best_shape are NaN, replace with 0
    for key in best_shape.keys():
        best_shape[key] = np.nan_to_num(best_shape[key])

    return acc, prec, rec, f1, max(best_shape, key=best_shape.get), losses


def find_best_reg_param(inputs_kfolds, targets_kfolds, reg_params, shape=(5, 5), activation_func="identity",
                        solver="lbfgs", learning_rate=1e-4):
    """
    Finds best regularisation parameter for a MLP, based on which parameter results in the best weighted
    mean of the precision, recall, f1 score and accuracy in k-fold cross val
    :param inputs_kfolds: array of inputs
    :param targets_kfolds: array of target values
    :param reg_params: list containing regularisation parameters (integers)
    :param activation_func: activation function that can be used with sklearn's MLP classifier. Defaults to "identity"
    :param solver: learning algorithm that can be used with sklearn's MLP classifier. Defaults to "sgd"
    :param shape: tuple of shape of network. Defaults to (5,5)
    :param learning_rate: learning rate used if the solver is "sgd" or "adam". Ignored otherwise. Defaults to 1e-3
    :return: accuracy (list), precision (list), recall (list), f1 score (list), best regularisation parameter (float),
    losses (list)
    """

    print("Finding best regularisation parameter, keeping all other parameters constant...")
    acc = []
    rec = []
    prec = []
    f1 = []
    losses = []
    best_reg_param = {}
    for param in reg_params:
        results = train_and_predict_k_fold(
            inputs_kfolds, targets_kfolds, activation_func, solver, shape, param, learning_rate)
        acc.append(results[1])
        prec.append(results[7])
        rec.append(results[8])
        f1.append(results[9])
        losses.append(results[0])
        best_reg_param[param] = 0.25 * results[1] + \
            0.25 * results[7] + 0.5 * results[8]
    # if any values in best_solver are NaN, replace with 0
    for key in best_reg_param.keys():
        best_reg_param[key] = np.nan_to_num(best_reg_param[key])

    return acc, prec, rec, f1, max(best_reg_param, key=best_reg_param.get), losses
