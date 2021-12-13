from sklearn.neural_network import MLPClassifier
from fomlads.plot.mlp_plots import plot_metrics, plot_training_loss
from fomlads.model.mlp_hyperparameter_tuning import find_best_reg_param, find_best_activation_function, \
    find_best_shape
import numpy as np


def get_hyperparamters(inputs_kfold, targets_kfold):
    """
    Takes inputs and targets and uses k-fold cross validation to find the best activation function, solver, network
    shape and regularisation parameter
    :param inputs_kfold: list containing input training and validation sets
    :param targets_kfold: list containing target training and validation sets
    :return: best activation function (string), best solver (string), best network shape (tuple), best regularisation
    parameter (int)
    """
    # find best activation function using solver found previously
    activation_functions = ["tanh", "relu", "logistic", "identity"]
    acc, prec, rec, f1, best_func, losses = find_best_activation_function(
        inputs_kfold, targets_kfold, activation_functions)
    # plot graphs of each metric
    plot_metrics(acc, prec, rec, f1, activation_functions, "activation_func")
    plot_training_loss(activation_functions, losses, "activation_func")

    # find best network shape using best_func and best_solver
    shapes = [(3, 1), (1, 3), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (1, 5), (2, 5), (3, 5), (4, 5),
              (3, 2, 1), (1, 2, 3), (6, 5, 4), (4, 5, 6)]
    acc, prec, rec, f1, best_shape, losses = find_best_shape(
        inputs_kfold, targets_kfold, shapes)
    # plot graphs of each metric
    plot_metrics(acc, prec, rec, f1, shapes, "shape")
    plot_training_loss(shapes, losses, "shape")

    # find best regularisation param using best_func, best_solver, best_shape
    reg = np.logspace(-2, 2, 20)
    acc, prec, rec, f1, best_reg_param, losses = find_best_reg_param(
        inputs_kfold, targets_kfold, reg)
    # plot graphs of each metric
    plot_metrics(acc, prec, rec, f1, reg, "reg_param")
    plot_training_loss(reg, losses, "reg_param")

    print("Best hyperparameters:\nActivation Function: {}\nNetwork Shape: {}\nRegularisation "
          "Parameter: {}\n".format(best_func, best_shape, best_reg_param))

    return best_func, best_shape, best_reg_param


def mlp_predict(train_inputs, train_targets, test_inputs, activation_func, solver, shape, reg_param,
                learning_rate=1e-3):
    """
    Makes predictions using the optimised hyperparameters using the MLP classifier
    :param train_inputs: matrix containing training set inputs (np.array)
    :param train_targets: array containing training set targets (np.array)
    :param test_inputs: matrix containing test set inputs (np.array)
    :param activation_func: activation function (string)
    :param solver: learning algorithm (string)
    :param shape: network shape (tuple)
    :param reg_param: regularisation parameter (float)
    :param learning_rate: learning rate, only relevant if the solver is 'sgd' or 'adam'
    :return: predictions: a list containing predictions for the test set
    """
    # set up MLP classifier
    clf = MLPClassifier(activation=activation_func, solver=solver, hidden_layer_sizes=shape,
                        random_state=2, max_iter=5000, alpha=reg_param, learning_rate_init=learning_rate)
    clf.fit(train_inputs, train_targets)
    predictions = clf.predict(test_inputs)

    return predictions
