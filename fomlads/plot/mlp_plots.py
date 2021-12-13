# file to plot different graphs for the MLP model
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_training_loss(hyperparameters, loss_curves, hyperparameter_type):
    """
    Function to plot loss curves for MLP, for different activation functions
    :param hyperparameters: a list of hyperparameters used to retrieve the loss curves
    :param loss_curves: a nested list of loss curves for each activation function
    :param hyperparameter_type: string of hyperparameter that is being changed
    """

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title("MLP Cross Entropy for Different {}".format(hyperparameter_type))
    for i, curve in enumerate(loss_curves):
        ax.scatter([i for i in range(len(curve))], curve)
    ax.legend(hyperparameters)
    fig.savefig(os.path.join("plots", "mlp", "loss_{}_comparison.png".format(hyperparameter_type)))


def plot_metrics(accuracies, precisions, recalls, f1s, hyperparameters, hyperparameter_type):
    """
    Plots different performance metrics as a function of different hyperparameters
    :param accuracies: list containing accuracy found for each hyperparameter
    :param precisions: list containing precision found for each hyperparameter
    :param recalls: list containing recall found for each hyperparameter
    :param f1s: list containing f1 score for each hyperparameter
    :param hyperparameters: list containing hyperparameters to be plotted
    :param hyperparameter_type: string containing hyperparameter name
    :return:
    """
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Comparing Metrics for Different {}".format(hyperparameter_type))
    metrics = [np.array(accuracies) / 100, precisions, recalls, f1s]

    for metric in metrics:
        if hyperparameter_type == "activation_func" or hyperparameter_type == "shape":
            ax.set_xlabel(hyperparameter_type)
            xs = np.arange(len(hyperparameters))
            ax.plot(xs, metric)
            ax.set_xticks(xs)
            ax.set_xticklabels(hyperparameters)
        elif hyperparameter_type == "reg_param":
            ax.set_xlabel("$log_{}$({})".format("{10}", hyperparameter_type))
            ax.plot(np.log(hyperparameters), metric)
    ax.legend(["Accuracy", "Precision", "Recall", "F1 Score"])
    fig.savefig(os.path.join("plots", "mlp", "{}_metric_comparison.png".format(hyperparameter_type)))
