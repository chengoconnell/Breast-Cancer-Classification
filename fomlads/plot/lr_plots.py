import os
import numpy as np
import matplotlib.pyplot as plt
from fomlads.model.logistic_regression_functions import sigmoid, sigmoid_derivative


def learning_rate_comparison(learning_rates, models):
    """
    Plots a comparison of cross entropy errors for different learning rates across multiple iterations.
    Regularisation parameter is kept constant at 0.

    :param learning_rates: a list of learning rate values (float)
    :param models: a dictionary containing all iterations of learning rate and their corresponding error
    """

    for i in learning_rates:
        plt.plot(np.squeeze(models[i]["costs"]),
                 label=str(models[i]["learning rate"]))

    plt.title("Cross Entropy Error for Different Learning Rates")
    plt.ylabel("Cross Entropy Error")
    plt.xlabel("Number of Iterations (hundreds)")

    plt.legend(loc="upper center", shadow=True)

    plt.savefig(os.path.join("plots", "logistic",
                             "learning_rate_comparison.png"))


def regularisation_comparison(regularisation_parameters, models):
    """
    Plots a comparison of cross entropy errors for different regularisation across multiple iterations. 
    Learning rate is kept constant at 0.01.

    :param regularisation_parameters: a list of regularisation parameter values (float)
    :param models: a dictionary containing all iterations of regularisation and their corresponding error
    """
    for i in regularisation_parameters:
        plt.plot(np.squeeze(models[i]["costs"]),
                 label=str(models[i]["regularisation"]))

    plt.title("Cross Entropy Error for Different Regularisation Parameters")
    plt.ylabel("Cross Entropy Error")
    plt.xlabel("Number of Iterations (hundreds)")

    plt.legend(loc="upper center", shadow=True)

    plt.savefig(os.path.join("plots", "logistic",
                             "regularisation_comparison.png"))


def hypothesis_representation():
    """ 
    Displays the Sigmoid Function and its derivative across an evenly spread set of values for a hypothesis
    representation.
    Shows how the Sigmoid Function asymptotes at both one and zero. As the values approach minus infinity,
    the Sigmoid Function approaches zero. As the values approach infinity, the Sigmoid Function approaches one.
    """
    values = np.linspace(-10, 10, 100)
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(values, sigmoid(values), "r")
    ax.plot(values, sigmoid_derivative(values), "b")
    ax.set_title("Sigmoid Function Representation")
    ax.set_xlabel("Values")
    ax.set_ylabel("Sigmoid/Sigmoid Derivative")
    ax.legend(["Sigmoid", "Sigmoid Derivative"])
    fig.savefig(os.path.join("plots", "logistic",
                             "sigmoid_function.png"))
