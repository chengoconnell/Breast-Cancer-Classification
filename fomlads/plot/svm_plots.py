import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def svm_plot_metrics(reg_params_value, hyperparameter_of_interest, plotting_accuracy, plotting_precision,
                     plotting_recall, plotting_f1):
    """ 
    Function plots performance metrics for each hyperparameter value

    Parameters
    ----------
    reg_params_value: different values of the regularisation parameter used during hyperparameter tuning.
    hyperparameter_of_interest: name of the hyperparameter being investigated
    plotting_accuracy: accuracy score for a given hyperparameter value 
    plotting_precision: precision score for a given hyperparameter value
    plotting_recall: recall score for a given hyperparameter value
    plotting_f1: f1 score for a given hyperparameter value 

    Returns
    -------
    plots chart with metric values for each hyperparameter value. 
    
    """
    # convert inputs into numpy arrays to allow plotting of arrays with different shapes
    reg_params_value = np.array(reg_params_value)
    plotting_accuracy = np.array(plotting_accuracy) / 100
    plotting_precision = np.array(plotting_precision)
    plotting_recall = np.array(plotting_recall)
    plotting_f1 = np.array(plotting_f1)

    # prepare plot 
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    # set x and y labels
    ax.set_ylabel("Score (0-1)")
    ax.set_xlabel(f"{hyperparameter_of_interest}")

    # set x label scale
    ax.set_xscale('log')
    ax.get_xticks([])

    # format X_axis
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.3f'))

    # set title
    ax.set_title(f"Performance metrics for each value of the {hyperparameter_of_interest}")

    # # prepare metrics 
    metrics_list = [plotting_accuracy, plotting_precision, plotting_recall, plotting_f1]

    # plot chart
    for metric in metrics_list:
        ax.plot(reg_params_value, metric)

    # add legend
    ax.legend(["Accuracy", "Precision", "Recall", "F1 Score"])

    # Save figure
    fig.savefig(os.path.join("plots", "svm",
                             "Comparison of metrics for each value of the {}.png".format(hyperparameter_of_interest)))


def svm_hpt_plot(reg_params_value, reg_params_score, hyperparameter_of_interest):
    """ 
    Function plots hyperparameter values vs weighted average score for comparison. 

    Parameters
    ----------
    reg_params_value: different values of the regularisation parameter used during hyperparameter tuning.
    reg_params_score: weighted average scores for each of the parameter values
    hyperparameter_of_interest: name of the hyperparameter being investigated

    Returns
    -------
    plots chart comparing hyperparameter values vs weighted average score. 
    
    """
    # prepare plot
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    # set labels
    ax.set_ylabel(f"Weighted Average Score")
    ax.set_xlabel(f"{hyperparameter_of_interest}")

    # set title
    ax.set_title(f"SVM's weighted average score for each value of the {hyperparameter_of_interest}")
    ax.set_xscale('log')

    # format X_axis
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.3f'))

    # plot chart
    ax.plot(reg_params_value, reg_params_score)

    # save figure
    fig.savefig(os.path.join("plots", "svm", "Weighted average score for each regularisation parameter"))
