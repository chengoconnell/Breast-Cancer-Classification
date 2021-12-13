# import external libraries
import numpy as np

# import required Sklearn functions
from sklearn.svm import LinearSVC

# import internal modules
from fomlads.model.metrics import accuracy, precision, recall, f1, ConfusionMatrix
from fomlads.plot.svm_plots import svm_plot_metrics, svm_hpt_plot


def svm_hpt(inputs_kfold, targets_kfold):
    """
    Conducts k-fold cross-validation to find the best value of the regularisation hyperparameter 'C' for a support
    vector machine (SVM) classifier, given arrays of the k training and validation sets.

    Parameters
    ----------
    inputs_kfold: list containing input training and validation sets.
    targets_kfold: list containing target training and validation sets.

    Returns
    -------
    best_reg_param: value of best regularisation parameter ('C') for the SVM classifier.
    """

    # assign regularisation parameter options to a list
    reg_param = [0.001, 0.01, 0.1, 1, 10, 100]

    # declare dictionary to store reg_param elements as keys (and later cross_val_scores as values)
    weighted_avg_scores = {key: None for key in reg_param}

    # declare lists to store averaged metrics for plotting later
    plotting_accuracy = []
    plotting_precision = []
    plotting_recall = []
    plotting_f1 = []

    # loop through hyperparameter setting (and later test out each one on all training folds)
    for i in range(len(reg_param)):

        # declare/reset lists to store metrics for each round of hyperparameter tuning
        fold_accuracy = []
        fold_cm = []
        fold_precision = []
        fold_recall = []
        fold_f1 = []

        # define classifier
        clf = LinearSVC(random_state=42, dual=False)

        # manually change parameter
        clf.C = reg_param[i]

        # loop through folds
        for j in range(len(inputs_kfold)):
            X_train = inputs_kfold[j]["train"]
            X_valid = inputs_kfold[j]["valid"]
            y_train = targets_kfold[j]["train"]
            y_valid = targets_kfold[j]["valid"]

            # fit model
            clf.fit(X_train, y_train)

            # use the trained model to make predictions of response variable for validation set
            y_valid_pred = clf.predict(X_valid)

            # assign variables to store metrics computed for this fold
            cv_accuracy = accuracy(y_valid, y_valid_pred)
            cv_cm = ConfusionMatrix(y_valid, y_valid_pred)
            cv_precision = precision(cv_cm)
            cv_recall = recall(cv_cm)
            cv_f1 = f1(cv_precision, cv_recall)

            # store metrics in list
            fold_accuracy.append(cv_accuracy)
            fold_cm.append(cv_cm)
            fold_precision.append(cv_precision)
            fold_recall.append(cv_recall)
            fold_f1.append(cv_f1)

        # calculate the mean of each metric for that fold
        fold_avg_accuracy = np.mean(fold_accuracy)
        fold_avg_precision = np.mean(fold_precision)
        fold_avg_recall = np.mean(fold_recall)
        fold_avg_f1 = np.mean(fold_f1)

        # store average of each metric for plotting later
        plotting_accuracy.append(fold_avg_accuracy)
        plotting_precision.append(fold_avg_precision)
        plotting_recall.append(fold_avg_recall)
        plotting_f1.append(fold_avg_f1)

        # Calculate the weighted average of the metrics for this fold made up of: accuracy (25%), precision(25%),
        # and  recall  (50%).
        fold_weighted_avg = (0.25 * fold_avg_accuracy) + \
                            (0.25 * fold_avg_precision) + (0.5 * fold_avg_recall)

        # add weighted average to dictionary containing scores for each parameter
        weighted_avg_scores[reg_param[i]] = fold_weighted_avg

    # print reg_param values and scores in a pretty table 
    print("\nCross-validation results: ")
    from tabulate import tabulate
    reg_params_value = list(weighted_avg_scores.keys())
    reg_params_score = list(weighted_avg_scores.values())
    reg_params_score = ["%.2f" % score for score in reg_params_score]
    headers = ["Regularisation parameter value", "Weighted average CV score"]
    reg_param_table = zip(reg_params_value, reg_params_score)
    print(tabulate(reg_param_table, headers=headers, tablefmt="pretty"))

    # identify hyperparameter with highest weighted average score
    best_reg_param = max(weighted_avg_scores, key=weighted_avg_scores.get)
    best_weighted_avg = max(weighted_avg_scores.values())
    print(f"Best regularisation parameter value: {best_reg_param}")
    print(f"Weighted average score for best regularisation parameter value: {best_weighted_avg:.2f}.")
    # print("\n")

    # prepare lists of reg_param values and scores for plots
    hyperparameter_of_interest = "Regularisation penalty ('C')"
    reg_params_value = list(weighted_avg_scores.keys())
    reg_params_score = list(weighted_avg_scores.values())

    # plot reg_pram vs each metric
    svm_plot_metrics(reg_params_value, hyperparameter_of_interest, plotting_accuracy, plotting_precision,
                     plotting_recall, plotting_f1)

    # plot reg_param vs weighted average score
    svm_hpt_plot(reg_params_value, reg_params_score, hyperparameter_of_interest)

    # return the best value for the reg_param
    return best_reg_param


def svm_predictions(inputs_train, targets_train, inputs_test, best_reg_param):
    """
    Returns an array of predictions for a given array of classes. 

    Parameters
    ----------
    inputs_train: training features. 
    targets_train: training targets.
    inputs_test: test features.
    best_reg_param: best regularisation parameter.

    Returns
    -------
    y_pred : predictions for test set.
    """

    # assign training and testing sets
    X_train = inputs_train
    y_train = targets_train
    X_test = inputs_test

    # define classifier
    clf = LinearSVC(random_state=42, dual=False)

    # manually change parameter
    clf.C = best_reg_param

    # fit model
    clf.fit(X_train, y_train)

    # use the trained model to make predictions of response variable for test set
    y_pred = clf.predict(X_test)

    # return predictions
    return y_pred
