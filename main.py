import numpy as np
from tabulate import tabulate
import time

from fomlads.data.external import import_and_process
from fomlads.plot.exploratory_plots import generate_exploratory_plots

from fomlads.model.fisher import fit_and_predict_fisher
from fomlads.model.mlp import get_hyperparamters, mlp_predict
from fomlads.model.svm import svm_hpt, svm_predictions
from fomlads.model.logistic_regression_functions import model_lr, predict
from fomlads.model.logistic_regression_hyperparameter_tuning import hyperparameter_tune
from fomlads.model.metrics import accuracy, precision, recall, ConfusionMatrix, f1


def main(fname, input_cols=None, target_col=None, model="fisher"):
    """
    Imports the breast cancer data-set, fits the 4 models and generates predictions, then evaluates models
    parameters
    ----------
    fname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot
    """
    # Set random seed for reproducibility of results - 17 was used to conduct all of our experiments
    np.random.seed(17)

    # import data
    inputs_kfold, targets_kfold, inputs_train, targets_train, inputs_test, targets_test, input_cols, classes = \
        import_and_process(fname, input_cols=input_cols, target_col=target_col)

    if model == "mlp":
        print("Model: Multi-Layer Perceptrons")
        start_time = time.time()
        # for mlp, get the best hyperparameters
        activation_func, shape, reg_param = get_hyperparamters(
            inputs_kfold, targets_kfold)

        # for mlp, use best hyperparameters to train on the training set then predict on the test set
        predictions = mlp_predict(
            inputs_train, targets_train, inputs_test, activation_func, "lbfgs", shape, reg_param)
        end_time = time.time()

    elif model == "fisher":
        print("Model: Fisher's Linear Discriminant")
        # Get predictions for unseen data using Fisher's linear discriminant
        start_time = time.time()
        predictions = fit_and_predict_fisher(
            inputs_kfold, targets_kfold, inputs_test, targets_test, classes)
        end_time = time.time()

    elif model == "logistic":
        print("Model: Logistic Regression")
        # Retrieve best learning rate and regularisation parameter
        start_time = time.time()
        optimum_hyperparameters = hyperparameter_tune(inputs_kfold, targets_kfold, num_iterations=1000,
                                                      download_graphs=True)

        # Use best hyperparameters to train on the training set and retrieve best weight and bias parameters
        optimum_parameters = model_lr(inputs_kfold, targets_kfold, num_iterations=1000,
                                      learning_rate=optimum_hyperparameters[0],
                                      regularisation=optimum_hyperparameters[1], print_cost=False, p_tuning=True)
        # Use best parameters to predict on unseen test data
        predictions = predict(optimum_parameters["weights"],
                              optimum_parameters["bias"], inputs_test)
        end_time = time.time()

    elif model == "svm":
        print("Model: Support Vector Machines")
        # tune SVM hyperparameter settings
        start_time = time.time()
        best_reg_param = svm_hpt(inputs_kfold, targets_kfold)
        # get predictions
        predictions = svm_predictions(
            inputs_train, targets_train, inputs_test, best_reg_param)
        end_time = time.time()
    else:
        print("Not a valid model. Try running again. Model options are:\n'fisher'\n'mlp'\n'svm'\n'logistic'")

    # Calculate evaluation metrics for the selected model
    model_cm = ConfusionMatrix(targets_test.astype(int), predictions)
    model_accuracy = accuracy(targets_test, predictions)
    model_precision = precision(model_cm)
    model_recall = recall(model_cm)
    model_f1 = f1(model_precision, model_recall)

    # Output evaluation metrics to console
    print("\nMetrics Results:")
    metric_table = [["Accuracy", '{0:.2f}'.format(model_accuracy)], ["Precision", '{0:.2f}'.format(model_precision)],
                    ["Recall", '{0:.2f}'.format(model_recall)], [
                        "F1 Score", '{0:.2f}'.format(model_f1)],
                    ["Time", '{0:.2f}s'.format((end_time - start_time))]]
    print(tabulate(metric_table, headers=[
        "Metric", "Value"], tablefmt="pretty"))

    # Output confusion matrix
    print("\nConfusion Matrix:")
    cm_print = [["Actual Class " + str(i)] + list(model_cm[i])
                for i in range(len(model_cm))]
    print(tabulate(cm_print, headers=[
        "Predicted Class 0", "Predicted Class 1"], tablefmt="pretty"))


if __name__ == '__main__':
    import sys

    # Assumes that the first argument is the input filename/path
    if len(sys.argv) > 3:
        # exclude argument 0 (name of script), argument 1 (name of data file),
        input_columns = sys.argv[3::]
        # and argument 2 (name of model)
        try:
            print("\nModel fit using features: ", input_columns, "\n")
            main(fname=sys.argv[1], input_cols=input_columns,
                 target_col='class', model=sys.argv[2])
            generate_exploratory_plots(fname=sys.argv[1])
        except KeyError as e:
            print("One of the attributes you typed does not exist. This caused the following error to be raised: ", e)
    elif len(sys.argv) == 3:
        print("\nModel fit using all available features\n")
        main(fname=sys.argv[1], target_col='class', model=sys.argv[2])
        generate_exploratory_plots(fname=sys.argv[1])
    else:
        print("Please provide the name of the model you would like to run!")
