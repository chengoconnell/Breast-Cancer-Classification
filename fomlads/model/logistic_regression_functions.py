import numpy as np
import time
import math

from fomlads.model.metrics import accuracy, recall, precision, ConfusionMatrix


def sigmoid(Z):
    """ 
    Z: weights transposed X inputs +  bias
    """
    s = 1 / (1 + np.exp(-Z))
    return s


def sigmoid_derivative(Z):
    """
    Z: weights transposed X inputs +  bias
    """
    s = sigmoid(Z)
    ds = s * (1 - s)
    return ds


def propagate(w, b, X, Y, regularisation):
    """
    w: weights 
    b: bias
    X: inputs
    Y: targets
    """

    # Take shape of X (number of examples, number of features)
    n, m = X.shape

    # Z equals to weights transposed multiplied by X + bias
    Z = np.dot(X, w.T) + b

    # Prediction function
    y_prediction = sigmoid(Z)

    # ----------- Forward propagation: calculating current loss -----------
    # Cost function
    cost = -(1 / m) * np.sum(
        Y * np.log(y_prediction) + (1 - Y) * np.log(1 - y_prediction) +
        regularisation / (2 * m) * (np.sum(w ** 2) + b ** 2)
    )

    # ----------- Backward propagation: calculating current gradient -----------
    # Gradient of loss with respect to weight
    dw = (
        np.dot(X.T, (y_prediction - Y)) + (regularisation * w)
    ) / m
    # Gradient of loss with respect to bias
    db = (np.sum(y_prediction - Y)) / m

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}

    return grads, cost


def optimise(w, b, X, Y, num_iterations, learning_rate, regularisation, print_cost=False):
    """
    w: weights 
    b: bias
    X: inputs
    Y: targets
    num_iterations: number of iterations chosen
    learning_rate: learning rate chosen
    regularisation: regularisation parameter chosen
    print_cost: if true, function will print a numpy array of costs
    """

    costs = []

    # loop through iterations
    for i in range(num_iterations):
        # cost and gradient calculation
        grads, cost = propagate(w, b, X, Y, regularisation)

        # retrieve partial derivatives
        dw = grads["dw"]
        db = grads["db"]

        # update weights and bias
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # track cost value against number_iterations
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    # insert updated weights and bias into dictionary
    params = {"w": w, "b": b}

    # insert updated derivatives into dictionary
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    This function takes input values, X, and assigns it to a predicted class (0,1).
    """

    # Z equals to weights transposed multiplied by X + bias
    Z = np.dot(X, w.T) + b

    # Sigmoid Function
    Y_predicted = sigmoid(Z)

    # Check if output is equal and greater than 0.5 or less than
    Y_predicted_class = [1 if probability >
                         0.5 else 0 for probability in Y_predicted]

    return Y_predicted_class


def model_lr(
        X_train,
        Y_train,
        num_iterations=2000,
        learning_rate=0.01,
        regularisation=0,
        print_cost=False,
        h_tuning=False,
        p_tuning=False,
):
    """
    This is the overall model function which places all the separate functions into one place.

    """
    if p_tuning:
        print("\n\nFinding best bias and weight values... \n")

    folds_list = []
    # Loop through cross-validation folds to train on training data and test on validation data
    for i in range(len(X_train)):
        number_samples, number_features = X_train[i]["train"].shape
        # Initialise weights and bias
        w = np.zeros(number_features)
        b = 0

        # Fit parameters to training data
        parameters, gradients, costs = optimise(
            w,
            b,
            X_train[i]["train"],
            Y_train[i]["train"],
            num_iterations,
            learning_rate,
            regularisation,
            print_cost,
        )

        # Retrieve maximum likelihood parameters
        w = parameters["w"]
        b = parameters["b"]

        # Make predictions on validation data with maximum likelihood parameters
        Y_prediction_test = predict(w, b, X_train[i]["valid"])

        # ------ Calculate validation evaluation scores -------
        validation_accuracy = accuracy(Y_train[i]["valid"], Y_prediction_test)

        validation_confusion_matrix = ConfusionMatrix(
            Y_train[i]["valid"], Y_prediction_test)

        validation_precision = precision(validation_confusion_matrix)

        validation_recall = recall(validation_confusion_matrix)

        # Calculate weighted score of fold based on precision, accuracy, and recall
        evaluation_score = 0.25 * validation_precision + 0.25 * \
            validation_accuracy + 0.5 * validation_recall

        folds_list.append({
            "fold": i,
            "costs": costs,
            "learning rate": learning_rate,
            "regularisation": regularisation,
            "weights": w,
            "bias": b,
            "overall evaluation score": evaluation_score
        })

    # Sort folds by highest evaluation score
    sorted_list = sorted(
        folds_list, key=lambda k: k["overall evaluation score"], reverse=True
    )
    # Calculate average score across k folds
    average_score = sum(d["overall evaluation score"] for d in sorted_list) / len(
        sorted_list)

    # If optimising hyperparameters
    if h_tuning:
        return average_score, sorted_list[math.floor(len(sorted_list) / 2)]
    # If optimising weights and bias...
    elif p_tuning:
        print("Best weight values: ", sorted_list[0]["weights"])
        print("Best bias value: ", sorted_list[0]["bias"])
        return sorted_list[0]
