import pandas as pd
import numpy as np


def accuracy(y_test, y_pred):
    """
    calculates accuracy as the fraction of correct predictions over all predictions.
    Note that with skewed classes, you can get high classification accuracy by just predicting the more common
    (usually negative) class more often (or in the extreme case for every given observation).
    Therefore, accuracy is not a great evaluation metric for imbalanced datasets and this is why there is m
    ore than one evaluation metric. Overfitting with an excessively complex model can result in low test accuracy
    and underfitting an overly-simple model can result in low training accuracy.

    Parameters
    ----------
    y_test : array of actual target values (aka ground truth).
    y_pred : array of predicted target values (aka model predictions).
    Returns
    -------
    accuracy - accuracy score calculated as the fraction of correctly predicted target values over all predictions.
    """
    # Convert input into panda series
    y_pred = pd.Series(y_pred).array
    y_test = pd.Series(y_test).array

    # define counter for correct classifications
    correct_classifications = 0

    # loop over test array
    for i in range(len(y_test)):
        # check if predicted value matches corresponding value in test set
        if y_test[i] == y_pred[i]:
            # increment counter variable
            correct_classifications += 1

    # define variable for all possible classifications
    possible_classifications = len(y_test)

    # calculate accuracy score
    accuracy = correct_classifications / float(possible_classifications) * 100.0

    return accuracy


def ConfusionMatrix(y_test, y_pred):
    """
    Creates a 2x2 confusion matrix similar to sklearn's confusion_matrix function.
    Confusion matrices are a great way to quantitatively evaluate model performance.
    They also simplify the task of calculating precision, recall and hence the F1 score.
    This is particularly relevant to this domain as the associated costs of a false negative could be dire.
    A good visual accompaniment to the confusion matrix may be ROC and AUC analysis.

    Parameters
    ----------
    y_test : array of actual target values (aka ground truth).
    y_pred : array of predicted target values (aka model predictions).

    Returns
    -------
    confusion_matrix : a 2x2 confusion matrix
    """
    # convert inputs into Numpy arrays
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    # identify number of classes
    num_classes = len(np.unique(y_test))

    # create an empty matrix to populate with TP, TN, FP, FN in next step
    empty_cm = np.zeros((num_classes, num_classes)).astype(int)

    # loop through y values
    for i in range(len(y_test)):
        # increment TP, TN, FP, FN cells in confusion matrix as appropriate
        empty_cm[int(y_test[i])][int(y_pred[i])] += 1

    # rename for clarity
    confusion_matrix = empty_cm
    return confusion_matrix


def decompose_confusion_matrix(confusion_matrix):
    """
    isolates and creates variables for each element of the confusion matrix for later calculating the precision and
    recall.

    Parameters
    ----------
    confusion_matrix : a 2x2 confusion matrix akin to sklearn's confusion_matrix

    Returns
    -------
    true_positives : an outcome where the model correctly predicts the positive class (non-recurrence)
    true_negatives : an outcome where the model correctly predicts the negative class (recurrence).
    false_positives : an outcome where the model incorrectly predicts the positive class.
    false_negatives : an outcome where the model incorrectly predicts the negative class.
    """
    true_negatives = confusion_matrix[0][0]
    true_positives = confusion_matrix[1][1]
    false_positives = confusion_matrix[0][1]
    false_negatives = confusion_matrix[1][0]
    return true_positives, true_negatives, false_positives, false_negatives


def precision(confusion_matrix1):
    """
    Calculates the precision score from the confusion matrix akin to Sklearn's precision_score.
    Precision = true positives / (predicted positives) = true positives / (true positive + false positive) ).
    Precision indicates the proportion of correctly classified predictions from all predictions made for
    the positive class. i.e. answers the question: what percentage of positive class predictions are correct?.
    High precision indicates a low false positive rate, which is useful for imbalanced datasets with lots of negative
    class vs positive because precision doesn't include true negatives and is not affected by imbalance.
    Is used in the calculation of the F1 score.

    Parameters
    ----------
    confusion_matrix1 : a 2x2 confusion matrix akin to sklearn's confusion_matrix.

    Returns
    -------
    precision: the precision score for the positive class in a binary classification task.
    """
    true_positives, true_negatives, false_positives, false_negatives = decompose_confusion_matrix(confusion_matrix1)

    # calculate precision
    precision = true_positives / (true_positives + false_positives)

    return precision


def recall(confusion_matrix):
    """
    Calculates the recall score from the confusion matrix akin to Sklearn's recall_score.
    Recall = recall = true positives / actual positives = true positives / true positives + false negatives

    Intuitively, this indicates the ability of the classifier to find all the positive samples.
    I.e. answers the question: what proportion of all the actual positive class examples in the dataset did we classify
    correctly? High recall means most positives are predicted correctly.
    Recall is great for imbalanced datasets because it doesn't rely on false negatives.

    Parameters
    ----------
    confusion_matrix : a 2x2 confusion matrix akin to sklearn's confusion_matrix.

    Returns
    -------
    recall: the recall score for the positive class in a binary classification task.
    """
    true_positives, true_negatives, false_positives, false_negatives = decompose_confusion_matrix(confusion_matrix)

    # calculate recall
    recall = true_positives / (true_positives + false_negatives)

    return recall


def f1(precision, recall):
    """
    Returns the harmonic average of the precision and recall scores.
    f1 = (2 * precision * recall) / (precision + recall)
    Evaluating algorithms based on precision and recall means you use need to calculate and evaluate two
    separate metrics. This can result in difficulties relating to how to choose the optimal combination of the two
    metrics, which slows down decision-making. Therefore, a single evaluation metric is preferred.

    A geometric average of precision and recall is not great because if you are setting a low threshold, you may have
    high recall (e.g. classifier predicts y=1 all the time) and consequently a relatively high average, favouring the
    classifier. However, this is not useful as predicting only one class is not a very useful classifier when
    trying to generalize. A harmonic average is superior because it accounts for extreme values in precision or recall,
    e.g. if Recall/precision is near 0, then numerator is zero and F score will be near zero and if recall/precision
    is near 1, then numerator is small and F score will be near zero. You can only get a high F score if both precision
    and recall and relatively large (and balanced values), e.g. if P=1 and R=1.
    Using geometric average of precision and recall [(precision + recall)/2] is not great because if you are setting a
    low threshold, you may have high recall (e.g. classifier predicts y=1 all the time) and consequently a relatively
    high average (and vice versa) --> but predicting only one class is not a very useful
    classifier when trying to generalize --> F-score.

    Parameters
    ----------
    precision: precision score
    recall: recall score

    Returns
    -------
    recall: the recall score for the positive class in a binary classification task.
    """
    return 2 * (precision * recall) / (precision + recall)
