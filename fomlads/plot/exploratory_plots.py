import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau


def load_data_for_plotting(path):
    """
    A simple load function for the breast cancer data. It replaces missing values with modes.

    :param path: path of the breast cancer dataset

    :return: breast cancer data loaded in a DataFrame
    """
    df = pd.read_csv(path)

    # some values are missing, replace them with mode
    modes = df.mode()
    mode_n_caps = modes["node-caps"][0]
    mode_b_quad = modes["breast-quad"][0]
    replace_map = {"node-caps": {"?": mode_n_caps},
                   "breast-quad": {"?": mode_b_quad}
                   }
    df.replace(replace_map, inplace=True)

    return df


def plot_class_histograms(df):
    """
    Plots histograms of the class distribution of each feature and saves the plots on disk.

    :param df: breast cancer DataFrame
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
    column_index = 0
    for i in range(2):
        for j in range(2):
            sns.countplot(
                ax=axes[i, j],
                x=df.columns.values[column_index],
                data=df,
                hue="class",
                palette="coolwarm",
            )
            column_index += 1

    for i in range(2):
        for j in range(3):
            sns.countplot(
                ax=axes2[i, j],
                x=df.columns.values[column_index],
                data=df,
                hue="class",
                palette="coolwarm",
            )
            column_index += 1

    fig.savefig(os.path.join("plots", "exploratory", "class_distributions1.png"))
    fig2.savefig(os.path.join("plots", "exploratory", "class_distributions2.png"))


def correlations_heatmap(df):
    """
    Plots a heatmap of Kendall's Tau rank correlations between all columns of the dataset and saves it to disk.

    :param df: breast cancer DataFrame
    """
    # Make an empty correlation matrix to fill out with values
    N = len(df.columns)
    corr_matrix = np.zeros((N, N))

    # Get correlations between each couple of columns in the dataset
    for i, column1 in enumerate(df.columns):
        for j, column2 in enumerate(df.columns):
            coef, p = kendalltau(df[column1], df[column2])
            corr_matrix[i][j] = coef

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)

    # Add a mask to remove top half of table
    # Source - https://towardsdatascience.com/formatting-tips-for-correlation-heatmaps-in-seaborn-4478ef15d87f
    mask = np.zeros(corr_matrix.shape, dtype=bool)
    mask[np.triu_indices(len(mask))] = True

    # Generate heatmap
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, xticklabels=df.columns,
                     yticklabels=df.columns, mask=mask)

    # Rotate the x-axis labels horizontally
    plt.xticks(rotation=0)
    fig.savefig(os.path.join("plots", "exploratory", "correlations-heatmap.png"))


def generate_exploratory_plots(fname):
    """
    Generates the class distributions histograms and correlations heatmap.

    :param fname: path to breast cancer dataset
    """
    # Turn interactive plotting off
    plt.ioff()

    # Load the data
    df = load_data_for_plotting(fname)

    # create histograms of class distributions across features
    plot_class_histograms(df)

    # create a heatmap of column correlations
    correlations_heatmap(df)
