from fomlads.plot.lr_plots import learning_rate_comparison, regularisation_comparison, hypothesis_representation
from fomlads.model.logistic_regression_functions import model_lr


# ------------ Choosing a learning rate ------------


def hyperparameter_tune(inputs_train_crossval,
                        targets_train_crossval,
                        num_iterations=1000, download_graphs=False):
    """
    :param inputs_kfolds: nested list containing training and validation inputs
    :param targets_kfolds: nested list containing training and validation targets
    :param num_iterations: number of iterations of gradient descent the model goes through
    :param download_graphs: whether to download comparison and sigmoid graphs onto user's computer
    """
    print("\nFinding best combination of learning rate and regularisation parameter... \n")

    models = {}
    models_graph_lrate = {}
    models_graph_r = {}

    learning_rates = [0.001, 0.002, 0.003, 0.005,
                      0.01]
    regularisation_parameters = [0, 0.01, 0.02, 0.04, 0.008,
                                 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]

    # Iterate through different combinations of learning rate and regularisation
    for i in learning_rates:
        for j in regularisation_parameters:
            if download_graphs:
                if j == 0:
                    models_graph_lrate[i] = model_lr(
                        inputs_train_crossval,
                        targets_train_crossval,
                        num_iterations,
                        learning_rate=i,
                        regularisation=j,
                        print_cost=False,
                        h_tuning=True
                    )[1]
                if i == 0.01:
                    models_graph_r[j] = model_lr(
                        inputs_train_crossval,
                        targets_train_crossval,
                        num_iterations,
                        learning_rate=i,
                        regularisation=j,
                        print_cost=False,
                        h_tuning=True
                    )[1]

            models[i, j] = model_lr(
                inputs_train_crossval,
                targets_train_crossval,
                num_iterations,
                learning_rate=i,
                regularisation=j,
                print_cost=False,
                h_tuning=True
            )[0]

    # Identify best learning rate and regularisation parameter based on overall evaluation score
    optimum_hyperparameters = max(
        models, key=models.get)

    print("Best learning rate: ",
          optimum_hyperparameters[0], "\nBest regularisation parameter: ", optimum_hyperparameters[1])

    if download_graphs:
        learning_rate_comparison(learning_rates, models_graph_lrate)
        regularisation_comparison(regularisation_parameters, models_graph_r)
        hypothesis_representation()

    return optimum_hyperparameters
