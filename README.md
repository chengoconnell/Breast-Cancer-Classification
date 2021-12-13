# FOMLADS Group Coursework

Report Template: Model Selection
## About this repository
This repository contains all the source code for the INST0060 UCL Module "Foundations of Machine Learning and Data Science" group project, completed by the above team members.
## Dependencies
Make sure you have the following dependencies installed before running the script:
- numpy
- pandas
- sklearn
- matplotlib
- tabulate   
- seaborn
- scipy

To install these, open up the command line and make sure you're in the root folder (the same folder that contains main.py). Then type `python -m pip install -r requirements.txt`.


## Running the models
All models are run from the command line. When running the script, you must select a model to run. You can also select 
certain columns from the dataset.

### Running different models
To run different models, type:
- `python main.py breast-cancer.data NAME_OF_MODEL`
- Where NAME_OF_MODEL should be swapped out for any one of:
     - `fisher`
     - `mlp`
     - `logistic`
     - `svm`

### Running the script with select columns
To run the script with a specific selection of columns, simply type the name of the columns you would like to use after the model. For example, to run the script using Fishers Linear Discriminant with the columns 'inv-nodes', 'deg-malig', 'irradiat' and 'node-caps' you should type:
- `python main.py breast-cancer.data fisher inv-nodes deg-malig irradiat node-caps`

## File Structure
The structure of this repository is as follows:

1. fomlads
     1. data
        - external.py : contains functions for importing and pre-processing data, as well as splitting into train-test sets
     2. model
        - mlp.py : uses functions from mlp_hyperparameter_tuning.py to return best hyperparameters, and uses these hyperparameters to train an MLP and finally get metrics based on the test set 
        - mlp_hyperparameter_tuning.py : contains code for tuning hyperparameters in the MLP model. This includes the activation function, regularisation parameter, and mlp shape.
        - fisher.py: contains functions that compute fisher's linear discriminant
        - logistic_regression_functions.py: contains functions for logistic regression
        - logistic_regression_hyperparameter_tuning.py: contains functions for adjusting hyperparameters
        - metrics.py: contains functions to compute the accuracy, precision, recall, f1 score and confusion matrix
        - svm.py: functions to compute the support vector machine model     
     3. plot
        - mlp_plots.py: functions to generate plots of findings during the training of the MLP
        - exploratory_plots.py: functions to create plots prior to fitting the dataset to a model
        - fisher_plots.py: functions to generate plots of findings when fitting fishers linear discriminant
        - lr_plots.py: functions to generate plots of findings during the training phase of logistic regression
        - svm_plots.py: functions to generate plots of findings during the training phase of the SVM model
2. plots
     - This folder contains graphs produced by each model. Each model has its own separate folder. There is also a 
        separate folder for exploratory plots generated prior to fitting a model
3. main.py
     - This is the main script that should be run to view the results of all 4 models.
4. breast-cancer.data

    - The raw data set used, downloaded from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer

5. breast-cancer.names

    - Information explaining the dataset, including a description of header names and some papers that have used the dataset
    
6. requirements.txt

    - Contains the names of dependencies which should be installed prior to running main.py

