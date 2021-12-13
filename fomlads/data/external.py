import numpy as np
import pandas as pd


def import_and_process(fname, input_cols=None, target_col=None):
    """
    Imports a dataset, pre-processes it and splits it into train and test parts.

    :param fname: Filename/path of data file.
    :param input_cols: List of column names for the input data.
    :param target_col: Column name of the target data.

    :return:
    inputs_train_crossval - train inputs data split into sub-arrays for cross-validation
    targets_train_crossval - train targets data split into sub-arrays for cross-validation
    inputs_train -- the whole training inputs data as a numpy.array object
    targets_train -- the whole training targets data as a 1d numpy array of class ids
    inputs_test -- the test inputs data as a numpy.array object
    targets_test -- the test targets data as a 1d numpy array of class ids
    input_cols -- ordered list of input column names
    classes -- ordered list of classes
    """
    # if no file name is provided then use synthetic data
    df = pd.read_csv(fname)
    # print("dataframe.columns = %r" % (dataframe.columns,) )
    N = df.shape[0]

    # pre-process and encode data
    df_onehot = pre_process_bc(df)

    # if no target name is supplied we assume it is the last column in the
    # data file
    if target_col is None:
        target_col = df_onehot.columns[-1]
        potential_inputs = df_onehot.columns[:-1]
    else:
        potential_inputs = list(df_onehot.columns)
        # print([potential_inputs])
        # target data should not be part of the inputs
        potential_inputs.remove(target_col)
    # if no input names are supplied then use them all
    if input_cols is None:
        input_cols = potential_inputs
    # get the class values as a pandas Series object
    class_values = df_onehot[target_col]
    classes = class_values.unique()

    # Split targets from inputs
    ys = df_onehot[target_col]
    xs = df_onehot[input_cols]

    # Split data into train and test parts (stratified)
    x_train, x_test, y_train, y_test = stratified_split(ys, xs, test_part=0.2)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # We now want to translate classes to targets, but this depends on our
    # encoding. For now we will perform a simple encoding from class to integer.
    # We do this for each of our k data splits
    targets_train = np.empty(len(y_train))
    targets_test = np.empty(len(y_test))
    for class_id, class_name in enumerate(classes):
        is_class_train = (y_train == class_name)
        is_class_test = (y_test == class_name)
        targets_train[is_class_train] = class_id
        targets_test[is_class_test] = class_id

    # We're going to assume that all our inputs are real numbers (or can be
    # represented as such), so we'll convert all these columns to a 2d numpy array object
    inputs_train = x_train.values
    inputs_test = x_test.values

    # Split the training data into 5 for cross validation
    data_split = stratified_split_k_fold(y_train, x_train, 5)
    targets_train_crossval, inputs_train_crossval = [], []

    # We repeat the same step for the cross-validation data
    for i in range(len(data_split)):
        y_train = np.empty(len(data_split[i][2]))
        y_valid = np.empty(len(data_split[i][3]))
        for class_id, class_name in enumerate(classes):
            is_class_train = (data_split[i][2] == class_name)
            is_class_valid = (data_split[i][3] == class_name)
            y_train[is_class_train] = class_id
            y_valid[is_class_valid] = class_id
        targets_train_crossval.append({"train": y_train, "valid": y_valid})

        x_train = data_split[i][0].values
        x_valid = data_split[i][1].values
        inputs_train_crossval.append({"train": x_train, "valid": x_valid})

    return inputs_train_crossval, targets_train_crossval, inputs_train, targets_train, inputs_test, targets_test, \
           input_cols, classes


def pre_process_bc(df):
    """
    Pre-processing function specific to the breast-cancer dataset.

    :param df: DataFrame of  breast cancer data.

    :return df_onehot: Processed df with encoded categories and missing values replaced.
    """
    # some values are missing, replace them with mode
    modes = df.mode()
    mode_n_caps = modes["node-caps"][0]
    mode_b_quad = modes["breast-quad"][0]
    replace_map = {"node-caps": {"?": mode_n_caps},
                   "breast-quad": {"?": mode_b_quad},
                   "class": {"no-recurrence-events": 0,
                             "recurrence-events": 1}
                   }
    df.replace(replace_map, inplace=True)

    # get the names of the headers
    headers = df.columns

    # currently data is categorical, encode the data so models can be applied
    # columns 1,2,3,4,5,7,8,9 need encoding
    # different types of encoding for certain cols
    # ordinal features can used number encoding (cols: 1, 3, 4)
    # non-ordinal features with more than 2 categories -> one-hot (cols: 2, 8)
    # non-ordinal features with 2 categories -> binary (cols: 5, 7, 9)

    # encoding with numbers automatically makes binary columns, so binary encoding and number encoding can be done
    # together
    headers_to_encode = headers[[1, 3, 4, 5, 7, 9]]
    for header in headers_to_encode:
        df[header] = df[header].astype("category")
        df[header] = df[header].cat.codes

    # encoding columns 2 (menopause) and 8 (breast-quad) to make one-hot vectors
    df_onehot = df.copy()
    df_onehot = pd.get_dummies(df_onehot, columns=['menopause', 'breast-quad'], prefix=['menopause', 'breast-quad'])

    return df_onehot


def stratified_split(y, x, test_part=0.2):
    """

    :param y: A pandas Series of target values. Assumes two classes, encoded as 0 and 1.
    :param x: A pandas DataFrame of input values.
    :param test_part: A float representing the percentage of data which should be used as test data (default 20%).

    :return : y and x split into train and test parts.
    """

    # Perform split on the two classes separately, due to class imbalance
    class0 = y[y == 0]
    class1 = y[y == 1]

    # Arrange and shuffle dataset (within each class)
    index_array_c0 = class0.index.values
    index_array_c1 = class1.index.values
    np.random.shuffle(index_array_c0)
    np.random.shuffle(index_array_c1)

    # Within each class, split dataset into two parts determined by the test_part param
    idx_test_class0, idx_train_class0 = np.array_split(index_array_c0, [int(test_part * len(index_array_c0))])
    idx_test_class1, idx_train_class1 = np.array_split(index_array_c1, [int(test_part * len(index_array_c1))])

    # Merge back into single array
    idx_test = np.concatenate((idx_test_class0, idx_test_class1))
    idx_train = np.concatenate((idx_train_class0, idx_train_class1))

    # Get the x_train and x_test using the indices
    x_test = x.iloc[idx_test]
    x_train = x.iloc[idx_train]

    # Same for y
    y_test = y.iloc[idx_test]
    y_train = y.iloc[idx_train]

    return x_train, x_test, y_train, y_test


def stratified_split_k_fold(y, x, k=5):
    """
    Given inputs and targets, splits them into k folds in a stratified manner.

    :param y: A pandas Series of target values. Assumes two classes, encoded as 0 and 1.
    :param x: A pandas DataFrame of input values.
    :param k: An integer representing the number of folds to split the data in.

    :return data_split: A list containing k sub-lists, each having 4 arrays of train and test inputs,
    and train and test targets.
    """
    data_split = []

    # Perform k-fold split on the two classes separately, due to class imbalance
    class0 = y[y == 0]
    class1 = y[y == 1]

    # Arrange and shuffle dataset (within each class)
    index_array_c0 = class0.index.values
    index_array_c1 = class1.index.values
    np.random.shuffle(index_array_c0)
    np.random.shuffle(index_array_c1)

    # Split dataset into k near-equal parts (test parts) within each class, then merge back into single array
    test_idx_split_0 = np.array_split(index_array_c0, k)
    test_idx_split_1 = np.array_split(index_array_c1, k)
    test_idx_split = [np.concatenate((test_idx_split_0[i], test_idx_split_1[i])) for i in range(k)]

    # For each part of the above split, use it as a holdout
    for array in test_idx_split:
        # Test part is the array of indexes
        x_test = x.iloc[array]
        # Train part is all other indexes
        x_test_idx = x.index.isin(array)
        x_train = x.iloc[~x_test_idx]

        # Same for y
        y_test = y.iloc[array]
        y_test_idx = y.index.isin(array)
        y_train = y.iloc[~y_test_idx]

        data_split.append([x_train, x_test, y_train, y_test])

    return data_split
