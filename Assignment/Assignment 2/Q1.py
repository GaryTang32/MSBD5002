import time
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


def get_data():
    # Read the training and testing data
    df = pd.read_csv(r'dataset/winequality_test.csv', sep=';')
    df2 = pd.read_csv(r'dataset/winequality_train.csv', sep=';')
    return df.iloc[:, :-1], df.iloc[:, -1], df2.iloc[:, :-1], df2.iloc[:, -1]


def evaluate(pred, test_y):
    accuracy = accuracy_score(test_y, pred)
    precision = precision_score(test_y, pred, average="macro")
    recall = recall_score(test_y, pred, average="macro")
    f1 = f1_score(test_y, pred, average="macro")
    return accuracy, precision, recall, f1


def DecisionTree(depth, method, train_x, train_y, test_x, test_y, eval_df):

    # Training and testing of the model.
    DT = DecisionTreeClassifier(criterion=method, max_depth=depth)
    start = time.time()
    DT.fit(train_x, train_y)
    stop = time.time()
    pred = DT.predict(test_x)

    # Get the classification report
    # report = classification_report(test_y, pred, output_dict=True)

    # Get the average performance of the multi-classification problem
    overall_accuracy, precision, recall, f1_score = evaluate(test_y, pred)

    # Save the performance of this result into the dataframe
    # There are toally 7 label from '3' to '9', so divide the result with 7

    a = {'classifier': 'Decision Tree',
         'accuracy': overall_accuracy,
         'precision': precision,
         'recall': recall,
         'f1': f1_score,
         'training_time': stop - start,
         'setting': f'depth: {depth} method: {method}'}

    return eval_df.append(a, ignore_index=True)


def RandomForest(n_estimator, depth, method, train_x, train_y, test_x, test_y, eval_df):
    # Define the parameter and the model
    RF = RandomForestClassifier(n_estimators=n_estimator, criterion=method, max_depth=depth, n_jobs=3)

    # Start to train, predict and calculate the classification report
    start = time.time()
    RF.fit(train_x, train_y)
    stop = time.time()
    pred = RF.predict(test_x)

    # Calculate the average metrics for all label
    overall_accuracy, precision, recall, f1_score = evaluate(test_y, pred)

    # save the result into dataframe
    a = {
        'classifier': 'RF',
        'accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'training_time': stop - start,
        'setting': f'n_estimators: {n_estimator} depth: {depth} method: {method}'
    }
    return eval_df.append(a, ignore_index=True)


def KNN(n_neighbors, test_x, test_y, train_x, train_y, eval_df):
    # Define the parameter and the model
    KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Start to train, predict and calculate the classification report
    start = time.time()
    KNN.fit(train_x, train_y)
    stop = time.time()
    pred = KNN.predict(test_x)

    # Calculate the average metrics for all label
    overall_accuracy, precision, recall, f1_score = evaluate(test_y, pred)

    # save the result into dataframe
    a = {
        'classifier': 'KNN',
        'accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'training_time': stop - start,
        'setting': f'n_neighbor: {n_neighbors}'
    }
    return eval_df.append(a, ignore_index=True)


if __name__ == '__main__':
    test_x, test_y, train_x, train_y = get_data()
    dt_eval_df = pd.DataFrame({
        'classifier': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'training_time': [],
        'setting': []
    })

    rf_knn_eval_df = pd.DataFrame({
        'classifier': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'training_time': [],
        'setting': []
    })

    # Call decision tree for task 1
    for i in ['gini', 'entropy']:
        for j in range(5, 25, 5):
            dt_eval_df = DecisionTree(j, i, test_x, test_y, train_x, train_y, dt_eval_df)

    # Call Random Forest and KNN for task2
    for i in ["gini", "entropy"]:
        for j in range(10, 25, 5):
            for k in range(50, 250, 50):
                rf_knn_eval_df = RandomForest(k, j, i, test_x, test_y, train_x, train_y, rf_knn_eval_df)

    for i in range(1, 6, 1):
        rf_knn_eval_df = KNN(i, test_x, test_y, train_x, train_y, rf_knn_eval_df)

    print('Decision Tree\n', dt_eval_df, sep='')
    print('-' * 100)
    print('KNN and RF\n', rf_knn_eval_df, sep='')
