import copy
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Define the binary classification neual network
class bi_Net(nn.Module):
    def __init__(self, input_n, output_n, hidden_n, device):
        super(bi_Net, self).__init__()
        # This is the forward stack for the network with weight at each layer
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n, bias=True),  # from the input layer to the hidden layer
            torch.nn.ReLU(),  # Use rectified linear unit function as activation function
            # torch.nn.Softmax(),
            # torch.nn.Sigmoid(),
            # torch.nn.Tanh(),
            torch.nn.Linear(hidden_n, output_n, bias=True),  # From the hidden layer to the output layer.
            # torch.nn.Softmax()
            torch.nn.Sigmoid()  # finally aggregate the result into a signmoid function as the output value.
        ).to(device)  # To CPU device

    def forward(self, x):
        # Declear the forward function
        x = self.stack(x)
        return x


# define the multi-classification neural network
class multi_Net(nn.Module):
    def __init__(self, input_n, output_n, hidden_n1, hidden_n2, device):
        super(multi_Net, self).__init__()
        # This is the forward stack for the network with weight at each layer
        # this
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n1, bias=True),  # from the input layer to the 1st hidden layer
            torch.nn.ReLU(),  # Use rectified linear unit as activation function
            torch.nn.Linear(hidden_n1, hidden_n2, bias=True),  # From the 1st hidden later to the 2nd hidden layer
            torch.nn.ReLU(),  # Use rectified linear unit as activaition function
            torch.nn.Linear(hidden_n2, output_n, bias=True),  # form the 2nd hidden layer to the output layer
            torch.nn.Sigmoid()  # Use sigmoid as activation function for all output notes.
        ).to(device)

    def forward(self, x):
        x = self.stack(x)
        return x


# read the data from the numpy zip
def read_data(path, objective):
    if objective == 'bi-class':
        raw_data = np.load(path)
        test_X = torch.from_numpy(raw_data['test_X']).type(torch.FloatTensor)
        test_y = torch.from_numpy(raw_data['test_Y']).type(torch.FloatTensor)
        train_X = torch.from_numpy(raw_data['train_X']).type(torch.FloatTensor)
        train_y = torch.from_numpy(raw_data['train_Y']).type(torch.FloatTensor)
    else:
        raw_data = np.load(path)
        test_X = torch.from_numpy(raw_data['test_X']).type(torch.FloatTensor)
        test_y = torch.from_numpy(raw_data['test_y']).type(torch.LongTensor)
        train_X = torch.from_numpy(raw_data['train_X']).type(torch.FloatTensor)
        train_y = torch.from_numpy(raw_data['train_y']).type(torch.LongTensor)
    return test_X, test_y, train_X, train_y


# Evaluate the binary classifier output
# Including the auc, accuracy, precision, recall, f1-score
def evaluate(y_pred, y_label, objective):
    if objective == 'bi-class':
        auc = roc_auc_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        acc = (y_label == y_pred).sum().item() / y_label.shape[0]
        precision = precision_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        recall = recall_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        f1 = f1_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        return y_pred, y_label, acc, precision, recall, f1, auc
    else:
        acc = (y_label == y_pred).sum().item() / y_label.shape[0]
        precision = precision_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        recall = recall_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        f1 = f1_score(y_label.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average="macro")
        return y_pred, y_label, acc, precision, recall, f1


# plot the training,  the validation loss, validation accuracy for each H and k-fold dataset.
# for each if there are 10 H to try, then will plot 5 times for each H. So 1 problem will generate
# 50 graphs.
def plot_result(loss_train_list, loss_valid_list, acc_list, neu_num):
    # define the plot diamension
    fig = plt.figure(figsize=(10, 4))
    g1 = plt.subplot(1, 3, 1)
    g2 = plt.subplot(1, 3, 2)
    g3 = plt.subplot(1, 3, 3)

    #
    plt.sca(g1)
    plt.title(f'H = {neu_num}')
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.plot(range(len(loss_train_list)), loss_train_list)

    plt.sca(g2)
    plt.title(f'H = {neu_num}')
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.plot(range(len(loss_valid_list)), loss_valid_list)

    plt.sca(g3)
    plt.title(f'H = {neu_num}')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(range(len(acc_list)), acc_list)
    plt.show()


# after finding the best H* with the k-fold cross validation.
# Retrain another new model with the selected H* to all the training sampele. Then test it with the testing set.
# finally output the accuracy and the training time.
def train_predict_with_best_h(record, train_X, train_y, test_X, test_y, iter, best_h_value, device, learning_rate, objective):
    print('-' * 100)
    print('Performance of each H:')
    print(record)
    print('-' * 100)
    # define the model for the classification problem. Different problem have different neural network.
    if objective == 'bi-class':
        final_model = bi_Net(train_X.shape[1], 1, best_h_value, device)
        loss = torch.nn.BCELoss()
    else:
        final_model = multi_Net(train_X.shape[1], 10, best_h_value[0], best_h_value[1], device)
        loss = torch.nn.CrossEntropyLoss()
    # define the optimizer using the gradient decent algorithm
    opt = torch.optim.SGD(final_model.parameters(), lr=learning_rate)

    # Train the model with all the testing samples.
    start = time.time()
    final_model.train()
    for iter_num in range(iter):
        y_train_pred = final_model(train_X).squeeze(dim=-1)
        loss_val = loss(y_train_pred, train_y)

        # update the weight and back propagation
        opt.zero_grad()
        loss_val.backward()
        opt.step()

        # if loss_val < 0.2:
        #     break
    end = time.time()
    training_time = end - start

    # predict the testing set.
    y_final_pred = final_model(test_X).squeeze(dim=-1)

    # calculate the accuracy on the testing record.
    if objective == 'bi-class':
        y_final_pred[y_final_pred >= 0.5] = 1
        y_final_pred[y_final_pred < 0.5] = 0
        acc = (test_y == y_final_pred).sum().item() / test_y.shape[0]
    else:
        test_y_pred_label = torch.argmax(y_final_pred, dim=1)
        acc = (test_y == test_y_pred_label).sum().item() / test_y.shape[0]

    # print out the final result.
    print('The H* is : ', best_h_value)
    print(f'final retrained  H:{best_h_value}, accuracy:{round(acc, 4):4}, training time:{round(training_time, 4):5}')


# after 5 k-fold cross validation
# calculate the average performance over the 5 k-fold.
# the result will be the performance of this "H"
def get_nn_k_fold_accuracy(auc_list, acc_list, precision_score_list, recall_list, f1_score_list, objective):
    accuracy_h = sum(acc_list) / len(acc_list)
    precision_h = sum(precision_score_list) / len(precision_score_list)
    recall_h = sum(recall_list) / len(recall_list)
    f1_h = sum(f1_score_list) / len(f1_score_list)
    if objective == 'bi-class':
        auc_h = sum(auc_list) / len(auc_list)
        return auc_h, accuracy_h, precision_h, recall_h, f1_h
    else:
        return accuracy_h, precision_h, recall_h, f1_h


# training a binary classification problem.
def bi_class_nn(test_X, test_y, train_X, train_y, H_range, kf, learning_rate, iter, device, output_neu=1):
    # define the evaluation matrix
    record = pd.DataFrame({'H': {}, 'Valid_ROC': {}, 'Valid_Accuracy': {}, 'Valid_Precision': {}, 'Valid_Recall': {}, 'Valid F1_Score': {}, 'CV': {}})

    # some variable for the best configuration
    best_cv_acc, best_h_value = 0, 0

    # loss function
    # loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.BCELoss()

    # loop through all the H and test it with the cross validation
    for neu_num in range(1, H_range + 1):
        cnt = 0
        best_model_trained = None  # used to save the best model for evaluation
        acc_list, f1_score_list, precision_score_list = [], [], []  # Store the evaluation matrix
        recall_list, auc_list, time_list = [], [], []  # Store the evaluation matrix

        for train_index, valid_index in kf.split(train_X, train_y):
            cnt += 1  # K-fold count

            # get the training and validation set.
            X_train, X_valid = train_X[train_index], train_X[valid_index]
            y_train, y_valid = train_y[train_index], train_y[valid_index]

            # define the binary classification neural network.
            model = bi_Net(train_X.shape[1], output_neu, neu_num, device)

            # define the optimizer with gradient decent
            opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

            # different list to sotre the intermediate result.
            loss_train_list, loss_valid_list, kf_acc_list = [], [], []

            # store the intermediate result
            best_acc, best_iter = 0, 0

            start = time.time()
            for iter_num in range(iter):

                # enter training mode to train with the training set.
                model.train()
                # Predict and Calculate the training loss value
                y_train_pred = model(X_train).squeeze(dim=-1)
                loss_train_val = loss(y_train_pred, y_train)
                loss_train_list.append(loss_train_val)

                # enter evaluation mode so that it will not update the gradient decent
                # because validation set is used to validate and testing, so the model cannot
                # update the weighting by the validation set.
                model.eval()
                with torch.no_grad():
                    # Predict and Calculate the validation(test) loss value
                    y_valid_pred = model(X_valid).squeeze(dim=-1)
                    loss_valid_val = loss(y_valid_pred, y_valid)
                    loss_valid_list.append(loss_valid_val)

                # get back to training mode to update the weight
                model.train()
                # Update the weight
                opt.zero_grad()
                loss_train_val.backward()
                opt.step()

                # Calculate the accracy on the validation set
                y_valid_pred[y_valid_pred >= 0.5] = 1
                y_valid_pred[y_valid_pred < 0.5] = 0
                acc = (y_valid == y_valid_pred).sum().item() / y_valid.shape[0]
                kf_acc_list.append(acc)

                # if this is the best accuracy of this k-fold trals, save the model for further evaluation
                if acc > best_acc:
                    best_acc = acc
                    best_iter = iter_num
                    best_model_trained = copy.deepcopy(model)

                # if the training loss is low, terminate the training prevent overfitting
                # if loss_train_val < 0.2:
                #     break

            end = time.time()
            training_time = end - start

            # plot the loss and the validation accuracy
            plot_result(loss_train_list, loss_valid_list, kf_acc_list, neu_num)

            # Enter evaluate mode, so that it will not affect the weighting.
            # Predict with the validation set to get the best performance under this H and k-fold.
            best_model_trained.eval()
            y_valid_pred = best_model_trained(X_valid).squeeze(dim=-1)
            # Calculate the evalation matrix
            y_valid_pred, y_valid, acc, precision, recall, f1, auc = evaluate(y_valid_pred, y_valid, 'bi-class')

            # save the evaluation result of this H and k-fold dataset.
            acc_list.append(acc)
            f1_score_list.append(f1)
            precision_score_list.append(precision)
            recall_list.append(recall)
            auc_list.append(auc)
            time_list.append(training_time)

            # print out the intermediate result for report evaluation
            print(f'H:{neu_num:2}, Kfold:{cnt:3}, acc:{round(acc, 3):5}, f1_score:{round(f1, 3):5}, precision:{round(precision, 3):5}, recall:{round(recall, 3):5} auc:{round(auc, 3):5}, iteration:{best_iter:3}, training time:{round(training_time, 4):5}')

        # find the average performance on all K-fold trials. The averaged value is the overall performance of this H.
        auc_h, accuracy_h, precision_h, recall_h, f1_h = get_nn_k_fold_accuracy(auc_list, acc_list, precision_score_list, recall_list, f1_score_list, 'bi-class')

        # save the record and performance.
        record_item = {'H': neu_num, 'Valid_ROC': auc_h, 'Valid_Accuracy': accuracy_h, 'Valid_Precision': precision_h, 'Valid_Recall': recall_h, 'Valid F1_Score': f1_h, 'CV': 5}
        record = record.append(record_item, ignore_index=True)

        # print out the performance of this H to the user.
        print(f'H={neu_num:2}, CV=5 : Kold Avg Accuracy:{round(accuracy_h, 3):4}, Kfold Avg Training Time: {round(sum(time_list) / len(time_list), 4):5}')

        # if this H have the higher average accuracy on all the k-fold datset, then pick this H as H*
        if accuracy_h > best_cv_acc:
            best_cv_acc = accuracy_h
            best_h_value = neu_num

    # finally retrain the model with the selected H* and output the performance to the user.
    train_predict_with_best_h(record, train_X, train_y, test_X, test_y, iter, best_h_value, device, learning_rate, 'bi-class')


# predicting the multi-class classification problem.
def multi_class_nn(test_X, test_y, train_X, train_y, H_range, kf, learning_rate, iter, device, output_neu=10):
    # define the evaluation matrix
    record = pd.DataFrame({'H1': {}, 'H2': {}, 'Valid_Accuracy': {}, 'Valid_Precision': {}, 'Valid_Recall': {}, 'Valid F1_score': {}, 'CV': {}})
    # some variable for the best configuration
    best_cv_acc, best_h1_value, best_h2_value = 0, 0, 0

    # loss function
    loss = torch.nn.CrossEntropyLoss()

    # loop through all the H1, H2 and test it with the k-fold cross validation
    for neu_num1 in H_range[0]:
        for neu_num2 in H_range[1]:
            cnt = 0
            best_model_trained = None  # used to save the best model for evaluation
            acc_list, f1_score_list, precision_score_list = [], [], []  # Store the evaluation matrix
            recall_list, time_list = [], []  # Store the evaluation matrix

            for train_index, valid_index in kf.split(train_X, train_y):
                cnt += 1  # K-fold count

                # get the training and validation set.
                X_train, X_valid = train_X[train_index], train_X[valid_index]
                y_train, y_valid = train_y[train_index], train_y[valid_index]

                # define the multi-class classification neural network.
                model = multi_Net(train_X.shape[1], output_neu, neu_num1, neu_num2, device)

                # define the optimizer with gradient decent
                opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

                # different list to sotre the intermediate result.
                loss_train_list, loss_valid_list, acc_list_kf = [], [], []

                # store the intermediate result
                best_acc, best_iter = 0, 0

                start = time.time()
                for iter_num in range(iter):

                    # enter training mode to train with the training set.
                    model.train()
                    # Predict and Calculate the training loss value
                    y_train_pred = model(X_train).squeeze(dim=-1)
                    loss_train_val = loss(y_train_pred, y_train)
                    loss_train_list.append(loss_train_val)

                    # enter evaluation mode and no_grad so that it will not update the gradient decent
                    # because validation set is used to validate and testing, so the model cannot
                    # update the weighting by the validation set.
                    model.eval()
                    with torch.no_grad():
                        # Predict and Calculate the validation(test) loss value
                        y_valid_pred = model(X_valid).squeeze(dim=-1)
                        loss_valid_val = loss(y_valid_pred, y_valid)
                        loss_valid_list.append(loss_valid_val)

                    # get back to training mode to update the weight
                    model.train()

                    # Update the weight
                    opt.zero_grad()
                    loss_train_val.backward()
                    opt.step()

                    # if the training loss is low, terminate the training prevent overfiting
                    if loss_train_val < 0.2:
                        break

                    # Calculate the accuracy on the validation set
                    y_valid_pred_label = torch.argmax(y_valid_pred, dim=1)
                    acc = (y_valid == y_valid_pred_label).sum().item() / y_valid.shape[0]
                    acc_list_kf.append(acc)

                    # if this is the best accuracy of this k-fold trals, save the model for further evaluation
                    if acc > best_acc:
                        best_acc = acc
                        best_iter = iter_num
                        best_model_trained = copy.deepcopy(model)

                end = time.time()
                training_time = end - start

                # plot the loss and the validation accuracy
                plot_result(loss_train_list, loss_valid_list, acc_list_kf, [neu_num1, neu_num2])

                # Enter evaluate mode, so that it will not affect the weighting.
                # Predict with the validation set to get the best performance under this H and k-fold.
                best_model_trained.eval()
                y_valid_pred = best_model_trained(X_valid).squeeze(dim=-1)
                y_valid_pred_label = torch.argmax(y_valid_pred, dim=1)

                # Calculate the evalation matrix
                y_valid_pred_label, y_valid, acc, precision, recall, f1 = evaluate(y_valid_pred_label, y_valid, 'multi-class')

                # save the evaluation result of this H and k-fold dataset.
                acc_list.append(acc)
                precision_score_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1)
                time_list.append(training_time)

                # print out the intermediate result for report evaluation
                print(f'H1:{neu_num1:2}, H2:{neu_num2}, Kfold:{cnt:3}, acc:{round(acc, 3):5}, f1_score:{round(f1, 3):5}, precision:{round(precision, 3):5}, recall:{round(recall, 3):5}, iteration:{best_iter:3}, training time:{round(training_time, 4):5}')

            # find the average performance on all K-fold trials. The averaged value is the overall performance of this H.
            accuracy_h = sum(acc_list) / len(acc_list)
            accuracy_h, precision_h, recall_h, f1_h = get_nn_k_fold_accuracy(None, acc_list, precision_score_list, recall_list, f1_score_list, 'multi-class')
            # save the record and performance.
            record_item = {'H1': neu_num1, 'H2': neu_num2, 'Valid_Accuracy': accuracy_h, 'Valid_Precision': precision_h, 'Valid_Recall': recall_h, 'Valid F1_score': f1_h, 'CV': 5}
            record = record.append(record_item, ignore_index=True)

            # print out the performance of this H to the user.
            print(f'H1={neu_num1:2}, H2={neu_num2}, CV=5 : Kold Avg Accuracy:{round(accuracy_h, 3):4}, Kfold Avg Training Time: {round(sum(time_list) / len(time_list), 4):5}')

            # if this H have the higher average accuracy on all the k-fold datset, then pick this H as H*
            if accuracy_h > best_cv_acc:
                best_cv_acc = accuracy_h
                best_h1_value = neu_num1
                best_h2_value = neu_num2

    # finally retrain the model with the selected H* and output the performance to the user.
    train_predict_with_best_h(record, train_X, train_y, test_X, test_y, iter, [best_h1_value, best_h2_value], device, learning_rate, 'multi-class')


# function used as a controller for the main program to call the training and prediction
def run(path: str, range, objective: str, learning_rate: float, iteration: int) -> None:
    # get the data set once, so taht no need to call get data again and again for each H and k-fold
    test_X, test_y, train_X, train_y = read_data(path, objective)

    # Using Kfold to cross validate (CV) the model
    # To make the result re-produceable and the same so the random state is set
    # the k-fold is shuffled, because the instruction said need to be chosen randomly
    kf = KFold(n_splits=5, random_state=32, shuffle=True)

    # we will do the training and prediction on the CPU device
    device = torch.device('cpu')

    # predict with the binary class or the multi-class
    if objective == 'bi-class':
        bi_class_nn(test_X, test_y, train_X, train_y, range, kf, learning_rate, iteration, device)
    elif objective == 'multi-class':
        multi_class_nn(test_X, test_y, train_X, train_y, range, kf, learning_rate, iteration, device)


# main program to call different datset.
if __name__ == '__main__':
    print('-' * 100)
    print('Binary Classification Problem')
    print('-' * 100)

    print('Iris Dataset.')
    path = r'datasets\bi-class\iris.npz'
    run(path, 10, 'bi-class', 0.05, 250)
    print('-' * 100)

    print('-' * 100)
    print('Breast Cancer Dataset')
    path = r'datasets\bi-class\breast-cancer.npz'
    run(path, 10, 'bi-class', 0.01, 2000)
    print('-' * 100)

    print('-' * 100)
    print('Diabetes Dataset')
    path = r'datasets\bi-class\diabetes.npz'
    run(path, 10, 'bi-class', 0.05, 2500)
    print('-' * 100)

    print('-' * 100)
    print('Wine Dataset')
    path = r'datasets\bi-class\wine.npz'
    run(path, 10, 'bi-class', 0.05, 500)
    print('-' * 100)

    print('-' * 100)
    print('Multi-class Classification Problem')

    print('-' * 100)
    print('Digits Datset')
    path = r'datasets\multi-class\digits.npz'
    run(path, [[15, 25, 50], [5, 10, 20]], 'multi-class', 0.05, 1500)
    print('-' * 100)
