import math

import numpy as np
import pandas as pd


class Adaboost:
    def __init__(self, data_df):
        self.data = data_df
        self.metrics = pd.DataFrame({'importance': [], 'error': [], 'method': [], 'threshold': [], 'target': []})
        self.sample_size = 0

    def fit(self, iter, sample_size):
        # Inital Variables and reset the matrix variables
        self.sample_size = sample_size
        self.metrics = pd.DataFrame({'importance': [], 'error': [], 'method': [], 'threshold': [], 'target': []})
        df = self.data.copy(deep=True)
        df['weight'] = 1 / df.shape[0]
        print('_' * 100)
        print('Original Data\n', df, sep='')
        # process with iteration

        for _ in range(iter):
            print('_' * 100)
            # using dataframe sample to sample the df with replacement and weight
            sample = df.sample(self.sample_size, replace=True, weights=df.weight)
            print(f'Sample for classifier {_ + 1}\n', sample[['x', 'y', 'weight']], sep='')

            # calculate the best threshold v and its error
            split_error, threshold, method = self.best_split(sample)

            # calculate the classifier error
            error = self.classifier_error(data, float(threshold), method, df.weight.values.tolist())

            # calculate the classifier importance
            importance = (1 / 2) * np.log((1 - error) / error) if (error != 0) or (error != 0.5) else 0

            # print out and save the classifier into the class variables
            print(f'error on training {split_error}, threshold {threshold}, method: {method}, error on whole data: {error}, classifier importance: {importance}')
            config = {'importance': importance, 'error': error, 'method': method, 'threshold': float(threshold),
                      'target': 0}

            self.metrics = self.metrics.append(config, ignore_index=True)
            df = self.update_weight(df, method)

    def best_split(self, data):
        # Base on the selected data point
        # Determine the best v to classifiy the selected data point
        # from 0 to 9 totally 20 splits
        train = data.copy(deep=True)
        config = [999, 999, '']
        for i in range(train.x.min() * 10, (train.x.max() * 10) + 1, 5):
            v = i / 10
            method, err = self.split_error(train, v)
            if err <= config[0]:
                config[0] = err
                config[1] = v
                config[2] = method
        return config[0], config[1], config[2]

    def split_error(self, train, v):
        # This function is used to calculate the error obtained by the selected v
        # determine the method to split the data
        # if "method == True" means that smaller then v classify as 1
        # if "method == False" means that larger than v classify as 1
        # so the method is either True or False indicting whatever it is larger than or less than v

        train['z'] = train.x < v
        train['method1'] = -1
        train['method2'] = -1
        train.loc[train.z == True, 'method1'] = 1
        train.loc[train.z == False, 'method2'] = 1
        err1 = train[train.y != train.method1].weight.sum()
        err2 = train[train.y != train.method2].weight.sum()
        if err1 > err2:
            return False, err2
        else:
            return True, err1

    def classifier_error(self, df, threshold, method, weight):
        # Find out the classifier error by using the whole dataset.
        # sum the misclassified data point's weight over the whole data count
        data = df.copy(deep=True)
        data['weight'] = weight
        data['c'] = data.x < threshold
        data['pred'] = -1
        data.loc[data.c == method, 'pred'] = 1
        return data[data.y != data.pred].weight.sum() / data.shape[0]

    def predict(self, data):
        # For each data point input. Used the stored Adaboost classifier to predict the label.
        # This function will print out the equation and the prediction result label

        df = self.metrics.copy(deep=True)
        print('\nAll classifiers: \n', self.metrics, sep='')
        test_x = data.x.tolist()
        result = []
        for i in test_x:
            df['target'] = i
            df['pred_pre'] = df.target < df.threshold
            df['pred'] = -1
            df.loc[(df.pred_pre == True) & (df.method == True), 'pred'] = 1
            df.loc[(df.pred_pre == False) & (df.method == False), 'pred'] = 1
            df['prob'] = df['pred'] * df['importance']
            sum = df['prob'].sum()
            label = self.sign_function(sum)
            result.append(label)
            temp1 = df.importance.tolist()
            temp2 = df.pred.tolist()
            print('-' * 100, '\ninput x = ', i)
            print(f'C*(x) = sign[{temp1[0]} * {temp2[0]} + {temp1[1]} * {temp2[1]} + {temp1[2]} * {temp2[2]} + {temp1[3]} * {temp2[3]} + {temp1[4]} * {temp2[4]}] ')
            print(f'C*(x) = sign[{sum}]')
            print('Predict: ', self.sign_function(sum))
        return result

    def sign_function(self, sum):
        # Sign function formula according to the given wikipedia link
        if sum > 0:
            return 1
        elif sum == 0:
            return 0
        else:
            return -1

    def update_weight(self, df, method):
        # if predict correct lower the weight
        # if predict wrong increase the weight
        config = self.metrics.iloc[-1].tolist()
        df['threshold'] = df.x < config[3]
        df['pred'] = -1
        df.loc[df.threshold == method, 'pred'] = 1
        df.loc[df['pred'] == df['y'], 'weight'] = df['weight'] * math.exp(-config[0])
        df.loc[df['pred'] != df['y'], 'weight'] = df['weight'] * math.exp(config[0])
        df['weight'] = df['weight'] / df.weight.values.sum()
        print('After updated weight:\n', df, sep='')
        return df


if __name__ == '__main__':
    # Define data
    data = pd.DataFrame(data={
        'x': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'y': [1, 1, -1, -1, -1, 1, 1, -1, -1, 1]
    })

    # Make Adaboost instance
    Adaboost = Adaboost(data)

    # Train the model
    Adaboost.fit(iter=5, sample_size=10)

    # Try to predict with the training dataset
    a = Adaboost.predict(data)

    data['z'] = a
    print('\nPrediction', a, 'Accuracy', data[data.z == data.y].shape[0] / 10)
