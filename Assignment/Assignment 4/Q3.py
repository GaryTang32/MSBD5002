import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Nested_Loop_Outlier_Detection:
    def __init__(self, x, y, distance_threshold, fraction_threshold):
        self.x_divide = x
        self.y_divide = y
        self.distance_threshold = distance_threshold
        self.fraction_threshold = fraction_threshold
        self.no_of_partition = x * y

    def detect(self, data):
        print('-' * 100)
        print('Nested Loop Algorithm')
        print(f'Setting: distance threshold = {self.distance_threshold}, fraction threshold = {self.fraction_threshold}')

        # devide the data into different partition
        data = self.devide_data(data)

        # load first 2 partition into the slots
        first_slot, second_slot, comb_cnt = 0, 1, 0

        # get the total combination for verification
        total_comparison = math.comb(self.no_of_partition, 2)

        flag = True
        rounds = 1

        # each parition compare with itself
        data['close_point_cnt'] = 0
        for i in range(self.no_of_partition):
            data = self.compare(i, i, data)

        # loop to compare with different partition with least number of swapping.
        while comb_cnt != total_comparison:

            # both partition compare with each other
            data = self.compare(first_slot, second_slot, data)
            data = self.compare(second_slot, first_slot, data)

            # Load different paritions with least number of swapping
            print('Comparing partition :', first_slot, 'and partition :', second_slot)
            if flag & (second_slot != self.no_of_partition - rounds):
                second_slot += 1
            elif flag & (second_slot == self.no_of_partition - rounds):
                flag = ~flag
                first_slot = second_slot - 1
            elif (~flag) & (first_slot != rounds):
                first_slot -= 1
            elif (~flag) & (first_slot == rounds):
                rounds += 1
                flag = ~flag
                second_slot = first_slot + 1
            comb_cnt += 1

        # Detect the outlier
        # Set all outlier flag as false
        # if a points having less than a certain number of neighbor points, considered as outlier.
        data['outlier'] = False
        data.loc[data['close_point_cnt'] / data.shape[0] <= self.fraction_threshold, 'outlier'] = True

        # Print results

        print('Totally there are ', comb_cnt, 'comparisons (swapping). ')
        print('Totally there are ', (data.outlier == True).sum(), 'outliers. with the Nested Loop algorithm')
        return data

    # Compare two partition
    def compare(self, first_slot, second_slot, data):
        # compare all pair within the same array and other array
        temp = data.loc[data['partition'] == first_slot, ['x', 'y']].apply(
            lambda x: self.cal_distiance(x, data[data['partition'] == second_slot]), axis=1)
        data.loc[data['partition'] == first_slot, 'close_point_cnt'] = data.loc[data[
                                                                                    'partition'] == first_slot, 'close_point_cnt'] + temp
        return data

    # Calculate the distance
    def cal_distiance(self, target, data):
        dis = ((np.array(data.x) - target[0]) ** 2 + (np.array(data.y) - target[1]) ** 2) ** 0.5
        return (dis <= self.distance_threshold).sum()

    # Split the data into different partition
    def devide_data(self, data):
        slice_x = max(data.x) / self.x_divide
        slice_y = max(data.y) / self.y_divide
        data['partition'] = 0
        partition_cnt = 0

        # loop to split the data into different partitions.
        for i in range(self.x_divide):
            for j in range(self.y_divide):
                data.loc[(data['x'] > slice_x * i) &
                         (data['x'] <= slice_x * (i + 1)) &
                         (data['y'] > slice_y * j) &
                         (data['y'] <= slice_x * (j + 1)), 'partition'] = partition_cnt
                partition_cnt += 1
        return data


# get the data and assign columns name to the data
def get_data(path):
    data = pd.read_csv(path, header=None, sep=' ')
    data.columns = ['x', 'y']
    return data


# becasue there are two algo named "Nested Loop Algorithm"
# So i did both :/

class Densitiy_Based_Nested_Loop_Outlier_Detection:
    # Determine the distance threshold and the fraction threshold
    def __init__(self, distance_threshold, fraction_threhold):
        self.distance_threshold = distance_threshold
        self.fraction_threshold = fraction_threhold

    # Use nested for loop to loop throught the datasets.
    def detection(self, data):
        all_data_point = list(zip(data.x.values.tolist(), data.y.values.tolist()))
        D = data.shape[0]
        result = []
        for i in all_data_point:
            cnt = 0
            for j in all_data_point:
                if i != j:
                    cnt += 1 if self.cal_distance(i, j) else 0
                    if cnt / D >= self.fraction_threshold:
                        break
            result.append(False if (cnt / D) >= self.fraction_threshold else True)
        data['outlier'] = result
        print('-' * 100)
        print('Density Based Nested Loop Algorithm')
        print(
            f'Setting: distance threshold = {self.distance_threshold}, fraction threshold = {self.fraction_threshold}')
        print(f'There are totally {(data.outlier == True).sum()} outliers.')
        return data

    def cal_distance(self, x, y):
        return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5 < self.distance_threshold


if __name__ == '__main__':
    data = get_data('Nested_Points.txt')
    colors = {True: 'red', False: 'black'}
    ##################################################################################################################

    detector = Densitiy_Based_Nested_Loop_Outlier_Detection(20, 0.05)
    data = detector.detection(data)

    print('Outlier Location')
    print(data.loc[data['outlier'] == True, ['x', 'y', 'outlier']])

    plt.title(f"Data's outlier detection with Density Based Outlier Detection method. Red = Outlier, Black = Normal ")
    plt.scatter(data.x, data.y, c=data['outlier'].map(colors))
    plt.show()

    ##################################################################################################################

    detector = Nested_Loop_Outlier_Detection(2, 2, 20, 0.05)
    data = detector.detect(data)

    print('Outlier Location')
    print(data.loc[data['outlier'] == True, ['x', 'y', 'outlier']])

    plt.title(f"Data's partition number, 1 colour = 1 partition")
    plt.scatter(data.x, data.y, c=data['partition'])
    plt.show()

    plt.title(f"Data's outlier detection with Nested Loop outlier Detection method. Red = Outlier, Black = Normal ")
    plt.scatter(data.x, data.y, c=data['outlier'].map(colors))
    plt.show()
