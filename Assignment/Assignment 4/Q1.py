import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Fuzzy_Clustering_EM:
    # Define the initial class attributes
    def __init__(self, iteration, k):
        self.num_of_cluster = k
        # if iteration is auto, then assign a very large number for iteration to mimic loop until converage
        self.iteration = 99999 if iteration == 'auto' else int(iteration)
        self.centroid = [[] for _ in range(self.num_of_cluster)]
        self.cluster_SSE = [0 for _ in range(self.num_of_cluster)]

    def cal_distance(self, data_x, data_y, cluster):
        # Calculate the denominator
        denominator = np.array([0.0 for i in range(len(data_x))])
        for i in range(self.num_of_cluster):
            denominator += 1.0 / ((data_x - self.centroid[i][0]) ** 2.0 + (data_y - self.centroid[i][1]) ** 2.0)

        # Calculate the distance between different cluster
        dis = 1 / ((data_x - self.centroid[cluster][0]) ** 2.0 + (data_y - self.centroid[cluster][1]) ** 2.0)
        dis /= denominator
        dis = np.nan_to_num(dis, nan=1)
        return dis

    def print_centroid(self, iter_cnt):
        # Print out all the centroid and values
        print('-' * 60, '\niteration : ', iter_cnt)
        for i in range(self.num_of_cluster):
            print(
                f'Cluster {i + 1}: x={self.centroid[i][0]: 20}  y= {self.centroid[i][1]:20}, SSE= {self.cluster_SSE[i]:10}')

    def fit(self, data):
        # Convert the data into numpy array, more easy for calculation
        data_x = np.array(data.x.values.tolist())
        data_y = np.array(data.y.values.tolist())

        update_magnitude = 100
        iter_cnt = 0

        # Initialization take the first three points as the initial centroid
        for i in range(self.num_of_cluster):
            self.centroid[i] = [data_x[i], data_y[i]]

        # Print the initial cluster info
        self.print_centroid(0)

        # Loop and update the centroid
        # stop when either the centriod update is <= 0.000005 or reach the target iteration
        while (update_magnitude > 0.000005) and (iter_cnt != self.iteration):
            iter_cnt += 1

            # calculate the E step
            matrix = []
            update_magnitude = 0

            # Calculate each points to each cluster distance
            for i in range(self.num_of_cluster):
                matrix.append(self.cal_distance(data_x, data_y, i))

            # calculate the M step
            # update the centroid one by one and calculate the update magnitude
            for i in range(self.num_of_cluster):
                temp = [sum(matrix[i] ** 2 * data_x) / sum(matrix[i] ** 2),
                        sum(matrix[i] ** 2 * data_y) / sum(matrix[i] ** 2)]

                # the update magnitude
                update_magnitude += abs(temp[0] - self.centroid[i][0]) + abs(temp[1] - self.centroid[i][1])

                # Update the centroid
                self.centroid[i] = temp
                self.cluster_SSE[i] = self.cal_SSE(data_x, data_y, matrix[i], self.centroid[i])

            # print the new centriod
            self.print_centroid(iter_cnt)
        return self.centroid

    # Calculate the SSE
    def cal_SSE(self, data_x, data_y, matrix, centroid):
        return sum(((data_x - centroid[0]) ** 2 + (data_y - centroid[1]) ** 2) * matrix)

    def predict(self, data):
        # Predict the data point with the closest
        cols = []

        # Calculate each points to each cluster's distance
        for i in range(self.num_of_cluster):
            temp = 'cluster_' + str(i)
            cols.append(temp)
            data[temp] = (data['x'] - self.centroid[i][0]) ** 2 + (data['y'] - self.centroid[i][1]) ** 2

        # select the cluster with the minimun distance to the data point
        data['label_pred'] = data[cols].apply(lambda x: x.argmin(), axis=1)
        return data


# Get data from the txt and assign back the columns name to it
def get_data(data_path):
    data = pd.read_csv(data_path, header=None, sep=' ')
    data.columns = ['x', 'y', 'label']
    return data


if __name__ == '__main__':
    Fuzzy_EM = Fuzzy_Clustering_EM('auto', 3)
    data = get_data('EM_Points.txt')
    centriod = Fuzzy_EM.fit(data)
    data = Fuzzy_EM.predict(data)

    # Print out the results
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Data clustering label comparison: True Label(Left), Pred Label(Right)")
    ax1.scatter(data.x, data.y, c=data.label)
    ax2.scatter(data.x, data.y, c=data.label_pred)
    fig.set_size_inches(15, 10)
    plt.show()
