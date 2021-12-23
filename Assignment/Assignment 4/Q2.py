import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DBSCAN():
    # Constructor
    def __init__(self, min_pts, eps):
        self.min_pts = min_pts
        self.eps = eps
        self.num_of_cluster = 0

    # Fit the data and find the label
    def fit(self, data):
        # Use np array for the calculation
        data_x = np.array(data.x)
        data_y = np.array(data.y)

        # Find the neighbor index of the data point.
        data['direct_reachable'] = data[['x', 'y']].apply(lambda x: self.find_N_eps(data_x, data_y, [x[0], x[1]]),
                                                          axis=1)

        # Find the total number of neighbor of the data point
        data['N_eps'] = data['direct_reachable'].apply(lambda x: len(x))

        # Determine whatever the data point is a core point.
        data['core_pts'] = data['N_eps'] >= self.min_pts

        # visited marker
        data['visited'] = 0

        # Final prediction table, -1 = outlier
        data['label'] = -1

        # loop to form the cluster
        while ((data['core_pts'] == True) & (data['visited'] == 0)).any():
            # Pick a new unvisited core point as the new cluster's starting point
            current_point = data[(data['core_pts'] == True) &
                                 (data['visited'] == 0)].index.tolist()[0]

            # find out all its neighbor
            cluster_set = set(data.loc[current_point, 'direct_reachable'])

            # loop to get all data point in the same cluster
            while True:

                cluster_set_cnt = len(cluster_set)

                # if the data point is core point add all its neighbor to the cluster
                temp = data.loc[list(cluster_set), :]
                data_point_reachable_list = temp.loc[temp['core_pts'] == True, 'direct_reachable'].values.tolist()

                # use set to get all the data point in the same cluster
                for i in data_point_reachable_list:
                    cluster_set.update(i)

                # Stop looping when all data point in the same cluster is found
                if cluster_set_cnt == len(cluster_set):
                    break
            # Assign new cluster label to the data point
            data.loc[list(cluster_set), 'label'] = self.num_of_cluster

            # Mark them as visited.
            data.loc[list(cluster_set), 'visited'] = 1
            self.num_of_cluster += 1
        return data

    # Find all the data point neighbor's index
    def find_N_eps(self, data_x, data_y, data_point):

        # Calculate the distance
        distance = ((data_x - data_point[0]) ** 2 + (data_y - data_point[1]) ** 2) ** 0.5

        # Sort all data points distance accendingly
        distance_rank = sorted(range(len(distance)), key=lambda k: distance[k])

        # Only select the data point which distance is within the threshold
        N_eps = (distance <= self.eps).sum()
        distance_rank = distance_rank[:N_eps]

        # output the list of neighbor data point
        return distance_rank


# Get the data from the txt file
def get_data(path):
    data = pd.read_csv(path, header=None, sep=' ')
    data.columns = ['x', 'y']
    return data


# Entry point for each different run
def run(setting, data_path, minpts, radius, result):
    colors = {-1: 'black', 0: 'green', 1: 'blue', 2: 'yellow', 3: 'red', 4: 'cyan', 5: 'orange', 6: 'pink', 7: 'brown',
              8: 'purple'}
    data = get_data(data_path)
    dbscan = DBSCAN(minpts, radius)
    data = dbscan.fit(data)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'Clustering Result: min_pts={minpts}, radius={radius}\n Left Graph with all data points, Right Graph with outlier only')
    scatter = ax1.scatter(data.x, data.y, c=data['label'].map(colors), s=10, label=colors.keys())
    scatter2 = ax2.scatter(data[data['label'] == -1].x, data[data['label'] == -1].y, c=data[data['label'] == -1]['label'].map(colors), s=10, label=colors.keys())
    plt.xlabel('x')
    plt.ylabel('y')
    fig.set_size_inches(15, 10)
    plt.show()
    result = result.append(
        {'Setting': setting, 'Parameter': f'min_pts={minpts}, 𝜖={radius}', 'No_of_Clusters': data.label.max() + 1,
         'Total no of Outlier': (data.label == -1).sum()}, ignore_index=True)
    return result


if __name__ == '__main__':
    colors = {-1: 'black', 0: 'green', 1: 'blue', 2: 'yellow', 3: 'red', 4: 'cyan', 5: 'orange', 6: 'purple'}
    result = pd.DataFrame({'Setting': [], 'Parameter': [], 'No_of_Clusters': [], 'Total no of Outlier': []})
    result = run(1, 'DBSCAN_Points.txt', 5, 3, result)
    result = run(2, 'DBSCAN_Points.txt', 20, 2.5, result)
    result = run(3, 'DBSCAN_Points.txt', 20, 2, result)
    result = run(4, 'DBSCAN_Points.txt', 15, 2, result)
    # Print all the result
    print(result)
