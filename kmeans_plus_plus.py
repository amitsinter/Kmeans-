import os
import pickle
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# function to plot the selected centroids
from sympy import Integer

# Authors: Idan Abergel, Amit Sinter

ENABLE_PLOT = False


def plot(data, centroids):
    if not ENABLE_PLOT:
        return

    plt.scatter(data[:, 0], data[:, 1], marker='.',
                color='gray', label='data points')
    plt.scatter(centroids[-1, 0], centroids[-1, 1],
                color='red', label='new centroid')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1],
                color='black', label='old centroids')
    plt.title('Initialize centroid #%d ' % (centroids.shape[0]))

    plt.legend()
    plt.show()


def plot_iteration_by_cluster(data, point_cluster, i, centroids):
    plt.scatter(data[:, 0], data[:, 1], c=point_cluster, marker='.',
                label='data points')
    plt.scatter([centroids[:, 0]], [centroids[:, 1]],
                color='red', label='centroid')
    plt.title("Iteration #" + str(i))

    plt.legend()
    plt.show()


# Centroids Initialization for K-means++
def initialize(data, k):
    """
    Centroids Initialization -  K-means++
    inputs:
        data - numpy array
        k - number of clusters
    """

    # Step 1) Randomly initialize the first centroids
    centroids = [data[np.random.randint(data.shape[0]), :]]
    # plot(data, np.array(centroids))

    # Step 2) Compute the k - 1 remaining clusters
    for c in range(k - 1):

        # distances lists contains the minimum distance between each point
        # to the closer centroid

        distances = []
        # for each point
        for i in range(data.shape[0]):
            point = data[i, :]

            # In order to check how point closed to the centroids,
            # get the minimum distance of the point from all centroids
            distances.append(min(DistanceMulti(point, centroids)))

        # select data point with maximum distance as our next centroid
        next_centroid = data[np.argmax(np.array(distances)), :]
        centroids.append(next_centroid)
        plot(data, np.array(centroids))

    return centroids


def cluster_points_to_cluster(data, centroids):
    k = len(centroids)
    # Calculate the distance between each point to centroids i
    # and determine point cluster
    point_cluster = np.zeros([data.shape[0], 1])
    for idx, point in enumerate(data):
        point_cluster[idx] = np.argmin(DistanceMulti(point, centroids))

    return point_cluster


def calculate_new_centroid(data, point_cluster, centroids):
    for c_idx in range(len(centroids)):
        point_idx = np.argwhere(point_cluster == c_idx)[:, 0]
        centroids[c_idx] = np.mean(data[point_idx, :], axis=0)
    return centroids


def create_data():
    # creating 2D data
    mean_01 = np.array([0.0, 0.0])
    cov_01 = np.array([[1, 0.3], [0.3, 1]])
    dist_01 = np.random.multivariate_normal(mean_01, cov_01, 100)

    mean_02 = np.array([6.0, 7.0])
    cov_02 = np.array([[1.5, 0.3], [0.3, 1]])
    dist_02 = np.random.multivariate_normal(mean_02, cov_02, 100)

    mean_03 = np.array([7.0, -5.0])
    cov_03 = np.array([[1.2, 0.5], [0.5, 1, 3]])
    dist_03 = np.random.multivariate_normal(mean_03, cov_01, 100)

    mean_04 = np.array([2.0, -7.0])
    cov_04 = np.array([[1.2, 0.5], [0.5, 1, 3]])
    dist_04 = np.random.multivariate_normal(mean_04, cov_01, 100)

    data = np.vstack((dist_01, dist_02, dist_03, dist_04))
    np.random.shuffle(data)

    return data


def DistanceMulti(p1, p2):
    return np.linalg.norm(p1 - p2, axis=1)


def Distance(p1, p2):
    # return np.linalg.norm(p1 - p2)
    return np.sqrt(np.dot(p1, p1) - 2 * np.dot(p1, p2) + np.dot(p2, p2))


def Silhouette(K, data, point_cluster, centroids):
    point_silhouette = np.zeros(len(data))

    for idx, p in enumerate(data):
        cluster = point_cluster[idx]
        j_points_idx = np.argwhere(point_cluster == cluster)[:, 0]

        if len(j_points_idx) == 1:
            point_silhouette[idx] = 0
            continue

        sum = 0
        for j in j_points_idx:
            if idx == j:
                continue
            sum += Distance(p, data[j])

        a = sum / (len(j_points_idx) - 1)

        sum = 0
        b = float('inf')
        for k in range(K):
            if point_cluster[idx] == k:
                continue
            j_points_idx = np.argwhere(point_cluster == k)[:, 0]
            for j in j_points_idx:
                sum += Distance(p, data[j])

            b_temp = sum / len(j_points_idx)
            b = min(b, b_temp)

        if a < b:
            point_silhouette[idx] = 1 - a / b
        elif a == b:
            point_silhouette[idx] = 0
        elif a > b:
            point_silhouette[idx] = b / a - 1

    return point_silhouette


def kmeans_pp(data, k, max_iteration):
    centroids = initialize(data, k)
    centroids = np.matrix(centroids)
    prev_centroid = []
    for i in range(max_iteration):
        # print("Old:")
        prev_centroid = centroids.copy()
        # print(centroids)
        # print("New:")
        point_cluster = cluster_points_to_cluster(data, centroids)
        centroids = calculate_new_centroid(data, point_cluster, centroids)
        # print(centroids)
        # plot_iteration_by_cluster(data, point_cluster, i, centroids)

        centroids_diff = centroids - prev_centroid
        if not centroids_diff.any():
            print("Converged at #" + str(i))
            break
    return centroids, point_cluster


def main():
    plt.close('all')

    if os.path.isfile('DanData.save'):
        with open('DanData.save', 'rb') as file:
            data = pickle.load(file)
    else:
        data = np.loadtxt(fname="exampleData.txt", delimiter=",")
        with open("DanData.save", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    data = data[1:2000, :]
    print("Start")
    # data = data[:,:]
    max_iteration = 20

    # Define k range
    k_range = range(3, 15)
    k_scores = np.zeros(len(k_range))

    for k in k_range:
        print("-----------")
        print("K = " + str(k))
        start_time = time.time()

        centroids, point_cluster = kmeans_pp(data, k, max_iteration)

        elapsed_time = time.time() - start_time
        print(str(elapsed_time) + " [s]")

        point_silhouette = Silhouette(k, data, point_cluster, centroids)
        k_scores[k - k_range[0]] = np.average(point_silhouette)
        print("k = " + str(k) + " | Score = " + str(k_scores[k - k_range[0]]))

    plt.bar(list(k_range), k_scores)
    plt.title("Silhouette Avg Score by number of clusters - k")
    plt.show()

    optimal_k = k_range[np.maxarg(k_scores)]
    centroids, point_cluster = kmeans_pp(data, optimal_k, max_iteration)


if __name__ == "__main__":
    main()
