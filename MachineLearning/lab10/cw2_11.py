import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# CW2 11
def kmeans(X, k, num_init):
    """
        Compute the score and the means of the clusters obtained by k-means with random initializations.
        With KMeans function,
            set its number of clusters, number of initialization to the respective given arguments
            set the initialization method to random
            set the k-means algorithm to full
            set the random state to 5
        Args:
            X - array of the sample points
            k - the number of clusters
            num_init - the number of random initialization
        Returns:
            score - the objective function / the mean squared distance between each sample point and its mean
            means - array of the cluster means
    """
    # Write your code here
    kmeans = KMeans(n_clusters=k, n_init=num_init,init='random',algorithm='full',random_state=5)
    kmeans.fit(X)
    return kmeans.inertia_, kmeans.cluster_centers_

    


def kmeans_on_blobs() :
    blob_centers = np.array([[0.2, 2.3],
                             [-1.5, 2.3],
                             [-2.8, 1.8],
                             [-2.8, 2.8],
                             [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers,
                      cluster_std=blob_std, random_state=5)
    k = 5
    num_init = 20
    # score = kmeans(X,k,num_init)
    score, means = kmeans(X, k, num_init)
    print(round(score, 10))
    print(means[means[:,0].argsort()])


def main():
    kmeans_on_blobs()


if __name__ == "__main__":
    main()