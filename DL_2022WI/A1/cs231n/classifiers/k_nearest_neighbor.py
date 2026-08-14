from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #trick:先计算好返回值的shape并初始化相应ndarray
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                diff=self.X_train[j]-X[i]
                dists[i,j]=np.sqrt(np.sum(diff**2))
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            diff=self.X_train-X[i]   #shape=(num_train,D)
            dists[i]=np.sqrt(np.sum(diff**2,axis=1))
            """
            self.X_train.shape=(num_train,D);
            X[i].shape=D
            可以使用广播机制,左边补充1,再扩充为num_train
            (或许实践中样本数作为'行'的原因就是为了方便广播)
            """
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        """标准实现:利用范数性质,|x-y|^2=|x|^2+|y|^2-2<x,y>"""
        """
        C=A@B.T C[i,j]=A[i]@B[j]
        dists[i,j]=|X[i]|**2+|self.X_train[j]|**2-2*X[i]@self.X_train[j]
        =>cross=X@self.X_train.T
        还需两个矩阵,M1[i,j]=|X[i]|**2,M2[i,j]=|self.X_train[j]|**2
        从而dists**2=M1+M2-2*cross , shape=(num_test,num_train)
        M1只与行有关,M2只与列有关,因此只需令M1.shape=(num_test,1)
        M2.shape=(num_train,)  即可利用广播自动扩充shape  
        """
        X_train_sq=(self.X_train**2).sum(axis=1)  #(num_train,)
        X_sq=(X**2).sum(axis=1,keepdims=True)   #(num_test,1)
        cross=X@self.X_train.T
        dists=np.sqrt(X_train_sq+X_sq-2*cross)
        """
        实现1:功能正确,但中间量diff.size太大,colab的RAM不够用
        X_reshaped=X.reshape(num_test,1,X.shape[1])
        #(num_test,1,D)   self.X_train (num_train,D)
        diff=self.X_train-X_reshaped  #shape=(num_test,num_train,D)
        dists=np.sqrt(np.sum(diff**2,axis=2))
        """
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            """notes:
                1.argsort(arr)返回排序后的与arr同形状的array,元素为索引     
            """
            closest_indices=np.argsort(dists[i])[:k]
            closest_y=self.y_train[closest_indices]

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            """notes:
                1.np.unique(arr) 返回包含arr中出现的值的ndarry,同一个值只计入一次
                np.unique(arr,return_counts=True) 在原本的基础上再返回各个值的出现次数,
                与唯一值数组组成一个元组再返回。
                可通过unique_values,counts=np.unique(arr,return_counts=True)来接收
                np.unique默认会从小到大排序unique_values
            """
            unique_labels,counts=np.unique(closest_y,return_counts=True)
            max_count=np.max(counts)
            y_pred[i] = unique_labels[counts == max_count][0]
            """
            与y_pred[i]=np.min(unique_labels[counts==max_count]) 
            是等价的,因为np.unique默认从小到大排序
            """


        return y_pred
