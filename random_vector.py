import numpy as np


class Random_Vector:
    """
    Random vector class.
    """
    def __init__(self, mean, covariance, length):
        """
        Constructor for the Random_Vector class.
        :param mean: Numpy object of the mean row vector.
        :param covariance: Numpy object of the covariance matrix.
        :param length: The lenght of the Random Vector. This length is not to be changed after initialization.
        """
        Random_Vector.check_mean_conditions(mean, length)
        Random_Vector.check_covariance_conditions(covariance, length)
        self.mean_ = mean
        self.covariance_ = covariance
        self.length = length

    @property
    def mean(self):
        Random_Vector.check_mean_conditions(self.mean_, self.length)
        return self.mean_

    @mean.setter
    def mean(self, new_mean):
        Random_Vector.check_mean_conditions(new_mean, self.length)
        self.mean_ = new_mean

    @property
    def covariance(self):
        Random_Vector.check_covariance_conditions(self.covariance_, self.length)
        return self.covariance_

    @covariance.setter
    def covariance(self, new_covariance):
        Random_Vector.check_covariance_conditions(new_covariance, self.length)
        self.covariance_ = new_covariance

    @staticmethod
    def check_mean_conditions(mean, n):
        """
        The mean needs to follow:
            1. It needs to be an numpy array
            2. It needs to be 2-dimensional column vector.
            3. It's size of the row needs to be consistent with the specified instance length.
        :param covariance: The mean row vector to check.
        :param n: the length of the Random Vector.
        :return: nothing
        """
        if type(mean) is not np.ndarray:
            raise TypeError("Expected mean of to be an numpy array.")
        if mean.ndim < 2:
            raise TypeError("Expected mean to be 2-dimensions.")
        if mean.shape[0] != n:
            raise ValueError("Size of mean vector is not consistent with length.")

    @staticmethod
    def check_covariance_conditions(covariance, n):
        """
        The covariance needs to follow:
            1. The covariance matrix is an Numpy object.
            2. The covariance matrix is 2-dimensional.
            3. The covariance matrix is square.
            4. The covariance matrix needs to have the size (n,n)
        :param covariance: The covariance matrix to check.
        :param n: the length of the Random Vector.
        :return:
        """
        if type(covariance) is not np.ndarray:
            raise TypeError("Expected covariance of to be an numpy array.")
        if covariance.ndim < 2:
            raise TypeError("Expected covariance to be 2-dimensions.")
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("The covariance matrix needs to be square.")
        if covariance.shape[0] != n:
            raise ValueError("Size of covariance vector is not consistent with length.")



