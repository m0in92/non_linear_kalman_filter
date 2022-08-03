import unittest
import numpy as np
from random_vector import Random_Vector


class TestRV(unittest.TestCase):

    def test_vector_init(self):
        init_mean = np.array([0,0,1]).reshape(-1,1)
        init_covariance = np.array([[0,0,0],[0,0,0],[0,0,0]])
        n = 3
        rv = Random_Vector(init_mean, init_covariance, n)

        # Check for initial mean and covariance
        np.testing.assert_allclose(rv.mean, init_mean)

    def test_change_vector_variables(self):
        init_mean = np.array([0, 0, 1]).reshape(-1, 1)
        init_covariance = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        n = 3
        rv = Random_Vector(init_mean, init_covariance, n)
        new_mean = np.array([1, 2, 1]).reshape(-1, 1)
        rv.mean = new_mean
        new_covariance = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        rv.covariance = new_covariance

        #check for new mean and covariance
        np.testing.assert_allclose(rv.mean, new_mean)
        np.testing.assert_allclose(rv.covariance, new_covariance)

    def test_incorrect_mean_init(self):
        init_mean = 0
        init_covariance = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        n = 3

        # test for non Numpy object as mean
        with self.assertRaises(TypeError):
            rv = Random_Vector(init_mean, init_covariance, n)

        # test for mean with 1-D input
        init_mean = np.array([2,1,1])
        with self.assertRaises(TypeError):
            rv = Random_Vector(init_mean, init_covariance, n)

        # test for mean with inconsistent n
        init_mean = np.array([[2], [1]])
        with self.assertRaises(ValueError):
            rv = Random_Vector(init_mean, init_covariance, n)

    def test_incorrect_mean_new(self):
        init_mean = np.array([0, 0, 1]).reshape(-1, 1)
        init_covariance = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        n=3
        rv = Random_Vector(init_mean, init_covariance, n)

        # test for non Numpy object as mean
        new_mean = 0
        with self.assertRaises(TypeError):
            rv.mean = new_mean

        # test for mean with 1-D input
        new_mean = np.array([2,1,1])
        with self.assertRaises(TypeError):
            rv.mean = new_mean

        # test for mean with inconsistent n
        new_mean = np.array([[2], [1]])
        with self.assertRaises(ValueError):
            rv.mean = new_mean

    def test_incorrect_cov_init(self):
        init_mean = np.array([0, 0, 1]).reshape(-1, 1)
        init_covariance = 0
        n = 3

        # test for non Numpy object as covariance
        with self.assertRaises(TypeError):
            rv = Random_Vector(init_mean, init_covariance, n)

        # test for covariance with inconsistent n
        init_covariance = np.array([[2,1,1],[1,1,1]])
        with self.assertRaises(ValueError):
            rv = Random_Vector(init_mean, init_covariance, n)

        # test for covariance with non-square matrix
        init_covariance = np.array([[2,1], [1,1], [1,1]])
        with self.assertRaises(ValueError):
            rv = Random_Vector(init_mean, init_covariance, n)

    def test_incorrect_cov_new(self):
        init_mean = np.array([0, 0, 1]).reshape(-1, 1)
        init_covariance = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        n=3
        rv = Random_Vector(init_mean, init_covariance, n)

        # test for non Numpy object as mean
        new_covariance = 0
        with self.assertRaises(TypeError):
            rv.covariance = new_covariance

        # test for mean with inconsistent n
        new_covariance = np.array([[2,1,1],[1,1,1]])
        with self.assertRaises(ValueError):
            rv.covariance = new_covariance

        # test for covariance with non-square matrix
        new_covariance = np.array([[2,1], [1,1], [1,1]])
        with self.assertRaises(ValueError):
            rv.covariance = new_covariance


if __name__ == '__main__':
    unittest.main()