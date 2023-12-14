"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""

import numpy as np
import time
import random
from functionUtils import AbstractShape
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, area_1):
        self.a = area_1

    def area(self):
        return self.a

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        def helper(x):
            points = contour(x)
            num = 0
            for i in range(len(points)):
                num += 0.5 * ((points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]))
            return num

        n = 250
        points_0 = helper(n)
        points_1 = helper(int(n*1.3))
        while n < 10000:
            if abs(abs(points_1)-abs(points_0)) / abs(points_1) < maxerr:
                return np.float32(abs(points_1))
            n = int(n*1.3)
            points_0 = points_1
            points_1 = helper(n)
        return np.float32(abs(points_1))


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """
        call_find_points = self.find_points(sample)
        call_km = self.km(call_find_points)
        shape = call_km
        a = 0
        for i in range(len(shape) - 1):
            a += (shape[i + 1][0] - shape[i][0]) * (((shape[i + 1][1]) + (shape[i][1])) / 2)
        ans = abs(a)
        res = MyShape(ans)
        return res

    def find_points(self, sample):
        points_list = np.array([sample() for i in range(120000)])
        return points_list

    def km(self, points_list):
        kmeans = KMeans(n_clusters=30, random_state=0).fit(points_list)
        shape_contour = kmeans.cluster_centers_
        shape_contour_arr = np.array(shape_contour)
        ans = self.sort_points(shape_contour_arr)
        return ans

    def sort_points(self, points_arr):
        centroid = np.mean(points_arr, axis=0)
        angles = np.arctan2(points_arr[:, 1] - centroid[1], points_arr[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points_arr[sorted_indices]
        return sorted_points

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
