
"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        segments_between_a_b = np.linspace(a, b, n) #segmets
        points = np.empty([n, 2]) #matrix: [0. 0.] 2 columns, n rows
        i = 0
        while i < n:
            x_val = segments_between_a_b[i]
            points[i] = [x_val, f(x_val)] #array of n arrays of points[x,y]
            i += 1
        A, B = self.build_vec_matrix(points)
        d = {}
        for i in range(len(points) - 1):
            d[(points[i][0], points[i][1])] = self.get_cubic(points[i], A[i], B[i], points[i + 1])
        dict_ans = d
        result = lambda x: self.find_x(x, dict_ans)
        return result

    def build_vec_matrix(self, points):
        n = len(points) - 1
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2
        down = list(np.diag(C, k=-1))
        middle = list(np.diag(C))
        up = list(np.diag(C, k=1))
        vec = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        vec[0] = points[0] + 2 * points[1]
        vec[n - 1] = 8 * points[n - 1] + points[n]
        return self.thomas(down,middle,up,vec, points)



    def thomas(self, down, middle, up, vec, points):
        n = len(middle)
        A = [0] * n
        for i in range(1,n):
            divide = down[i-1] / middle[i-1]
            middle[i] = middle[i] - up[i-1] * divide
            vec[i] = vec[i] - vec[i - 1] * divide
        A[n-1] = vec[n-1] / middle[n-1]
        for i in range(n - 2, -1, -1):
            A[i] = (vec[i] - up[i] * A[i+1]) / middle[i]
        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1][1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n][1]) / 2
        return A, B


    def get_cubic(self, a, b, c, d):
        return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


    def find_x(self, x, dict_ans):
        s = (0,0)
        e = (0,0)
        for key in dict_ans.keys():
            if x > key[0]:
                s = key
            else:
                e = key
                break
        res = dict_ans[s]((x-s[0]) / (e[0]-s[0]))[1]
        return res



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
