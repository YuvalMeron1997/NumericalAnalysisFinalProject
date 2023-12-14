"""
In this assignment you should find the intersection points for two functions.
"""
import numpy as np
import time
import random
from collections.abc import Iterable
import math


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        intersections_list = []
        call_intersect_func = self.intersect_func(f1, f2)
        numbers_between_a_b = np.linspace(a, b, 1000)
        num = 0
        while num < len(numbers_between_a_b) - 1:
            if numbers_between_a_b[num] != None:
                check = self.check_func(numbers_between_a_b[num], numbers_between_a_b[num + 1], call_intersect_func,
                                        maxerr,intersections_list)
            if check != None:
                self.newton_raphson_func(check, numbers_between_a_b[num], numbers_between_a_b[num + 1],
                                         call_intersect_func, maxerr, intersections_list)
                num += 1
            else:
                num += 1
        return intersections_list

    def intersect_func(self, f1: callable, f2: callable):
        return lambda x: f1(x) - f2(x)

    def check_func(self, left, right, call_intersect_func, maxerr,list):
        if call_intersect_func(left) == 0:
            list.append(left)
            return
        if call_intersect_func(right) == 0:
            list.append(right)
            return
        if (call_intersect_func(left) * call_intersect_func(right)) > 0:
            return
        else:
            bisection = self.bisection_func(left, right, call_intersect_func, maxerr)
            return bisection

    def bisection_func(self, left, right, call_intersect_func, maxerr):
        ans = left
        maximum = (left + right) / 2
        for i in range(15):
            ans = (left + right) / 2
            if ans > maximum:
                maximum = ans
            if call_intersect_func(ans) == 0.0 or abs(call_intersect_func(ans)) < maxerr:
                return ans
            if call_intersect_func(left) * call_intersect_func(ans) > 0:
                left = ans
            else:
                right = ans
        return


    def regula_falsi_func(self, f, left, right, maxerr=0.001):
        while abs(right - left) > maxerr:
            ans = (left * f(right) - right * f(left)) / (f(right) - f(left))
            if f(left) * f(right) < 0:
                right = ans
            else:
                left = ans
        return ans

    def secant_func(self, f, left, right, maxerr = 0.001):
        while abs(right - left) > maxerr:
            x = right - f(right) * (right - left) / (f(right) - f(left))
            left = right
            right = x
        return right

    def newton_raphson_func(self, root, left, right, call_intersect_func, maxerr, list):
        # if call_intersect_func(left) <= maxerr:
        #     list.append(left)
        #     return
        # if call_intersect_func(right) <= maxerr:
        #     list.append(right)
        #     return
        r = root
        small_num = 0.000000000001
        derive = (call_intersect_func(r + small_num) / small_num) - (call_intersect_func(r) / small_num)
        for i in range(15):
            e = call_intersect_func(r) / derive
            r = r - e
            derive = (call_intersect_func(r + small_num) / small_num) - (call_intersect_func(r) / small_num)
        if r in list:
            return
        else:
            list.append(r)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    # def test1(self):
    #     f1 = lambda x: 5
    #     f2 = lambda x: x**2 - 3*x + 5
    #     f3 = lambda x: math.sin(x**2)
    #     f4 = lambda x: math.e**(-2*(x**2))
    #     f5 = lambda x: np.arctan(x)
    #     f6 = lambda x: (np.sin(x))/x
    #     f7 = lambda x: 1 / (np.log(x))
    #     f8 = lambda x: math.e**(math.e**(x))
    #     f9 = lambda x: np.log(np.log(x))
    #     # f10 = lambda x: math.sin(math.log(x))
    #     f11 = lambda x: (2**(1/(x**2))) * (math.sin(1/x))
    #     f12 = lambda x: x**7
    #     f13 = lambda x: 2*(x**9)
    #     f14 =lambda x: 5.5*x**10 - 0.1*x**9 + 1.7*x**8 - 8.15*x**7 - 0.65*x**6 + 0.4*x**5 + 1.2*x**4 + 0.75*x**3 - 2.90*x**2 + 0.34*x - 3.88
    #     f15 = lambda x: math.sin(100*x)
    #     f16 = lambda x: math.cos(100*x)
    #
    #     a = -20
    #     b = 20
    #
    #     intersector = Assignment2()
    #     intersector.intersections(f1, f5, a, b, maxerr=0.001)
    #     # intersector.intersections(f2, f8, a, b, maxerr=0.001)
    #     intersector.intersections(f4, f9, a, b, maxerr=0.001)
    #     intersector.intersections(f6, f7, a, b, maxerr=0.001)
    #     intersector.intersections(f5, f9, a, b, maxerr=0.001)
    #     # intersector.intersections(f7, f11, a, b, maxerr=0.001)


if __name__ == "__main__":
    unittest.main()
