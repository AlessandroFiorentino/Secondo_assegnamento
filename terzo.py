
"""
Module: basic Python
Assignment #4 (October 7, 2021)


--- Goal
Create a ProbabilityDensityFunction class that is capable of throwing
preudo-random number with an arbitrary distribution.

(In practice, start with something easy, like a triangular distribution---the
initial debug will be easier if you know exactly what to expect.)


--- Specifications
- the signature of the constructor should be __init__(self, x, y), where
  x and y are two numpy arrays sampling the pdf on a grid of values, that
  you will use to build a spline
- [optional] add more arguments to the constructor to control the creation
  of the spline (e.g., its order)
- the class should be able to evaluate itself on a generic point or array of
  points
- the class should be able to calculate the probability for the random
  variable to be included in a generic interval
- the class should be able to throw random numbers according to the distribution
  that it represents
- [optional] how many random numbers do you have to throw to hit the
  numerical inaccuracy of your generator?

"""

from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.stats import norm
from random import random
import numpy as np

N = 1000 #Global variable for the dimension of an array

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """This ProbabilityDensityFunction derived class from
    InterpolatedUnivariateSpline(x, y, w=None, bbox=[None, None], k=3, ext=0, check_finite=False)
    class describes a probability density function.
    """
    def __init__(self, x, y, a = 3):
        """This is the constructor of the class, x and y are two numpy arrays sampling
        the pdf on a grid of values and a is the degree of the polinomial"""
        f = InterpolatedUnivariateSpline(x, y, k = a)
        self._F = f.antiderivative() # Mi tengo da parte la cumulata
        self._xmin = np.min(x)
        self._xmax = np.max(x)
        self._F_max = self._F(self._xmax)
        InterpolatedUnivariateSpline.__init__(self, self._F(x)/self._F_max, x, k = a)

    def probIn(self, min, max):
        """ Method calculating the probability in a given interval
        """

        if min < self._xmin or max > self._xmax:
            raise Exception(f"Bounds are out of range [{self._xmin}, {self._xmax}]")
        return (self._F(max) - self._F(min))/self._F_max

    def random(self, size = N):
        """Return an array of random values from the pdf.
        """
        q = np.random.uniform(low=0., high=1., size=(size,))
        return self(q)


if __name__ == '__main__':
    x = np.linspace(0., 1., 30)
    y = np.linspace(0., 3., 30)
    #Gaussian
    #x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    #y = norm.pdf(x)
    g = ProbabilityDensityFunction(x, y)
    print(f"the probability is: {g. probIn(0,0.5)}")

    plt.plot(range(N), g.random(N), '.')

    #plt.plot(x, y, 'o')
    #plt.plot(x, f(x))
    plt.show()