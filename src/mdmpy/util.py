"""
This module contains utility functions which are called by the main class.
"""
from typing import Callable
from functools import partial
import numpy as np
import numpy.typing as npt
import pyomo.environ as aml
from scipy.optimize import bisect


# in general, lambda = 1
# will be used as default cdf and pdfs
def exp_cdf(x, lambda_: float = 1):
    """Exponential Cumulative Distribution Function (CDF), with default parameter 1"""
    return 1 - aml.exp(-lambda_ * x)


def exp_pdf(x, lambda_: float = 1):
    """Exponential Probability Density Function (PDF), with default parameter 1"""
    return lambda_ * aml.exp(-lambda_ * x)


# in general, beta = 1
def gumbel_cdf(x, beta: float = 1):
    """Gumbel Cumulative Distribution Function (CDF), with default parameter 1"""
    return aml.exp(-aml.exp(-x / beta))


def gumbel_pdf(x, beta: float = 1):
    """Gumbel Probability Density Function (PDF), with default parameter 1"""
    return 1 / beta * aml.exp(-((x / beta) + aml.exp(-(x / beta))))


# Default Bisection function
# Would be a different function if individual-specific coefficients are used
def default_bisect_func(
    input_cdf: Callable[..., float], input_beta, input_x, lambda_: float
) -> float:
    """This is the default bisection function. The last input will be varied
    during the bisection search using partial from functools."""
    return (
        sum(
            1
            - input_cdf(
                lambda_
                - sum(x * y for x, y in zip(input_beta, input_x[k]))  # sum k in K
            )  # dot product
            for k, _ in enumerate(input_x)
        )
        - 1
    )


def find_corresponding_lambda(
    input_cdf: Callable[..., float],
    input_beta: npt.NDArray[np.float64],
    input_x: npt.NDArray[np.float64],
    bisect_func=default_bisect_func,  # can be changed if required
    lamb_const: float = 50000,  # starting lambda guess
    max_lamb_retries: int = 1000,
    lamb_coef: float = 1.4,  # any number >1 should work
) -> float:
    """This function is called to find lambda given the model, input beta
    and input x_i. It does this by first using a large number which may
    be too big and causes overflows, but then reduces the positive and
    negative parts separately until the valid gives a valid output.
    Then, with a positive output and a negative output, a bisection search
    for the root is performed."""
    part_func = partial(bisect_func, input_cdf, input_beta, input_x)
    corr_lamb = None
    lamb_retry = 0
    pos_search_const = lamb_const
    neg_search_const = lamb_const
    while not corr_lamb and lamb_retry <= max_lamb_retries:
        # OverflowError is when the cdf function overflows
        # reduce lambda constant if so
        try:
            part_func(pos_search_const)
        except OverflowError:
            pos_search_const = pos_search_const / lamb_coef
        try:
            part_func(-neg_search_const)
        except OverflowError:
            neg_search_const = neg_search_const / lamb_coef

        # ValueError is when the bisect function has same sign for both
        # positive and negative constants
        # Would likely need to increase lamb_const or decrease lamb_coef
        try:
            corr_lamb = bisect(part_func, -neg_search_const, pos_search_const)
        except ValueError:
            print(pos_search_const)
            print(neg_search_const)
            print(part_func(pos_search_const))
            print(part_func(-neg_search_const))
            break
        except OverflowError:
            pass

        lamb_retry += 1
    return corr_lamb
