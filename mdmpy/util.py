from functools import partial
import pyomo.environ as aml
from scipy.optimize import bisect

# in general, lambda = 1
# will be used as default cdf and pdfs
def exp_cdf(x, lambda_ = 1):
    return 1-aml.exp(-lambda_*x)
def exp_pdf(x, lambda_ = 1):
    return lambda_*aml.exp(-lambda_*x)

# Default Bisection function
# Would be a different function if individual-specific coefficients are used
def default_bisect_func(input_cdf, input_beta, input_x, lambda_):
    """This is the default bisection function. The last input will be varied
    during the bisection search using partial from functools."""
    return (sum(1-input_cdf(lambda_-                            # sum k in K
            sum(x*y for x,y in zip(input_beta,input_x[k]))      # dot product
                        ) for k,_ in enumerate(input_x))-1)

def find_corresponding_lambda(input_cdf, input_x, input_beta,
                              bisect_func = default_bisect_func, # if required to be changed
                              lamb_const:float = 50000, # starting lambda guess
                              max_lamb_retries:int = 1000,
                              lamb_coef:float = 1.4 # any number >1 should work
                              ) -> float:
    part_func  = partial(bisect_func, input_cdf, input_beta, input_x)
    cor_lamb   = None
    lamb_retry = 0
    pos_search_const = lamb_const
    neg_search_const = lamb_const
    while not cor_lamb and lamb_retry <= max_lamb_retries:
        # OverflowError is when the cdf function overflows
        # reduce lambda constant if so
        try:
            part_func(pos_search_const)
        except OverflowError:
            pos_search_const = pos_search_const/lamb_coef
        try:
            part_func(-neg_search_const)
        except OverflowError:
            neg_search_const = neg_search_const/lamb_coef

        # ValueError is when the bisect function has same sign for both
        # positive and negative constants
        # Would likely need to increase lamb_const or decrease lamb_coef
        try:
            cor_lamb = bisect(part_func,
                              -neg_search_const,
                              pos_search_const)
        except ValueError:
            print(pos_search_const)
            print(neg_search_const)
            print(part_func(pos_search_const))
            print(part_func(-neg_search_const))
            break
        except OverflowError:
            pass

        lamb_retry += 1
    return cor_lamb
