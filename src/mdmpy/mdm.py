"""
This module contains the main class, MDM, in this package.

Within the class are methods related to solving for the Maximum
Likelihood Estimate (MLE) of coefficients. Two methods are used
to initialise a model using pyomo, which can then be passed onto
a solver to be solved. Another way to solve for the coefficients
is to use a gradient ascent.
"""
from typing import Callable
import math
from pandas import DataFrame
import numpy as np
import numpy.typing as npt
import pyomo.environ as aml
from . import util


class MDM:
    """
    The main class of the package. This models the Marginal Distribution
    Models (MDM)
    """

    def __init__(
        self,
        input_dataframe: DataFrame,
        ch_id: int,
        num_choices: int,
        list_of_coefs: npt.NDArray[np.float64],
        input_cdf: Callable[..., float] = util.exp_cdf,
        input_pdf: Callable[..., float] = util.exp_pdf,
    ):
        """ "
        The class is initialised with a 2D NumPy array, which acts as a
        table in the 'wide' format in Discrete Choice Modelling.
        One set of inputs are relevant column indices, such as the choice
        index, and the coefficients that will be considered in the model
        Another input is the number of choices or alternatives each
        individual has. The data is then made into other NumPy arrays used
        in the class methods.

        Default CDF/PDF of the model is the Multinomial Logit (MNL) model,
        which is globally convex in its support. Other CDFs and PDFs will
        be supported

        This will require some other changes to allow for individual-specific
        coefficients, which will be added at a later date.
        """

        num_indiv = int(input_dataframe.shape[0] / num_choices)
        # check if numerically equivalent
        if num_indiv != (input_dataframe.shape[0] / num_choices):
            raise ValueError(
                """Unexpectedly, the number of columns does not divide
                                by the number of choices, as inputed."""
            )
        num_attr = len(list_of_coefs)
        Z = np.array(input_dataframe.iloc[:, ch_id]).reshape((num_indiv, num_choices))
        X = np.reshape(
            np.array(input_dataframe.iloc[:, list_of_coefs]),
            (num_indiv, num_choices, num_attr),
        )
        self._X = X  # is indexed in the order (indiv, choice, attr)
        self._Z = Z  # choice of each individual
        self._num_indiv = num_indiv  # number of individuals
        self._num_attr = num_attr  # number of attributes/coefficients
        self._num_choices = num_choices  # number of alternatives/choices
        self._cdf = input_cdf  # sets the cdf for the model
        self._vcdf = np.vectorize(input_cdf)
        self._pdf = (
            input_pdf  # sets corresponding pdf (has to be inputted, not automatic)
        )
        self._vpdf = np.vectorize(input_pdf)

    def ll(self, input_beta: npt.NDArray[np.float64], corr_lambs=None) -> float:
        """This function gets the log-likelihood using the current beta. If
        the corresponding lambdas for each individual are given, then it will
        use those, rather than re-computing them, which saves computations"""
        loglik = 0
        for i in range(self._num_indiv):
            x_i = self._X[i]
            if corr_lambs is None:
                cor_lamb = util.find_corresponding_lambda(self._cdf, input_beta, x_i)
            else:
                cor_lamb = corr_lambs[i]
            for k, choice in enumerate(self._Z[i]):
                if choice:
                    x_ik = x_i[k]
                    f_arg_ik = cor_lamb - sum(x * y for x, y in zip(input_beta, x_ik))
                    loglik += math.log(1 - self._cdf(f_arg_ik))
        return loglik

    ### TODO - refactor the double summation over I and K, which is repeated
    def __lag_f(self, arg):
        return aml.log(1 - self._cdf(arg))

    def __loglikexpr(self, heteroscedastic=False, asc=False, lag_f=__lag_f):
        ### TODO Figure out best way to write out the different cases
        if heteroscedastic:
            if asc:
                return sum(
                    sum(
                        self._Z[i][k]
                        * self.m.alpha[k]
                        * lag_f(
                            self.m.lambda_[i]
                            - sum(self.m.beta[l] * self._X[i][k][l] for l in self.m.L)
                            - self.m.ASC[k]
                        )
                        for k in self.m.K
                    )
                    for i in self.m.I
                )
            else:
                return sum(
                    sum(
                        self._Z[i][k]
                        * self.m.alpha[k]
                        * lag_f(
                            self.m.lambda_[i]
                            - sum(self.m.beta[l] * self._X[i][k][l] for l in self.m.L)
                        )
                        for k in self.m.K
                    )
                    for i in self.m.I
                )
        else:
            if asc:
                return sum(
                    sum(
                        self._Z[i][k]
                        * lag_f(
                            self.m.lambda_[i]
                            - sum(self.m.beta[l] * self._X[i][k][l] for l in self.m.L)
                            - self.m.ASC[k]
                        )
                        for k in self.m.K
                    )
                    for i in self.m.I
                )
            else:
                return sum(
                    sum(
                        self._Z[i][k]
                        * lag_f(
                            self.m.lambda_[i]
                            - sum(self.m.beta[l] * self._X[i][k][l] for l in self.m.L)
                        )
                        for k in self.m.K
                    )
                    for i in self.m.I
                )

    def model_init(
        self,
        heteroscedastic=False,
        alpha_to_be_fixed=0,
        alpha_cons_limits: tuple[float, float] = (0, 1),
        usc_ascs=False,
        asc_fixed_to_zero=None,
        model_seed: int | None = None,
        v=0,
    ):
        """
        This method initializes the pyomo model as an instance
        attribute m. Values can later be taken directly from m.

        The various keyword arguments of this method are used to customise
        the resulting model. They include alternative-specific constants
        (ASCs) which act as y-intercepts depending on which choice was made.

        Another keyword argument is involved in deciding whether the model
        will handle heteroscedasticity.
        """
        self.m = aml.ConcreteModel()
        # Model Sets
        self.m.K = aml.Set(initialize=range(self._num_choices))
        self.m.L = aml.Set(initialize=range(self._num_attr))
        self.m.I = aml.Set(initialize=range(self._num_indiv))

        ### Model Variables ###
        # Initialize at some certain seed
        # For checking if convexity gives numerical stability
        if model_seed:
            np.random.seed(model_seed)
            np_stan_exp = np.random.standard_exponential
            self.m.beta = aml.Var(
                self.m.L, initialize=lambda _: np_stan_exp()
            )  # 1 arg required for initialize
            self.m.lambda_ = aml.Var(self.m.I, initialize=lambda _: np_stan_exp())
            if usc_ascs:
                self.m.ASC = aml.Var(self.m.K, initialize=lambda _: np_stan_exp())
            if v >= 1:
                print([self.m.beta[q].value for q in self.m.L])
                print([self.m.lambda_[q].value for q in self.m.I])
                if usc_ascs:
                    print([self.m.ASC[q].value for q in self.m.K])
        else:
            self.m.beta = aml.Var(self.m.L)
            self.m.lambda_ = aml.Var(self.m.I)
            if usc_ascs:
                self.m.ASC = aml.Var(self.m.K)
                if asc_fixed_to_zero:
                    self.m.ASCConstr = aml.Constraint(
                        expr=self.m.ASC[asc_fixed_to_zero] == 0
                    )

        # Variable for handling heteroscedascity
        if heteroscedastic:
            # known heteroscedasticity
            if isinstance(heteroscedastic, list):
                self.m.alpha = {idx: v for idx, v in enumerate(heteroscedastic)}
            # else, unknown heteroscedasticity
            else:
                self.m.alpha = aml.Var(self.m.K, domain=aml.NonNegativeReals)
                self.m.FixOneAlphaC = aml.Constraint(
                    expr=self.m.alpha[alpha_to_be_fixed] == 1
                )

                def _alpha_cons(model, k):
                    # alpha_cons_limits is a keyword argument to model_init
                    return (alpha_cons_limits[0], model.alpha[k], alpha_cons_limits[1])

                self.m.AlphaTol = aml.Constraint(self.m.K, rule=_alpha_cons)

        # Objective Function
        ### TODO Hardcode in the other common distributions, especially
        ### Those with a region which is simultaneously reliability function
        ### convex and logconcave, which guarantees concave objective
        ### This hardcoding should simplify the algebraic expression
        ### for the solver, and hence allow to see more
        ###### Dists under consideration:
        ###### Hyperbolic secant - region: non-negative numbers AKA above median
        ###### Extreme Value - region: non-negative numbers AKA above median
        ###### ... distributions with suitable unbounded regions
        ###### ... satisfying tail convex+tail logconcave
        if heteroscedastic:
            if self._cdf == util.exp_cdf and usc_ascs:
                obj = self.__loglikexpr(
                    heteroscedastic=True, asc=True, lag_f=lambda arg: -arg
                )
            elif self._cdf == util.exp_cdf:
                obj = self.__loglikexpr(heteroscedastic=True, lag_f=lambda arg: -arg)
            else:
                obj = self.__loglikexpr(heteroscedastic=True)

        else:
            # Model CDF simplifications
            if self._cdf == util.exp_cdf:
                obj = self.__loglikexpr(lag_f=lambda arg: -arg)
            elif self._cdf == util.gumbel_cdf:
                obj = self.__loglikexpr(
                    lag_f=lambda arg: aml.log(1 - aml.exp(-aml.exp(-arg)))
                )
            elif usc_ascs:
                obj = self.__loglikexpr(asc=True)
            else:
                obj = self.__loglikexpr()

        # Model Objective
        self.m.O = aml.Objective(expr=obj, sense=aml.maximize)

        # Lagrangian Constraints (for each individual)
        def _lag_cons(model, i):
            # MEM
            if heteroscedastic and usc_ascs and self._cdf == util.exp_cdf:
                lhs_sum_expr = sum(
                    aml.exp(
                        model.alpha[k]
                        * (
                            sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                            + model.ASC[k]
                            - model.lambda_[i]
                        )
                    )
                    for k in model.K
                )

            ### TODO Figure out best way to write out the different cases
            elif heteroscedastic and self._cdf == util.exp_cdf:
                lhs_sum_expr = sum(
                    aml.exp(
                        model.alpha[k]
                        * (
                            sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                            - model.lambda_[i]
                        )
                    )
                    for k in model.K
                )
            elif self._cdf == util.exp_cdf:
                lhs_sum_expr = sum(
                    aml.exp(
                        sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                        - model.lambda_[i]
                    )
                    for k in model.K
                )
            elif self._cdf == util.gumbel_cdf:
                lhs_sum_expr = sum(
                    1
                    - aml.exp(
                        -aml.exp(
                            sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                            - (model.lambda_[i])
                        )
                    )
                    for k in model.K
                )
            elif usc_ascs:
                lhs_sum_expr = sum(
                    (
                        1
                        - self._cdf(
                            model.lambda_[i]
                            - sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                            - model.ASC[k]
                        )
                    )
                    for k in model.K
                )
            # Default
            else:
                lhs_sum_expr = sum(
                    (
                        1
                        - self._cdf(
                            model.lambda_[i]
                            - sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                        )
                    )
                    for k in model.K
                )
            return lhs_sum_expr <= 1

        self.m.C = aml.Constraint(self.m.I, rule=_lag_cons)

        # Scale restriction - not required
        # but might help solver not get lost and diverge
        def _beta_size_cons(model, l):
            return (-20, model.beta[l], 20)

        self.m.BetaSizeCon = aml.Constraint(self.m.L, rule=_beta_size_cons)

        def _lamb_size_cons(model, i):
            return (-100, model.lambda_[i], 100)

        self.m.LambSizeCon = aml.Constraint(self.m.I, rule=_lamb_size_cons)

    def add_conv(self, conv_min: float = 0):
        """This function restricts the argument of the CDF and PDF such
        that they are above a set limit, commonly zero. This restricts
        the domain to a region whereby the 1-CDF is convex."""

        def _con_cons(model, i, k):
            return (
                model.lambda_[i]
                - sum(model.beta[l] * self._X[i][k][l] for l in model.L)
                >= conv_min
            )

        self.m.convcon = aml.Constraint(self.m.I, self.m.K, rule=_con_cons)

    def model_solve(
        self, solver, solver_exec_location=None, tee: bool = False, **kwargs
    ):
        """Start a solver to solve the model"""
        self.solver = aml.SolverFactory(solver, executable=solver_exec_location)
        self.solver.options.update(kwargs)
        return self.solver.solve(self.m, tee=tee)

    def _calc_grad_lambda_beta(self, beta_iterate, corr_lambs):
        f_arg_input = corr_lambs - np.dot(self._X, beta_iterate).T
        f_lambda = self._vpdf(f_arg_input)
        numerator = np.sum(
            (self._X.T * f_lambda), 1
        )  # sum 1 means sum over all choices
        denominator = np.sum(f_lambda, 0)
        return (numerator / denominator).T

    def _calc_grad_ll_beta(
        self, beta_iterate: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """This function is the part where the gradient is actually calculated."""
        corr_lambs = np.empty(self._num_indiv)
        for i in range(self._num_indiv):
            corr_lambs[i] = util.find_corresponding_lambda(
                self._cdf, beta_iterate, self._X[i]
            )
        grad_mat = (
            (
                (
                    self._X
                    - self._calc_grad_lambda_beta(beta_iterate, corr_lambs)[
                        :, np.newaxis, :
                    ]
                )
                * ((self._vpdf(corr_lambs - np.dot(self._X, beta_iterate).T)).T)[
                    :, :, np.newaxis
                ]
            )
            / (1 - (self._vcdf(corr_lambs - np.dot(self._X, beta_iterate).T)).T)[
                :, :, np.newaxis
            ]
        ) * self._Z[:, :, np.newaxis]
        return grad_mat.sum((0, 1)), corr_lambs

    def grad_desc(
        self,
        initial_beta: npt.NDArray[np.float64],
        max_steps: int = 50,
        grad_mult=1,
        eps: float = 10**-7,
        verbosity=0,
    ):
        """
        Starts a gradient-descent based method using the CDF and PDF.
        Requires a starting beta iterate.

        TODO: to add f_arg_min which will be pass onto the gradient
        calculators and use grad_lambda_beta to move towards
        a convex region, which is above f_arg_min

        TODO: Add some sort of linesearch method. That is, a method
        that uses a direction and tries varies stepsizes to check
        what happens to the loglikelihood with those stepsizes
        and uses some rules or heuristics to decide which
        stepsize to choose
        """
        last_log_lik = self.ll(initial_beta)
        beta_iterate = initial_beta  # initialize
        for num_step in range(max_steps):
            grad, corr_lambs = self._calc_grad_ll_beta(beta_iterate)
            beta_iterate: npt.NDArray[np.float64] = beta_iterate + grad_mult * (
                grad / (num_step + 1)
            )
            cur_ll = self.ll(beta_iterate, corr_lambs=corr_lambs)
            if verbosity == 1:
                print(cur_ll)
            if math.isnan(cur_ll):
                print("An Error occurred in calculating Loglikelihood")
                break  # no point continuing when LL has is nan
            # once no more big gains are made, stop
            if abs(last_log_lik - cur_ll) < eps:
                break
            last_log_lik = cur_ll
        return beta_iterate
