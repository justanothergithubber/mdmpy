import math
import numpy as np
import pyomo.environ as aml
from . import util

class MDM:
    """
    The main class of the package. This models the Marginal Distribution
    Models (MDM)
    """
    def __init__(self, input_dataframe, ch_id:int, num_choices:int,
                 list_of_coefs,
                 input_cdf = util.exp_cdf,
                 input_pdf = util.exp_pdf):
        """"The class is initialised with a dataframe, along with the relevant
        column indices of the dataframe, such as the choice index, the number of
        and the coefficients that will be taken into account. All of such data
        are reused in the class methods.

        Default CDF/PDF of the model is the Multinomial Logit (MNL) model, which
        is globally convex in its support. But of course this can be changed.

        This will require some other changes to allow for individual-specific
        coefficients, which will be added at a later date.
        """
        num_indiv = int(input_dataframe.shape[0]/num_choices)
        # check if numerically equivalent
        if not num_indiv == input_dataframe.shape[0]/num_choices:
            raise ValueError("""Unexpectedly, the number of columns does not divide
                                by the number of choices, as inputed.""")
        num_attr = len(list_of_coefs)
        Z = list(zip(*[iter(input_dataframe.iloc[:, ch_id])]*num_choices))
        X = np.reshape(np.array(input_dataframe.iloc[:, list_of_coefs]),
                       (num_indiv, num_choices, num_attr))
        self._X = X
        self._Z = Z
        self._ch_id       = ch_id       # choice column index
        self._num_indiv   = num_indiv   # number of individuals
        self._num_attr    = num_attr    # number of attributes/coefficients
        self._num_choices = num_choices # number of alternatives/choices
        self._cdf = input_cdf   # sets the cdf for the model
        self._pdf = input_pdf   # sets corresponding pdf (has to be inputted, not automatic)

    def ll(self, input_beta, corr_lambs = None) -> float:
        """This function gets the log-likelihood using the current beta. If
        the corresponding lambdas for each individual are given, then it will
        use those, rather than re-computing them, which saves computations"""
        loglik = 0
        for i in range(self._num_indiv):
            x_i = self._X[i]
            if corr_lambs is None:
                cor_lamb = util.find_corresponding_lambda(self._cdf, x_i, input_beta)
            else:
                cor_lamb = corr_lambs[i]
            for k, choice in enumerate(self._Z[i]):
                if choice:
                    x_ik = x_i[k]
                    f_arg_ik = cor_lamb - sum(x*y for x, y in zip(input_beta, x_ik))
                    loglik += math.log(1-self._cdf(f_arg_ik))
                else:
                    pass
        return loglik


    def model_init(self, heteroscedastic = False,
                   model_seed = None):

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
            numpy_stan_exp = np.random.standard_exponential
            self.m.beta    = aml.Var(self.m.L, initialize = lambda _: numpy_stan_exp())
            self.m.lambda_ = aml.Var(self.m.I, initialize = lambda _: numpy_stan_exp())
            print([self.m.beta[h].value for h in self.m.L])
            print([self.m.lambda_[g].value for g in self.m.I])
        else:
            self.m.beta    = aml.Var(self.m.L)
            self.m.lambda_ = aml.Var(self.m.I)
        
        # Handling heteroscedascity
        if heteroscedastic:
            # known heteroscedasticity
            if isinstance(heteroscedastic, list):
                self.m.alpha = {idx:v for idx, v in enumerate(heteroscedastic)}
            # else, unknown heteroscedasticity
            else:
                self.m.alpha = aml.Var(self.m.K, domain=aml.PositiveReals)
#                 self.m.AlphaSumCons = aml.Constraint(expr=sum(self.m.alpha[k] for k in self.m.K)==num_choices)
                self.m.FixOneAlphaC = aml.Constraint(expr=self.m.alpha[0] == 1)

                def _tol_cons(model, k, ALPHA_TOL = 0.3):
                    return model.alpha[k] >= ALPHA_TOL

                self.m.AlphaTol = aml.Constraint(self.m.K,rule = _tol_cons)

        # Objective Function
        if heteroscedastic:
            if self._cdf == util.exp_cdf:
                O_expr = sum(sum(
                    self._Z[i][k]*self.m.alpha[k]*(sum(
                        self.m.beta[l]*self._X[i][k][l] for l in self.m.L)
                             -self.m.lambda_[i]) for k in self.m.K) for i in self.m.I)
            else:
                O_expr = sum(sum(
                    self._Z[i][k]*self.m.alpha[k]*aml.log(1-self._cdf(
                        self.m.lambda_[i]-sum(
                            self.m.beta[l]*self._X[i][k][l] for l in self.m.L))) for k in self.m.K) for i in self.m.I)
        else:
            # Model CDF simplifications
            if self._cdf == util.exp_cdf:
                O_expr = sum(sum(
                    self._Z[i][k]*(sum(
                        self.m.beta[l]*self._X[i][k][l] for l in self.m.L)
                             -self.m.lambda_[i]) for k in self.m.K) for i in self.m.I)

            else:
                O_expr = sum(sum(
                    self._Z[i][k]*aml.log(1-self._cdf(
                        self.m.lambda_[i]-sum(
                            self.m.beta[l]*self._X[i][k][l] for l in self.m.L))) for k in self.m.K) for i in self.m.I)

        # Model Objective
        self.m.O = aml.Objective(expr=O_expr, sense=aml.maximize)

        # Lagrangian Constraints (for each individual)
        # MEM
        if heteroscedastic and self._cdf == util.exp_cdf:
            def lag_cons(model, i):
                return sum(aml.exp(model.alpha[k]*(sum(
                    model.beta[l]*self._X[i][k][l] for l in model.L)-model.lambda_[i])) for k in model.K) <= 1
        else:
            def lag_cons(model, i):
                return sum(1-self._cdf(model.lambda_[i]-sum(
                    model.beta[l]*self._X[i][k][l] for l in model.L)) for k in model.K) <= 1
        self.m.C = aml.Constraint(self.m.I,rule=lag_cons)

        # Scale restriction - not required
        # but might help solver not get lost and diverge
        def _beta_size_cons(model, l):
            return (-20, model.beta[l], 20)
        self.m.BetaSizeCon = aml.Constraint(self.m.L, rule = _beta_size_cons)

        def _lamb_size_cons(model, i):
            return (-100, model.lambda_[i], 100)
        self.m.LambSizeCon = aml.Constraint(self.m.I, rule = _lamb_size_cons)

    def add_conv(self, conv_min: float = 0):
        """This function restricts the argument of the CDF and PDF such
        that they are above a set limit, commonly zero. This restricts
        the domain to a region whereby the 1-CDF is convex."""
        def _con_cons(model, i, k):
            return model.lambda_[i]-sum(model.beta[l]*self._X[i][k][l] for l in model.L) >= conv_min
        self.m.convcon = aml.Constraint(self.m.I, self.m.K, rule = _con_cons)

    def model_solve(self, solver, solver_exec_location, tee: bool = False, **kwargs):
        """Start a solver to solve the model"""
        self.solver = aml.SolverFactory(solver,executable=solver_exec_location)
        self.solver.options.update(kwargs)
        return self.solver.solve(self.m,tee=tee)

    def grad_desc(self, initial_beta,
                  max_steps: int = 50, f_arg_min = None,
                  eps: float = 10**-7):
        """Starts a gradient-descent based method using the CDF and PDF.
        Requires a starting beta iterate. f_arg_min is to ensure that
        the argument stays above a certain value, which is useful for
        ensuring that certain Probability Distributions remain in their
        convex regions."""
        last_log_lik = self.ll(initial_beta)
        beta_iterate = initial_beta #initialize
        for num_step in range(max_steps):
            grad = np.zeros(self._num_attr)
            corr_lambs = {}
            for i in range(self._num_indiv):
                x_i = self._X[i]
                corr_lambs[i] = util.find_corresponding_lambda(self._cdf, x_i, beta_iterate)
                cor_lamb = corr_lambs[i]
                for k, choice in enumerate(self._Z[i]):
                    if choice:
                        vector_collector = np.zeros(self._num_attr)
                        denom = 0
                        for x_im in x_i: # m var unused
                            f_arg_im = cor_lamb - sum(x*y for x,y in zip(beta_iterate, x_im))
                            if f_arg_min is not None:
                                if f_arg_min >= f_arg_im: raise AssertionError
                            vector_collector = vector_collector + (self._pdf(f_arg_im) * x_im)
        #                     print(vector_collector)
                            denom += self._pdf(f_arg_im)
                        x_ik = x_i[k]
                        f_arg_ik = cor_lamb - sum(x*y for x,y in zip(beta_iterate, x_ik))
                        if f_arg_min is not None:
                            if f_arg_min >= f_arg_ik: raise AssertionError
                        grad = grad + (((x_ik - (vector_collector / denom)) * self._pdf(f_arg_ik)) /
                                       (1-self._cdf(f_arg_ik)))
                    else:
                        pass
            beta_iterate = beta_iterate + grad/(num_step+1)
            # once no more big gains are made, stop
            cur_ll = self.ll(beta_iterate, corr_lambs = corr_lambs)
            if abs(last_log_lik-cur_ll)<eps:
                break
            last_log_lik = cur_ll
        return beta_iterate
