# MDM Py

[![Documentation Status](https://readthedocs.org/projects/mdmpy/badge/?version=latest)](https://mdmpy.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/c13f603535364a7ba5a6a18ea6756a64)](https://app.codacy.com/app/justanothergithubber/mdmpy?utm_source=github.com&utm_medium=referral&utm_content=justanothergithubber/mdmpy&utm_campaign=Badge_Grade_Dashboard)
[![PyPI version](https://badge.fury.io/py/mdmpy.svg)](https://badge.fury.io/py/mdmpy)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

This package is a `Python` implementation of
[Marginal Distribution Models](https://pubsonline.informs.org/doi/10.1287/mnsc.2014.1906)
(MDMs), which can be used in Discrete Choice Modelling.

## Documentation

Documentation is kindly hosted by
[Read The Docs](https://mdmpy.readthedocs.io/ "mdmpy Documentation").

## Install

This package is uploaded to
[PyPI](https://pypi.org/ "Python Package Index"). Hence,

```bat
pip install mdmpy
```

should work.

## How to use

### Simplest Case

#### Gradient Descent

In the simplest case, we will use the Multinomial Logit (MNL) model,
which is used as a default. Assuming `numpy`, `scipy` and `pandas` are
installed, we generate choice data assuming a random utility model:

```python
from string import ascii_uppercase as letters
import pandas as pd
import scipy.stats as stats
import numpy as np

NUM_INDIV   = 57
NUM_CHOICES = 3
NUM_ATTR    = 4

np.random.seed(2019)
X = np.random.random((NUM_ATTR, NUM_INDIV * NUM_CHOICES))
true_beta = np.random.random(NUM_ATTR)
V = np.dot(true_beta.T, X)
V = np.reshape(V, (NUM_INDIV,NUM_CHOICES))
eps = stats.gumbel_r.rvs(size=NUM_INDIV * NUM_CHOICES)
eps = np.reshape(eps, (NUM_INDIV, NUM_CHOICES))
U = V + eps
highest_util = np.argmax(U, 1)

df = pd.DataFrame(X.T)
df['choice'] = [1 if idx == x else 0 for idx in highest_util for x in range(NUM_CHOICES)]
df['individual'] = [indiv for indiv in range(NUM_INDIV) for _ in range(NUM_CHOICES)]
df['altvar'] = [altlvl for _ in range(NUM_INDIV) for altlvl in letters[:NUM_CHOICES]]
```

With this package, we will assume that `df` is the dataframe which is
simply given to us. Instead of having the code itself find out how
many individuals, choices and coefficients or attributes there are, we
will simply feed them into the class. To perform a gradient descent
with this class, we will use the `grad_desc` method, using the `df`
from above as input,

```python
import mdmpy

# In a typical case one would load df before this line
mdm = mdmpy.MDM(df, 4, 3, [0, 1, 2, 3])
np.random.seed(4)
init_beta = np.random.random(4)
grad_beta = mdm.grad_desc(init_beta)
print(grad_beta)
# expected output [0.30238122 0.07955214 0.86779824 0.50951981]
```

#### Solver

The `MDM` class acts as a wrapper and adds the necessary `pyomo`
variables and sets to model the problem, but requires a solver.
[IPOPT](https://projects.coin-or.org/Ipopt "Ipopt home page"), an
interior point solver, is recommended. If you have such a solver, it
can be called. Assuming IPOPT is being used:

```python
import mdmpy

ipopt_exec_path = /path/to/ipopt # Replace with proper path
mdm = mdmpy.MDM(df, 4, 3, [0, 1, 2, 3])
mdm.model_init()
mdm.model_solve("ipopt",ipopt_exec_path)
print([mdm.m.beta[idx].value for idx in mdm.m.beta])
# expected output [0.30238834989235025, 0.07953888508425154, 0.8678050334295714, 0.5095096796373667]
```
