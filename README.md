# MDM Py

This package is a `Python` implementation of Marginal Distribution Models (MDMs), which can be used in Discrete Choice Modelling.

# Install

This package should eventually be uploaded onto PyPI when the package is more ready. In that case:

```
pip install mdmpy
```

should work.

# How to use
## Simplest Case
### Gradient Descent
In the simplest case, we will use the Multinomial Logit (MNL) model, which is used as a default. Assuming `numpy`, `scipy` and `pandas` are installed, we generate choice data assuming a random utility model:

```python
from string import ascii_uppercase as letters
import pandas as pd
import scipy.stats as stats
import numpy as np

NUM_INDIV = 57
NUM_CHOICES = 3
NUM_ATTR    = 4

np.random.seed(2019)
X = np.random.random((NUM_ATTR,NUM_INDIV*NUM_CHOICES))
true_beta = np.random.random(NUM_ATTR)
V = np.dot(true_beta.T,X)
V = np.reshape(V,(NUM_INDIV,NUM_CHOICES))
eps = stats.gumbel_r.rvs(size=NUM_INDIV*NUM_CHOICES)
eps = np.reshape(eps,(NUM_INDIV,NUM_CHOICES))
U = V+eps
highest_util = np.argmax(U,1)

df = pd.DataFrame(X.T)
df['choice'] = [1 if idx==x else 0 for idx in highest_util for x in range(NUM_CHOICES)]
df['individual'] = [indiv for indiv in range(NUM_INDIV) for _ in range(NUM_CHOICES)]
df['altvar'] = [altlvl for _ in range(NUM_INDIV) for altlvl in letters[:NUM_CHOICES]]
```

With this package, we will assume that `df` is the dataframe which is simply given to us. Instead of having the code itself find out how many individuals, choices and coefficients or attributes there are, we will simply feed them into the class. To perform a gradient descent with this class, we will use the `grad_desc` method.

```python
import mdmpy

# In a typical case one would load df here
mdm = mdmpy.MDM(df,4,3,[0,1,2])
init_beta = np.random.random(3)
grad_beta = mdm.grad_desc(init_beta)
print(grad_beta)
# expected output [0.30031206 0.01145427 0.93931724]
```

### Solver
The `MDM` class acts as a wrapper and adds the necessary `pyomo` variables and sets to model the problem, but requires a solver. [IPOPT](https://projects.coin-or.org/Ipopt), an interior point solver, is recommended. If you have such a solver, we can call it. Assuming we are using IPOPT,

```python
import mdmpy

ipopt_exec_path = /path/to/ipopt
mdm = mdmpy.MDM(df,4,3,[0,1,2])
mdm.model_init()
mdm.model_solve("ipopt",ipopt_exec_path)
print([mdm.m.beta[idx].value for idx in mdm.m.beta])
# expected output [0.30031260573741614, 0.011454067389814948, 0.9393163389096663]
```

# Todo

1. Add documentation.
2. Add tests.