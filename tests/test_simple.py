from string import ascii_uppercase as letters
import pandas as pd
import scipy.stats as stats
import numpy as np
import mdmpy


def make_example_df() -> pd.DataFrame:
    NUM_INDIV = 57
    NUM_CHOICES = 3
    NUM_ATTR = 4

    np.random.seed(2019)
    X = np.random.random((NUM_ATTR, NUM_INDIV * NUM_CHOICES))
    true_beta = np.random.random(NUM_ATTR)
    V = np.dot(true_beta.T, X)
    V = np.reshape(V, (NUM_INDIV, NUM_CHOICES))
    eps = stats.gumbel_r.rvs(size=NUM_INDIV * NUM_CHOICES)
    eps = np.reshape(eps, (NUM_INDIV, NUM_CHOICES))
    U = V + eps
    highest_util = np.argmax(U, 1)

    df = pd.DataFrame(X.T)
    df["choice"] = [
        1 if idx == x else 0 for idx in highest_util for x in range(NUM_CHOICES)
    ]
    df["individual"] = [indiv for indiv in range(NUM_INDIV) for _ in range(NUM_CHOICES)]
    df["altvar"] = [
        altlvl for _ in range(NUM_INDIV) for altlvl in letters[:NUM_CHOICES]
    ]
    return df


def test_gradient_descent():
    df = make_example_df()
    mdm = mdmpy.MDM(df, 4, 3, [0, 1, 2, 3])
    np.random.seed(4)
    init_beta = np.random.random(4)
    grad_beta = mdm.grad_desc(init_beta)
    assert np.allclose(grad_beta, [0.30238122, 0.07955214, 0.86779824, 0.50951981])


def test_solver():
    df = make_example_df()
    mdm = mdmpy.MDM(df, 4, 3, [0, 1, 2, 3])
    mdm.model_init()
    mdm.model_solve("ipopt")

    sols = [mdm.m.beta[idx].value for idx in mdm.m.beta]
    assert np.allclose(
        sols,
        [
            0.30238834989235025,
            0.07953888508425154,
            0.8678050334295714,
            0.5095096796373667,
        ],
    )
