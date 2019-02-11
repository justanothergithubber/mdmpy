Examples
========

Travel Mode Choice
------------------

We look at the Travel Choice data as examined in Chapter 18 of :cite:`9780131395381`.

Preprocessing
^^^^^^^^^^^^^

We will use :mod:`requests` to get the dataset, and then process it with :mod:`csv` and :mod:`pandas`.

.. code-block:: python

   import csv
   import requests
   import pandas as pd
   import numpy as np
   
   url = "http://pages.stern.nyu.edu/~wgreene/Text/Edition7/TableF18-2.csv"
   # Preallocate 840 rows and 7 columns, which is the size of the data
   # We will skip the header row
   df = pd.DataFrame(index=np.arange(0, 840), columns=np.arange(0,7))
   with requests.Session() as s:
       download = s.get(url)
       cr = csv.reader(download.content.decode().splitlines())
       next(cr) # skip header
       for ix, row in enumerate(cr):
           df.loc[ix] = [int(x) for x in row]

Solving for the MLE
^^^^^^^^^^^^^^^^^^^

We can initialise the model using the :meth:`model_init` method.

.. code-block:: python

   import mdmpy
   mdm0 = mdmpy.MDM(df,   # input dataframe
                    0,    # the choice is index 0
                    4,    # there are 4 possible choices
                    [2,3])
   mdm0.model_init()


Assuming that the `IPOPT` solver is in `PATH`, we can choose to simply put `"IPOPT"` as an argument to the :meth:`model_solve` method.

.. code-block:: python

   mdm0.model_solve("ipopt")
   beta0 = [mdm0.m.beta[idx].value for idx in mdm0.m.beta]
   print(beta0)

We can also get the loglikelihood of such an outcome.

.. code-block:: python

   ll0 = mdm0.ll(beta0)
   print(ll0)

Alternatively, we can use all of the data of the choices:

.. code-block:: python

   mdm1 = mdmpy.MDM(df,   # input dataframe
                    0,    # the choice is index 0
                    4,    # there are 4 possible choices
                    [1, 2, 3, 4] # now instead we use all 4 choice-specific-data columns
                    )
   mdm1.model_init()
   mdm1.model_solve("ipopt")
   beta1 = [mdm1.m.beta[idx].value for idx in mdm1.m.beta]
   print(beta1)
   ll1 = mdm1.ll(beta1)
   print(ll1)

Full Code
^^^^^^^^^

.. code-block:: python

   import csv
   import requests
   import pandas as pd
   import numpy as np
   import mdmpy
   
   url = "http://pages.stern.nyu.edu/~wgreene/Text/Edition7/TableF18-2.csv"

   df = pd.DataFrame(index=np.arange(0, 840), columns=np.arange(0,7))
   with requests.Session() as s:
       download = s.get(url)
       cr = csv.reader(download.content.decode().splitlines())
       next(cr) # skip header
       for ix, row in enumerate(cr):
           df.loc[ix] = [int(x) for x in row]

   mdm0 = mdmpy.MDM(df,   # input dataframe
                    0,    # the choice is index 0
                    4,    # there are 4 possible choices
                    [2,3] # we will use columns index 2 and 3 (3rd and 4th column, Python indices start from 0)
                    )
   mdm0.model_init()
   mdm0.model_solve("ipopt")
   beta0 = [mdm0.m.beta[idx].value for idx in mdm0.m.beta]
   print(beta0)
   ll0 = mdm0.ll(beta0)
   print(ll0)

   mdm1 = mdmpy.MDM(df,   # input dataframe
                    0,    # the choice is index 0
                    4,    # there are 4 possible choices
                    [1, 2, 3, 4] # now instead we use all 4 choice-specific-data columns
                    )
   mdm1.model_init()
   mdm1.model_solve("ipopt")
   beta1 = [mdm1.m.beta[idx].value for idx in mdm1.m.beta]
   print(beta1)
   ll1 = mdm1.ll(beta1)
   print(ll1)

.. bibliography:: references.bib
   :filter: docname in docnames