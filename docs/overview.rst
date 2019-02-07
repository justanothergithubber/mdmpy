Overview
--------
We follow :cite:`Mishra2014` and wish to solve the following Maximum Likelihood problem:

.. math::
   :nowrap:

   \begin{align*}
   \max_{\bf{\Lambda}, \bf{B}} &\  \sum_{i\in\mathcal{I}} \sum_{k\in\mathcal{K}}z_{ik}\ln\left(1-F_{ik}\left(\lambda_{i}-\bf{B}'\mathbf{x_{ik}}\right)\right) \\
   \text{s.t.} &\  \sum_{k\in\mathcal{K}}\left(1-F_{ik}\left(\lambda_{i}-\bf{B}'\mathbf{x_{ik}}\right)\right)=1, \quad \forall i\in\mathcal{I}
   \end{align*}

.. note::
   Due to limitations with MathJax, Greek lower-case letters cannot be bolded. Hence, we use upright Greek bolded letters to denote vectors.

To do so, we can use the :class:`mdmpy.mdm.MDM` class, which includes a method for initialising the model in `Pyomo`, and another to solve it. In the next section, we will explore a few examples to see this in action.

.. bibliography:: references.bib
   :filter: docname in docnames

