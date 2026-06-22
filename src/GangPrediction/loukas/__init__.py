"""From-scratch implementations of:

* Loukas (2019) "Graph Reduction with Spectral and Cut Guarantees" — restricted
  spectral approximation (RSA) coarsening by *local variation* (``loukas_coarsening``).
* Section 8-12 of the project note "Spectral Energy of Planted Gangs" — the
  trainable Simple-Graph-Convolution (SGC) polynomial filter that builds the
  coarsening target subspace R = span(g_theta(A_hat) X) (``sgc_subspace``).

The two modules are intentionally self-contained (numpy / scipy only) so they can
be read and audited against the two source papers without touching the rest of
the GangPrediction pipeline.
"""

from . import loukas_coarsening, sgc_subspace  # noqa: F401
