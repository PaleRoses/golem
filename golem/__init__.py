"""GOLEM v0.3 -- the living lane.

Semantic layout:

- ``kernel/``    solvers and compilers: ``engine`` (quarantined vocabulary),
                 ``body`` (stance compiler), ``anatomy`` (vascular descent),
                 ``sheaf`` (Field -> Section Laplacian).
- ``senses/``    the perception plane: ``proprio``, ``symmetry``,
                 ``silhouette``, ``elemental_render``.
- ``contracts/`` postural contracts: ``asserts``.
- ``evals/``     executable instruments; run via ``python -m golem.evals.<name>``.
- ``goldentext`` byte-exact write-once golden harness.
- ``paths``      all repo geometry + the single quarantined pilots bootstrap.

The frozen pilots (``engine``, ``render``, ``hand``, ``judge``) stay plain
top-level modules; importing this package makes them resolvable.
"""

from golem import paths as _paths  # noqa: F401  (pilots bootstrap side effect)
