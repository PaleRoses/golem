# Dissect the residual multiplicity vertex on the Sable Stride formed solve.
# Reproduces the solve internals (scaffold -> relaxation) and, for each vertex
# the relaxed-segment uniqueness check rejects, reports whether the SCAFFOLD
# segment was already multi-crossing (genuine projection ambiguity) or the
# tangential relaxation shift introduced a grazing chord (false positive).
from __future__ import annotations

import sys
from functools import reduce
from pathlib import Path

import json
import numpy as np

from golem.kernel import body, engine
from golem.kernel.engine import compile as C
from golem.kernel.engine.types import decode_graph, require_accepted

SP = Path("specs/sable_stride_formed.json")


def main() -> None:
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    compiled = body.Compiler(json.loads(SP.read_text()), spec_dir=SP.parent).compile()
    graph = require_accepted(decode_graph(compiled.graph))
    problem = graph.skin
    policy = C.SEALED_SKIN_POLICY
    # Rebuild the flesh morphology the solver consumes (skin stripped).
    from dataclasses import replace as dc_replace
    stripped = dc_replace(graph, skin=None)
    flesh_only = C.evaluate_checked(stripped, res)
    assert isinstance(flesh_only, engine.EvaluatedMorphology), flesh_only
    sections = C._skin_formation_sections(graph, flesh_only)

    flesh_field = np.asarray(flesh_only.field, dtype=np.float64)
    skin_field = flesh_field - problem.thickness
    scaffold_result = C._skin_local_scaffold(
        flesh_only, skin_field, problem, policy
    )
    assert not isinstance(scaffold_result, engine.SkinRelaxationObstruction), scaffold_result
    scaffold, min_grad, normals = scaffold_result
    faces = np.asarray(flesh_only.faces, dtype=np.int64)
    lower = np.asarray(flesh_only.lower, dtype=np.float64)
    pitch = np.asarray(flesh_only.world_pitch, dtype=np.float64)
    maximum_pitch = flesh_only.maximum_pitch
    flesh_vertices = np.asarray(flesh_only.vertices, dtype=np.float64)
    adjacency = C._skin_adjacency(faces, scaffold, normals, maximum_pitch, policy)
    initial = C._SkinIterationState(scaffold, min_grad, 0.0, (), ())

    def relax(state, _i):
        return C._skin_iteration(
            state, scaffold, adjacency, skin_field, lower, pitch,
            maximum_pitch, policy,
        )

    relaxed = reduce(relax, range(policy.iteration_budget), initial)
    tol = max(1.0e-12, policy.minimum_gradient * maximum_pitch)
    scaffold_cross = C._skin_segment_crossings(
        flesh_vertices, scaffold, skin_field, lower, pitch,
        policy.minimum_gradient, tol,
    )
    relaxed_cross = C._skin_segment_crossings(
        flesh_vertices, relaxed.vertices, skin_field, lower, pitch,
        policy.minimum_gradient, tol,
    )
    scaffold_over = np.flatnonzero(scaffold_cross > 1)
    relaxed_over = np.flatnonzero(relaxed_cross > 1)
    print(f"res={res} vertices={len(flesh_vertices)}")
    print(f"scaffold overshoots: {scaffold_over.size}")
    print(f"relaxed overshoots:  {relaxed_over.size}")
    disp = np.linalg.norm(relaxed.vertices - flesh_vertices, axis=1)
    print("flesh bbox lo", flesh_vertices.min(axis=0))
    print("flesh bbox hi", flesh_vertices.max(axis=0))
    for v in scaffold_over.tolist():
        d = np.linalg.norm(scaffold - scaffold[v], axis=1)
        d[v] = np.inf
        print(f"== SCAFFOLD-over vertex {v} flesh={flesh_vertices[v]}")
        print(f"   scaffold cross={scaffold_cross[v]} disp/t={np.linalg.norm(scaffold[v]-flesh_vertices[v])/problem.thickness:.3f}")
        print(f"   nearest other scaffold vertex dist/pitch {np.min(d)/maximum_pitch:.3f}")
    for v in relaxed_over.tolist():
        print(f"-- vertex {v}")
        print(f"   flesh    {flesh_vertices[v]}")
        print(f"   scaffold {scaffold[v]}  cross={scaffold_cross[v]}")
        print(f"   relaxed  {relaxed.vertices[v]}  cross={relaxed_cross[v]}")
        print(f"   displacement/thickness {disp[v]/problem.thickness:.3f}")
        print(f"   scaffold displacement/thickness {np.linalg.norm(scaffold[v]-flesh_vertices[v])/problem.thickness:.3f}")
        # nearest other relaxed skin vertex distance
        d = np.linalg.norm(relaxed.vertices - relaxed.vertices[v], axis=1)
        d[v] = np.inf
        print(f"   nearest other skin vertex dist/pitch {np.min(d)/maximum_pitch:.3f}")


if __name__ == "__main__":
    main()
