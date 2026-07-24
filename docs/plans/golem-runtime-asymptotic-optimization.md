# Collapse GOLEM runtime asymptotics without changing verdicts

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document is maintained in accordance with `docs/llm-guidance/PLANS.md` from the repository root.

## Purpose / Big Picture

GOLEM's authoring loop must return the same typed diagnostics, assembly verdict, geometry, vascular graph, and receipts substantially faster and with less peak memory. The representative accepted workload is `specs/vigil_hound_demo.json`, which contains 41 authored parts, 15 mirrored parts, 56 concrete instances, four circulation circuits, and 256 terminal pairs. The initial three cold `python -m golem check` samples were 11.21, 11.86, and 11.61 seconds, for an 11.61-second median; median CPU time was 14.65 seconds and median peak resident memory was 2,091,433,984 bytes. After this work, the same command must produce byte-identical standard output and the same exit status while repeated cold measurements demonstrate the retained speed and memory change.

The work is not licensed to weaken a resolution, omit assembly acceptance, reduce the terminal-pair budget, alter a tolerance, accept an obstruction, change a mesh, or rely on a successful cache hit. Performance is accepted only when the local sections still glue into exactly the same global result.

## Progress

- [x] (2026-07-23 11:44Z) Read repository planning, sediment, functional-programming, architecture, command, session, assembly, geometry, and vascular ownership surfaces.
- [x] (2026-07-23 11:44Z) Captured the initial cold end-to-end baseline for `specs/vigil_hound_demo.json`: exit 0, 11.32 seconds wall time, 2,068,283,392 bytes maximum resident set size.
- [x] (2026-07-23 11:44Z) Captured a cold `cProfile` receipt: 15.001 profiled seconds, with 9.817 seconds under assembly compilation, 8.843 cumulative seconds across two vasculature realizations, 5.439 seconds in vascular tree construction, and 4.884 seconds in surface evaluation.
- [x] (2026-07-23 20:20Z) Recorded the byte-exact stdout baseline, three-sample cold median, full evaluated-field/mesh hashes, vascular cache growth, and independent terminal-budget and resolution scale curves.
- [x] (2026-07-23 20:20Z) Retained four source-proven eliminations: exact vascular cache projection, prepared SDF reuse, constant-station interpolation elision, and neutral-element regional gluing. Moved obligation clearance after the authoritative self-clearance obstruction.
- [x] (2026-07-23 20:20Z) Rejected three semantically exact but slower candidates: regional concatenate gluing, slabwise smooth blending, and process parallelism below resolution 256.
- [x] (2026-07-23 20:20Z) Passed 46 focused engine/profile tests, 6 slow owner tests, 38 vascular/anatomy tests, 90 acceptance/session tests, the 1,031-test living non-slow suite, and sealed conformance with 25 passes and 3 declared xfails.
- [x] (2026-07-23 20:20Z) Repeated cold end-to-end measurements and completed the architectural-closure audit. No parallel owner, adapter, compatibility layer, generated artifact, or dependency was added.

## Surprises & Discoveries

- Observation: the documented statement that `check` never meshes is no longer true in the live implementation.
  Evidence: `golem/session/protocol.py::_evaluate_authoring_state` calls `compile_assembly`; `golem/assembly/compile.py::_compile_element` calls `golem.kernel.engine.evaluate`; the cold profile attributes 4.874 cumulative seconds to that call. The CLI still prints `no mesh`, so text is not proof of execution topology.

- Observation: an accepted body is realized as vasculature twice during one `check`.
  Evidence: `golem/session/protocol.py::_compiled_evidence` and `golem/assembly/compile.py::_compile_element` each call `realize_vasculature`; the profile records two calls totaling 8.842 cumulative seconds. This is a candidate for direct result reuse, not for a second cache or facade.

- Observation: the cold authoring command is memory-bound as well as compute-bound.
  Evidence: `/usr/bin/time -lp` reports about 2.07 GB maximum resident set size for a 56-instance body. The surface evaluator constructs resolution-cubed fields and repeatedly composes local sections into the global field; memory acceptance must therefore be measured, not inferred from wall time.

- Observation: the two vascular descents were semantically identical but received graph dictionaries that differed by derived assembly views.
  Evidence: the session graph and assembly graph had identical anatomy and identical consumed `parts`, `intent.provenance`, and `intent.landmarks`; the assembly graph additionally carried `conduits`. `realize_vasculature` reads only the former projection. Canonicalizing that exact consumed section turns the second descent into the existing LRU result, with no new cache or authority.

- Observation: `prepared_part_sdf` was already the canonical prepared evaluator, but ordinary raster sampling bypassed it except for skeleton-integral muscle.
  Evidence: all 56 canonical instances compared direct and prepared evaluation over 10,000 deterministic points with exact array equality and maximum difference zero. Routing `_sample_part_sdf` through the existing owner removes repeated profile sweep, axes, rotation, and primitive setup.

- Observation: the profiled hound has constant station series for many width, exponent, depth, offset, and roll coordinates.
  Evidence: `_prepared_collinear_profile` called `np.interp` for every sampled point even though the existing `_station_values` law returns the same constant field without a search. Focused prepared-vs-direct tests are bit exact for plain gencyl, profiled gencyl, blob, and box.

- Observation: localized chamfer and crease gluing allocated a global boolean mask and a global `where` result despite possessing exact neutral elements.
  Evidence: positive infinity is the identity for `minimum` and for the chamfer expression outside the local section. Exact tests compare the new neutral padding with the old masked formulation for both operators.

- Observation: retained scale is not hidden by one favorable specimen.
  Evidence: accepted terminal budgets 64, 128, and 256 took 0.564, 1.093, and 2.374 seconds and produced 375, 695, and 1,336 edges. Surface resolutions 64, 96, 128, 160, 192, and 222 took 0.156, 0.504, 0.951, 1.578, 2.533, and 3.703 seconds for 262,144 through 10,941,048 voxels.

- Observation: four remaining growth laws are semantic work, not disposable ceremony.
  Evidence: legacy exact fields still require resolution-cubed raster formation; accepted CCO insertion repeatedly checks a bounded candidate family against a growing tree; `_possible_vascular_edge_pairs` forms a dense edge-pair broad phase; balance calibration performs repeated sparse solves. Changing any of these algorithms without a new proof would change fields, tie-breaking, obstruction order, or solver receipts.

- Observation: process-lifetime memo tables remain an unbounded growth risk.
  Evidence: one canonical command left 9,880 packed-capsule entries (14,451,360 live array bytes) and 7,976 packed-segment entries (6,741,120 live array bytes), in addition to identity memos in `geometry.py`. Replacing these requires threading an immutable prepared carrier through the constructive search; another cache would merely lacquer the tumor.

## Decision Log

- Decision: use the live `check specs/vigil_hound_demo.json` path as the primary end-to-end authority and bypass persisted circuit and field caches with `GOLEM_CIRCUIT_CACHE_COLD=1`.
  Rationale: the command exercises body compilation, senses, vascular feasibility, assembly acceptance, and surface formation on a tracked, accepted canonical workload. Cache bypass proves computation rather than filesystem luck.
  Date/Author: 2026-07-23 / Codex

- Decision: preserve stdout, exit status, typed obstructions, vascular topology and receipts, evaluated field/mesh values, and resolution policy as independent semantic gates.
  Rationale: a faster command obtained by silently dropping any one of these sections would reduce semantics and fail the request.
  Date/Author: 2026-07-23 / Codex

- Decision: repair existing owners in this order: remove repeated descent across the session-to-assembly overlap, strengthen the vascular realization owner, then strengthen sampled-field composition. Do not add a parallel performance service, intermediate representation, or compatibility path.
  Rationale: `protocol.py`, `assembly/compile.py`, `kernel/anatomy/realize`, and `kernel/engine/compile.py` already own these concepts. New machinery would duplicate authority before the existing owners were exhausted.
  Date/Author: 2026-07-23 / Codex

- Decision: project the vascular cache identity to exactly `parts`, `intent.provenance`, and `intent.landmarks`, and test that derived views share while a part change misses.
  Rationale: these are the only graph sections consumed by `_realize_vasculature_uncached`; including downstream `conduits` made a derived view poison the semantic cache identity.
  Date/Author: 2026-07-23 / Codex

- Decision: retain assembly source snapshot recompilation instead of injecting a session `Compiled` carrier into the assembly compiler.
  Rationale: snapshot digesting and assembly compilation own source capture, mounts, appearance seeding, geometry, integrity, and coupled stages. The measured duplicate was the vascular descent, and the existing vascular owner can reuse it exactly without a new cross-layer prepared-element boundary.
  Date/Author: 2026-07-23 / Codex

- Decision: reuse `prepared_part_sdf` directly and strengthen its existing constant-station law rather than introduce a second sampler or raster IR.
  Rationale: the semantic owner already existed and had multiple primitive implementations. The defect was bypass, not absence.
  Date/Author: 2026-07-23 / Codex

- Decision: retain only the self-clearance short circuit from the CCO experiments.
  Rationale: reusing packed existing segments and prefiltering incident pairs added surface area without a repeatable profile improvement. The short circuit alone skips obligations only after the earlier authoritative obstruction is known, preserving diagnostic order.
  Date/Author: 2026-07-23 / Codex

- Decision: reject regional concatenate gluing, slabwise smooth blend, and a lower process threshold.
  Rationale: despite exact stdout, they respectively measured 10.84, 8.60, and 10.76 seconds against 7.67-8.67-second neighboring retained candidates, and the first two raised peak memory. Smaller code is not absolution for slower code.
  Date/Author: 2026-07-23 / Codex

## Outcomes & Retrospective

The dominant duplicate vascular descent is gone: the final profile records two public calls but only one `_realize_vasculature_uncached` computation. The low-contention retained profile fell from 15.001 to 9.559 profiled seconds; surface evaluation was 3.930 seconds and the single vascular solve 4.347 seconds.

The final cold samples were 8.13, 10.42, and 11.80 seconds under substantial concurrent Chrome and WindowServer load. Their 10.42-second wall median is 10.2% below 11.61 seconds; the more stable CPU median fell from 14.65 to 12.00 seconds, an 18.1% reduction. The fastest final sample was 29.9% faster. Standard output remained byte-identical in every run with SHA-256 `6df2fc40430e47de3d7ac1962b40d89ea2b65e1712b7ca857d4814bf4abf9ebf`.

An alternating clean-HEAD/current cohort under the same active desktop load measured HEAD at 13.11 and 10.58 seconds versus current at 8.42 and 8.47 seconds. Median wall time improved 28.7% and CPU time improved 19.1%; this pairing exposes the retained change more honestly than pretending the machine was thermally quiescent.

Peak resident memory measured 2,098,495,488, 2,091,008,000, and 2,100,232,192 bytes. The 2,098,495,488-byte median is 0.34% above the baseline median, within the observed allocator/runtime noise; the second alternating pair measured 2,106,589,184 bytes for HEAD and 2,110,275,584 bytes for current. Source allocation strictly decreased for neutral regional composition, but this run does not establish a lower process peak because full-grid sampling remains the dominant allocation.

All required verdicts remained authoritative: 41 parts, 15 mirrored parts, 56 instances, 1,080 vascular nodes, 1,336 edges, 256 terminal pairs, residual `3.945e-13`, balance `3.018e-13`, the same regional allocation, and accepted assembly. The work improves the owners it can prove without mutilating the result. The remaining asymptotic laws are listed explicitly rather than disguised as completion theater.

## Context and Orientation

The repository root is `/Users/bluerose/Developer/pale-meridian`; GOLEM is the Python 3.12 project in `potentialimprovements/golem-kernel`.

`golem/session/effect.py::compile_authored` compiles a body document into the immutable `golem.session.state.Compiled` outcome. That outcome already contains the derived part graph, body receipt, decoded anatomy, senses, contract verdicts, and anomalies.

`golem/session/protocol.py::evaluate_authoring_surfaces` is the authoring acceptance gluing point. Its local analytic section, `_compiled_evidence`, derives diagnostics and realizes vasculature from the compiled anatomy and graph. If that section accepts, `_evaluate_authoring_state` wraps the raw body document as a one-element assembly and calls `golem.assembly.compile_assembly`.

`golem/assembly/core.py::compile_assembly` is the assembly authority. `golem/assembly/ingest.py` snapshots and recompiles body sources into `_LoadedElement`; `golem/assembly/compile.py::_compile_element` realizes vasculature, evaluates the geometry field into a mesh, derives conduits and records, and checks vocabulary. Coupled physical stages then descend from those compiled elements. A direct reuse must enter this owner as an already-proved local section and remain subject to all assembly compatibility checks.

`golem/kernel/anatomy/realize/allocation.py::realize_vasculature` owns allocation of terminal pairs and immutable realization results. `golem/kernel/anatomy/realize/lineage.py` owns content-addressed lineage and circuit solve scheduling. `golem/kernel/anatomy/realize/cco.py`, `bifurcation.py`, `clearance.py`, `geometry.py`, and `parity.py` own constructive constrained optimization, candidate bifurcations, clearance certification, geometry, and parity. The initial profile shows repeated candidate/capsule comparisons within this cover.

`golem/kernel/engine/compile.py::evaluate_checked` is the surface-formation authority. The legacy path creates a global three-dimensional signed-distance field, samples local part fields, glues each local section with `_compose_sampled_region`, extracts an isosurface with marching cubes, and optionally solves the skin layer. Its observable result is `EvaluatedMorphology`; downstream records are derived views.

An asymptotic bottleneck means work whose growth rate is unnecessarily worse than the semantic input requires, such as recomputing the same local section, comparing every candidate with irrelevant geometry, or allocating an entire resolution-cubed temporary for a bounded local rewrite. Constant-factor changes are retained only when measured and semantic-preserving, but the investigation prioritizes growth-rate reductions.

## Plan of Work

First, freeze semantic and performance evidence. Save the canonical command output, hashes of any derived immutable arrays or serialized receipts used by focused tests, three or more cold wall-time and maximum-resident-set measurements, a `cProfile` call graph, and scale measurements that vary terminal-pair count, part count, and grid resolution independently. Use existing constructors and test fixtures; do not create a parallel input corpus when a tracked spec or focused existing fixture already owns the workload.

Second, inspect the session-to-assembly overlap. Preserve assembly snapshot digesting, source-dependent appearance seeding, mount application, surface formation, integrity checks, fit, and coupled stages. Reuse the already realized vasculature only through its existing semantic cache owner after proving that the session and assembly graphs descend to the same consumed vascular section. The result is one uncached vascular realization per authoring evaluation while assembly retains its source-capture authority.

Third, measure vascular growth independently. Derive scale curves from the existing `realize_vasculature` owner and inspect candidate generation, capsule clearance, segment distance, repeated packing, and balance calibration. Replace repeated scans with immutable indexed sections only when the index has a stable domain law and multiple concrete call sites, or fuse repeated reductions directly when no abstraction is earned. Preserve deterministic candidate order, tie-breaking, exact topology, radii, residuals, balance, typed exhaustion, and cache keys.

Fourth, measure surface-formation growth independently. Inspect `_sampled_part_field_uncached`, `_compose_sampled_region`, local-region bounds, field gluing, and skin formation for whole-grid temporaries, repeated decoding, repeated coordinate construction, and repeated evaluation of unchanged sections. Retain only transformations that keep the evaluated field, vertices, faces, normals, evidence, and downstream records byte-identical where current tests demand exactness; otherwise require the existing declared tolerance and receipt authority.

Fifth, rerun the profile after each retained change. Reject attractive rewrites whose wall time, allocation, locality, or semantics regress. Update this plan's progress, discoveries, decisions, concrete transcript, and outcome each time the chosen path changes.

## Concrete Steps

All commands run from:

    cd /Users/bluerose/Developer/pale-meridian/potentialimprovements/golem-kernel

The primary cold end-to-end measurement is:

    GOLEM_CIRCUIT_CACHE_COLD=1 /usr/bin/time -lp \
      .venv/bin/python -m golem check specs/vigil_hound_demo.json

The initial short transcript is:

    PROPRIO vigil-hound-demo txn:1 | 41 parts (15 mirrored, 56 instances) | no mesh
    VASCULAR feasibility: ACCEPTED
      vascular nodes=1080 edges=1336 terminal_pairs=256 residual=3.945e-13 balance=3.018e-13
    ASSEMBLY acceptance: ACCEPTED
    real 11.32
    maximum resident set size 2068283392

The primary profile command is:

    GOLEM_CIRCUIT_CACHE_COLD=1 /usr/bin/time -lp \
      .venv/bin/python -m cProfile -o /tmp/golem-vigil.prof \
      -m golem check specs/vigil_hound_demo.json

Inspect it with:

    .venv/bin/python -c \
      'import pstats; pstats.Stats("/tmp/golem-vigil.prof").strip_dirs().sort_stats("cumulative").print_stats(80)'

Run focused tests named by each changed owner before broad tests. Then run:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -m "not slow" tests
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest

The completed validation transcript is:

    46 passed, 6 deselected  tests/test_profile.py tests/test_engine.py -m "not slow"
    6 passed, 46 deselected  tests/test_profile.py tests/test_engine.py -m slow
    38 passed                tests/test_vascular_relaxation.py tests/test_anatomy.py -m "not slow"
    90 passed                acceptance, CLI, session protocol, and authoring-surface tests
    1031 passed, 13 deselected  tests -m "not slow"
    25 passed, 3 xfailed     sealed conformance

## Validation and Acceptance

Acceptance requires all of the following.

The primary cold command must exit zero and its stdout must be byte-identical to the saved baseline. It must still report 41 parts, 15 mirrored parts, 56 instances, the same anomalies and carrier suggestions, four closed circuits, 1080 vascular nodes, 1336 vascular edges, 256 terminal pairs, residual `3.945e-13`, balance `3.018e-13`, the same allocation map, and `ASSEMBLY acceptance: ACCEPTED`.

Focused geometry tests must prove identical evaluated fields or the repository's existing stronger declared parity. Focused vascular tests must prove identical graph topology, deterministic node and edge ordering, radii, receipts, typed failures, and cache-lineage behavior. Session and acceptance-oracle tests must prove that direct checks and submitted transactions accept and reject the same assembly surface.

Repeated cold end-to-end measurements must report their individual samples and median. Maximum resident set size must not regress. Profiles must show the removed calls or reduced growth in the exact owners; a faster unrelated stage is insufficient. Scale curves must demonstrate that the selected asymptotic work has changed, not merely that one specimen happened to run during a favorable thermal interval.

The living non-slow suite and sealed conformance suite must pass. Slow owner-specific tests touched by the change must also pass. Any broader pre-existing failure must be separated with a baseline comparison rather than compiled-chased into unrelated dirty work.

## Idempotence and Recovery

All measurements use tracked inputs and write profiles or captured output under `/tmp`; they can be repeated safely. Persisted circuit and field caches are bypassed rather than deleted, so user cache state is preserved. Existing dirty files `rehearsal/formation-trial/verdict-06.json`, `rehearsal/formation-trial/verdict-07.json`, `rehearsal/formation-trial/verdict-08.json`, and `specs/sable_stride.json` are outside this plan and must not be edited.

If a candidate changes any semantic hash, deterministic ordering, typed verdict, or declared tolerance, revert that candidate rather than adding a compatibility shim. If a benchmark is thermally noisy, preserve every sample and compare medians from alternating baseline/candidate invocations instead of selecting the flattering run.

## Artifacts and Notes

The initial profile's dominant cumulative entries were:

    14.421 s  evaluate_authoring_surfaces
     9.817 s  compile_assembly
     8.842 s  two realize_vasculature calls
     5.439 s  vascular constructive tree path
     4.884 s  three evaluate calls, one dominant assembly body surface
     4.604 s  _compiled_evidence, almost entirely the first vascular realization
     4.015 s  terminal candidate evaluation
     2.778 s  58 sampled local part fields
     1.787 s  58 sampled-region compositions
     1.713 s  13,766 capsule-set clearance checks

The two small additional `evaluate` calls derive eye records and are not the dominant surface cost.

The retained low-contention profile records:

     8.841 s  evaluate_authoring_surfaces
     4.492 s  compile_assembly
     4.347 s  two public realize_vasculature calls, one uncached computation
     3.930 s  three evaluate calls
     2.751 s  vascular constructive tree path
     2.460 s  sampled local part fields
     1.345 s  one balance calibration
     1.258 s  sampled-region compositions

The default-runtime cache path was also measured, rather than worshipping only the cold torture path. After the exact caches were populated, the final two samples were 2.95 and 2.90 seconds with 1.49-1.50 GB peak RSS and byte-identical output. A clean HEAD export under the same runtime measured 6.24 and 3.16 seconds with stable CPU time of 3.06 seconds and 1.51 GB peak RSS; the 6.24-second wall sample was externally preempted, while CPU time exposes the retained improvement.

## Interfaces and Dependencies

Use the pinned Python 3.12.12 environment in `.venv` and the existing NumPy, SciPy, scikit-image, and trimesh dependencies. Do not add a dependency unless the existing numeric stack cannot express a measured retained transformation.

The authoritative public result types remain `golem.session.state.CompileOutcome`, `golem.session.protocol.AuthoringEvaluation`, `golem.assembly.carriers.AssemblyResult`, `golem.kernel.anatomy.VasculatureResult`, and `golem.kernel.engine.types.EvaluatedMorphology | RejectedSurfaceFormation`.

Any strengthened internal boundary must accept immutable, typed carriers already owned by these modules. It must not accept an untyped dictionary claiming to be cached evidence, must not add a service locator, and must not make a generated mesh or rendered receipt authoritative over the compiled body, vascular graph, or evaluated morphology.

Revision note, 2026-07-23: created the living plan after the initial live baseline and profile established the session/assembly overlap, vascular constructor, and volumetric field composer as the dominant covers.

Revision note, 2026-07-23: closed the execution record with retained and rejected transformations, independent scale curves, exact semantic receipts, repeated cold and default-runtime timings, complete test receipts, closure audit, and explicit remaining growth laws.
