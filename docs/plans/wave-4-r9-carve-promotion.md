# Promote authored carves into the body/0.3 vocabulary

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept current while the work proceeds. It is maintained in accordance with `../../docs/llm-guidance/PLANS.md` from the pale-meridian repository root.

## Purpose / Big Picture

After this change, a body/0.3 author can declare a cavity on a skeleton bone using the same bone-local placement and inherited sagittal mirroring model as flesh. The body compiler emits that declaration into the existing `GeometryGraph.carves` channel used by eye sockets, so the engine still owns exactly one subtraction pipeline. A close-lipped articulated jaw can therefore retain a real oral void without senses reporting an accidental bridge, while solid integrity and vascular acceptance continue to judge the carved morphology rather than the uncut positive parts.

The behavior is visible by compiling a body with a bone-local oral carve and observing a `carves` entry in the compiled graph, a positive signed-distance value at the cavity center, no mouth-specific false anomaly, one watertight solid, and a typed rejection when a cavity destroys a vascular carrier or exchange bed. A spec with no authored carves must produce byte-for-byte the same compiled product as before.

## Progress

- [x] (2026-07-23T00:42Z) Read ruling R9, the eye socket carve compiler, the body compiler merge point, canonical bone projection, senses anomaly construction, and vascular acceptance paths.
- [x] (2026-07-23T00:42Z) Verified every existing symbol and test target named by this plan against source; new symbols and `tests/test_carves.py` are explicitly creation targets.
- [x] (2026-07-23T00:58Z) Defined the strict bone-local carve vocabulary: closed `CarveKind` enum, typed obstructions, and `golem/kernel/body/carves.py::compile_carves`. Every malformation names the legal set.
- [x] (2026-07-23T00:58Z) Threaded `carves` through `records.py::project_bone_record` and merged authored + eye socket carves once at `Compiler._emit_body` under the single `graph["carves"]` key.
- [x] (2026-07-23T00:58Z) Made senses distinguish an authored cavity from a false bridge via `anomaly.py::_carve_severs_bridge`, consulting the compiler's own carve field; off-corridor carves do not over-suppress.
- [x] (2026-07-23T00:58Z) Proved authored cavities mesh to one watertight component and a carrier-gutting carve rejects with `ChannelInducedDisconnectionObstruction` (pinned in `tests/test_anatomy.py`, passing before any production edit — anatomy left unchanged).
- [x] (2026-07-23T00:58Z) Proved absent and explicit-empty carve declarations preserve compiled bytes (`test_absent_and_explicit_empty_carves_preserve_body_product_bytes`; existing `test_body`/`test_muscle` hash pins remain green).
- [x] (2026-07-23T00:58Z) Ran touched pytest files: `tests/test_carves.py` (20 passed) plus `tests/test_anatomy.py test_eyes.py test_body.py test_muscle.py test_proprio.py test_senses_model.py` (151 passed), and a downstream sweep `test_material_classes test_acceptance_oracle test_session_protocol test_assembly test_quadruped_prior test_schematic test_engine` (127 passed).
- [x] (2026-07-23T00:58Z) Appended friction entry A12 to `rehearsal/verdigris-trial/FRICTION.md`.

## Surprises & Discoveries

- Observation: Eye sockets already compile as ordinary `blob` parts in `AcceptedEyes.carves`; `Compiler._emit_body` is the only body-side merge into `graph["carves"]`. Engine decode, mirroring, field sampling, meshing, and assembly mounting already consume that channel.
  Evidence: `golem/kernel/body/eyes.py::_place_eye`, `golem/kernel/body/compile.py::Compiler._emit_body`, and `golem/kernel/engine/algebra.py::carve_part_instances`.

- Observation: Canonical bone projection deliberately rejects unthreaded authored keys. A bone-level `carves` field must be registered once in `golem/kernel/body/records.py` or compilation fails loudly.
  Evidence: `project_bone_record` raises `UnthreadedBoneFieldError` for keys outside `KNOWN_BONE_KEYS`.

- Observation: Proprioception derives instances and pair gaps from positive `graph["parts"]`; carves are preserved only through `Senses.graph`. A mouth carve can therefore answer the existing advisory “may bridge at mesh” only if anomaly evaluation consults the authoritative carved field.
  Evidence: `golem/senses/proprio/build.py::_instance_list` and `golem/senses/proprio/anomaly.py::_gap_anomalies`.

- Observation: The session analytic stage accepts `realize_vasculature` directly, while assembly material masking later checks the evaluated carved field. The negative test must determine whether the user-visible acceptance path already returns a typed obstruction before anatomy containment is changed.
  Evidence: `golem/session/protocol.py` calls `realize_vasculature(accepted, outcome.graph)`; `golem/kernel/anatomy/material/carve.py` rejects lumen cells outside `evaluated.field <= 0` as `VascularMaterialEscapeObstruction`.

- Observation: The existing vascular material boundary already rejects a carrier-gutting authored carve with `ChannelInducedDisconnectionObstruction`; no anatomy production change is required.
  Evidence: `.venv/bin/python -m pytest tests/test_anatomy.py -q -k authored_cavity_gutting_carrier` passed before any production edit. The test uses `engine.evaluate` and `derive_vascular_material_masks`, whose `load_bearing_solid` component check observes the authoritative carved field.

## Decision Log

- Decision: Authored carves live on skeleton bone records rather than in a new top-level subsystem.
  Rationale: Flesh is bone-local, inherited bone mirroring is the existing body/0.3 mirror vocabulary, and the canonical record projection then guarantees the declaration reaches both FK consumers without a second host-frame mechanism.
  Date/Author: 2026-07-23, Kimi.

- Decision: Authored and eye socket carves are concatenated once at `Compiler._emit_body` and emitted under the single existing `graph["carves"]` key.
  Rationale: R9 promotes vocabulary, not geometry evaluation. A parallel subtraction field would duplicate the engine law and violate the ruling.
  Date/Author: 2026-07-23, Kimi.

- Decision: The legal carve kinds are a closed enum derived from the primitive kinds accepted by the existing engine carve path; no string fallback is permitted.
  Rationale: R4 requires diagnostics to name the legal vocabulary, and R9 explicitly limits the initial surface to already-evaluated shapes.
  Date/Author: 2026-07-23, Kimi.

- Decision: Write and run the vascular negative test before changing anatomy containment.
  Rationale: Assembly material masks already compare vascular lumen against the authoritative carved field. If the acceptance surface already returns a typed rejection, production containment changes would be duplicate ceremony outside this lane’s stated territory.
  Date/Author: 2026-07-23, Kimi.

- Decision: Keep anatomy containment unchanged and pin the existing material-mask rejection.
  Rationale: The requested negative behavior already holds through the canonical evaluated field. Reimplementing subtraction in anatomy realization would create a second field owner and cross this lane's body/senses territory without changing the observable rejection.
  Date/Author: 2026-07-23, Kimi.

## Outcomes & Retrospective

The promotion landed exactly as R9 designed: one authored vocabulary riding the one subtraction pipeline. A body/0.3 author can now declare `carves` on any skeleton bone in bone-local coordinates with the inherited mirror vocabulary, and the compiler emits world-space engine primitives into the same `graph["carves"]` channel the eye sockets have always used — verified by a shared channel test asserting the tuple `("mouth.slot", "eye.watcher.socket")`.

The two law claims that the dispatch expected to need new machinery turned out to already hold in the field, and both were proven rather than assumed. The vascular carrier-gutting carve rejected with `ChannelInducedDisconnectionObstruction` before any production edit, because the material-mask boundary reads the authoritative carved morphology; anatomy containment was deliberately left unchanged (its carve-blindness is a real, separate ownership wound recorded in friction A12, not silently patched here). The mouth law needed one genuine senses change — `_carve_severs_bridge` — because anomaly detection measured pair gaps from positive parts only and therefore lied about a carved oral void; the fix consults the compiler's own carve field, never a second SDF.

Lesson: the expensive part of a vocabulary promotion is proving which laws were already total. The geometry was never the gap; the vocabulary and the one field-blind advisory were.

## Context and Orientation

A body/0.3 document describes a skeleton whose root and child bone records carry `flesh` arrays. `golem/kernel/body/kinematics.py` and `golem/kernel/body/geometry.py` both construct compiled bone records through `golem/kernel/body/records.py::project_bone_record`. `golem/kernel/body/emit.py::_EmitMixin` converts bone-local flesh declarations into world-space engine parts. `golem/kernel/body/eyes.py` independently decodes eye declarations after forward kinematics and emits each eye socket as a world-space carve dictionary. `golem/kernel/body/compile.py::Compiler._emit_body` currently inserts only those eye socket dictionaries into the graph.

A carve is a negative primitive: the engine first unions all positive parts, then computes the hard signed-distance difference `max(host, -carve)` for each mirrored carve instance. The legal engine primitive kinds are generalized cylinders (`gencyl`), ellipsoidal blobs (`blob`), and rounded boxes (`box`). Body-local placement converts a declaration attached to a bone into one of those world-space dictionaries. A mirrored bone produces the authored instance and its exact reflection through the existing `mirror: true` marker, exactly as flesh does.

Senses are immutable observations built by `golem/senses/proprio/build.py::build_senses`. Pair anomalies are classified in `golem/senses/proprio/anomaly.py`. The classifier currently estimates possible smooth-union bridges from positive part gaps. Because the compiled graph retained on `Senses.graph` includes carves, the classifier can consult the authoritative final field at the closest approach instead of inventing a separate cavity registry or SDF implementation.

Vascular realization receives the compiled body graph. Assembly material masks use the final evaluated morphology, including carves, and expose typed failures from `golem.kernel.anatomy.material.types`. The implementation must first pin that observable behavior. Only if the test proves silent acceptance may carve containment be threaded deeper, and that expansion must remain limited to the established anatomy field seam rather than changing engine code.

## Plan of Work

Create a body-side carve decoder following the accepted/rejected algebra used by eyes and material classes. Define one `CarveKind` closed enum and typed obstruction records in `golem/kernel/body/types.py`. The decoder must reject a non-array carve section, non-object entries, unknown fields, unknown kinds, malformed finite vectors and numbers, illegal rounded-box radii, malformed generalized-cylinder stations/radii, duplicate emitted ids, and collisions with existing positive or eye-derived ids. Every unknown-kind obstruction must carry the complete legal kind tuple. Accepted values are normalized once into immutable typed records.

Register `carves` in `golem/kernel/body/records.py` as a projected bone field. Add body emission code that converts each accepted bone-local carve into the same world-space primitive dictionaries already accepted under `GeometryGraph.carves`. Centered blob and box declarations use `placed_center`; box/blob orientation uses the bone rotation quaternion; generalized cylinders use the bone axis and authored stations. Mirroring is inherited from the compiled bone record exactly as flesh mirroring is. Carves do not enter positive part provenance or anatomy carrier classification.

In `golem/kernel/body/compile.py::Compiler._emit_body`, compile authored carves after positive parts exist so id collisions can be diagnosed, then concatenate authored carve dictionaries and `AcceptedEyes.carves`. Emit the `carves` key only when the concatenated tuple is non-empty. Do not alter graph key order or any field emitted for carve-free specifications.

For senses, add only the minimum field-aware decision required to prevent a close lip/jaw pair from receiving a false “may bridge at mesh” anomaly when the existing carve subtraction opens the corridor. Reuse `Senses.graph` and engine field sampling; do not implement carve SDFs in senses. Pin the behavior with a paired test: the positive-only graph reports `blend_ambiguity`, while the same graph with an oral carve does not. All other anomaly branches and assertions remain unchanged.

Add tests in a focused carve test module or the nearest existing body/senses files. The tests must cover strict legal-kind diagnostics, bone mirroring, merge with eye sockets, cavity field sign, absent versus explicit-empty byte identity, mouth anomaly suppression, watertight solid integrity, and vascular carrier destruction. The vascular test must run first against current production behavior and assert a concrete typed obstruction. If it already passes at the material-mask boundary, do not modify anatomy containment; record the diagnostic distance as friction. If it silently accepts, extend only the existing anatomy containment field to subtract `body_graph["carves"]`, with mirrored instances and the same `max(host, -carve)` law, then retain the test as the proof.

Finally run only the touched test files with `.venv/bin/python -m pytest`, update this plan with exact results, and append a concise wave-3 amendment friction entry describing any non-obvious authoring or law-threading wound.

## Concrete Steps

From `potentialimprovements/golem-kernel`:

    .venv/bin/python -m pytest tests/test_carves.py -q

Run the vascular negative test alone before any anatomy change:

    .venv/bin/python -m pytest tests/test_carves.py -q -k vascular

Run any touched senses file directly if the anomaly test remains there:

    .venv/bin/python -m pytest tests/test_proprio.py -q

After implementation, run every touched test file in one command and expect zero failures. Record the exact command and count in `Progress` and `Artifacts and Notes`.

## Validation and Acceptance

A valid bone-local blob, box, or generalized-cylinder carve compiles to one entry under `CompiledBody.graph["carves"]`. A carve on a mirrored bone retains `mirror: true` and engine instancing yields exact sagittal reflection. An invalid kind returns a typed body obstruction whose `required` or equivalent field is exactly the complete legal kind tuple.

At the authored cavity center, `golem.kernel.engine.sample_graph_field` returns a positive value while the same positive-only graph returns a negative value. Mesh coherence at the test resolution reports one component and `watertight_main is True`. A mouth graph that reports `blend_ambiguity` without the carve reports no corresponding anomaly after the carve opens the bridge corridor. A carrier- or exchange-bed-gutting carve produces a named typed rejection rather than `AcceptedVasculature` or an accepted assembly. Removing all authored carve declarations restores the exact pre-change JSON bytes for the compiled graph, receipt, anatomy projection, and eye globe projection.

No existing assertion may be weakened, and existing eye socket tests must remain green.

## Idempotence and Recovery

All compiler changes are pure and deterministic. Re-running compilation or tests does not mutate specifications. The vascular test is observational before any containment edit; if it already passes, no production anatomy code should change. If a surgical edit fails because a file changed concurrently, re-read the affected range and reapply only the intended hunk. Do not use destructive git commands or branch changes.

## Artifacts and Notes

Cavity field sign at the authored mouth.slot center (single skull blob plus a shallow box carve):

    baseline field: -0.0212
    carved field:    0.025

Watertight integrity over the authored cavity (`engine.evaluate` res=96, `coherence_report`):

    components=1  watertight_main=True  largest_component_volume_share=1.0

Vascular carrier-gutting negative (off-centre slab carve, `derive_vascular_material_masks`):

    RejectedVascularMaterial((ChannelInducedDisconnectionObstruction(component_count>1),))

Mirror atomism (mirrored cheek blob carve, authored +x center and sagittal reflection):

    center    baseline/carved: -0.06 / 0.025
    reflected baseline/carved: -0.06 / 0.025

Touched test files, run with `.venv/bin/python -m pytest`:

    tests/test_carves.py -> 20 passed
    tests/test_anatomy.py tests/test_eyes.py tests/test_body.py tests/test_muscle.py
      tests/test_proprio.py tests/test_senses_model.py -> 151 passed
    tests/test_material_classes.py tests/test_acceptance_oracle.py
      tests/test_session_protocol.py tests/test_assembly.py
      tests/test_quadruped_prior.py tests/test_schematic.py tests/test_engine.py -> 127 passed

Friction entry appended: `rehearsal/verdigris-trial/FRICTION.md` A12.

## Interfaces and Dependencies

`golem/kernel/body/types.py` must expose one closed carve-kind enum, immutable accepted carve records, a rejected result carrying typed obstructions, and obstruction rules that preserve the authored address, value, and legal requirement.

A body-side compiler function must accept compiled bone records plus occupied positive and eye-derived ids and return either accepted world-space carve dictionaries with provenance or typed obstructions. Its output dictionaries must be valid existing engine `Part` payloads; it must not call or modify engine subtraction code.

`golem/kernel/body/records.py::project_bone_record` must carry authored `carves` exactly once. `golem/kernel/body/compile.py::Compiler._emit_body` must remain the sole body graph merge point for authored and eye socket carves.

Senses may depend on the existing public `golem.kernel.engine.sample_graph_field` wrapper. No new dependency, material API, CLI option, session format, contract prose, or engine primitive is introduced.
