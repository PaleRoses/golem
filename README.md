# GOLEM — CSS for agents

GOLEM is a creature-construction kernel: you declare **relations**, the engine
solves **all geometry deterministically**, and anything it cannot realize comes
back as a **typed obstruction** — never a crash, never a silent deformation.

An author — human or LLM agent — never writes a load-bearing coordinate. You
say *attach the forearm at the landmark `upper_arm/wrist`*, *mirror this limb*,
*keep the eye above the muzzle line*, *run this muscle from origin to
insertion*. The compiler derives the rest: poses, flesh, musculature, eyes,
vasculature, assembly. When a design is infeasible, the answer is a named
obstruction with witness detail — which address, which margin, required versus
observed — so the next edit is obvious. An obstruction is the engine handing
the pen back with measurements attached. It is how the system tells you what
it needs to say yes.

The thesis, proven across the rehearsal record in this repo: *the model is the
constant; the tool surface is the variable.* The same model produced a
shapeless mitt in one substrate and a pack-passing hand in another.
Observability, not intelligence, was the bottleneck. GOLEM is the surface
built from that lesson.

## Five minutes

From this directory:

```sh
# Sense a creature without meshing: part census, assertions, anomalies,
# circulation — the full proprioceptive receipt.
uv run python -m golem check specs/vigil_hound_demo.json

# Compile and render orthographic PNGs (front/side/top) into ./out.
uv run python -m golem look specs/vigil_hound_demo.json --out out

# Author a new creature from a brief, iterated through typed check/compile
# rounds until it closes. Requires the Codex CLI (`codex exec`) on PATH —
# it is the authoring channel (default model gpt-5.6-sol at xhigh effort).
uv run python -m golem forge "a stocky obsidian hound with a keeled chest" \
    --out specs/my_hound.json
```

`check` is the loop you will live in: it compiles the spec, runs the
assertion pack, and reports what the body would be — components, fusions,
clearances, circulation — before any mesh exists.

The authoring surface itself is self-describing:

```sh
uv run python -m golem contract            # section index
uv run python -m golem contract relations  # the relation vocabulary, verbatim
```

## Architecture

```
golem/
  kernel/       The compilers and solvers.
    engine/     Part-graph substrate: typed parts → SDF field → mesh,
                with coherence metrics (components, watertightness, dust).
    body/       The body-plan compiler: skeleton + relations solve (attach at
                named sites, above/below, mirror_of, perpendicular_to,
                between) → deterministic poses → flesh (gencyl, blob, box,
                loft), myology (muscles from origin/insertion selectors),
                eyes, and the compiled intent block.
    anatomy/    Vascular descent and tissue envelopes over the compiled body.
    mechanics/  Fit, load, and structural cost over realized carriers.
    sheaf/      Field → section restriction and gluing.
    blame/      Minimal-blame (WHY): approximate MUS over authored
                declarations — which authored units own a rejection.
  assembly/     Scene assembly: composes compiled elements into one verdict —
                AcceptedAssembly with export artifacts (GLB + receipts), or
                RejectedAssembly with typed obstructions.
  contract/     The live authoring contract rendered by `golem contract`:
                primer, relations, vocabulary, exemplar, receipts, envelope,
                schema. The same text forge hands to the authoring model.
  cli/          The command surface: check, compile, contract, eval, forge,
                look, pose-prior.
  session/      The authoring session: a typed edit-op algebra with inverses
                (set/unset/add/remove/rename), a journal with undo/redo,
                branch/compare/merge, outline views, and a REPL
                (`uv run python -m golem.session.repl`). Edits are deltas;
                the session re-derives the world from the journal.
```

Supporting planes: `senses/` (the perception plane — proprio, symmetry,
silhouette, anomaly classification), `materials/` (appearance palette baked
into GLB exports), `plates/` (surface cell complexes), `conduits/` (surface
conduit authoring), `contracts/` (postural assertion packs), `evals/` (sealed
executable instruments), `addressing/` (the anchor/scope/cell grammars
authored names resolve through).

`specs/` holds authored creatures (the vigil hound is the flagship);
`golden/` holds write-once golden texts and frozen fixtures; `tests/` is the
living pytest suite.

## Performance

No uncited numbers here — the repo's rule is that a performance claim needs a
checked-in receipt, and the headline figures currently in circulation have
none. What the tree does show, structurally:

- `check` never meshes. Sensing runs on compiled primitives, so the authoring
  loop's cost is the solve, not the render.
- The session layer keeps the last compile per branch and re-derives only
  what an edit touches; the journal is the source of truth, everything else
  is a view.
- The vascular lane pins a content-addressed compile cache — repeated calls
  return the same immutable object and receipt. The last measured receipts
  for that lane live in `rehearsal/vascular/README.md`; treat them as
  historical, since the canonical element set has changed since they were
  taken.

## Tests

```sh
uv run pytest                 # sealed conformance suite over the frozen oracle
uv run pytest tests           # the living suite (-m "not slow" for the fast subset)
```

`pilots/`, `outputs/`, `conformance/`, and `phase-minus-one/` are the frozen
regression oracle: reference artifacts that are only ever run, never edited.
`rehearsal/` is experiment evidence — records, not code.

## Appendix: the design document

[GOLEM_KERNEL_SPEC.md](GOLEM_KERNEL_SPEC.md) is the design specification the
system was built from — RFC-style axioms, rules, planes, and the lessons
register each rule traces to. It is a design document, not the user manual:
the commands above are the system.
