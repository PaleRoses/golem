# Golem Conduit Harness — Fresh-Operator Experience Report (probe_b)

**Subject:** "the acolyte" (hooded robed figure)
**Brief:** (1) internal circulation rooted sensibly with visible surface reinforcement; (2) a grown, mirrored crown-of-thorns appendage from the hood, using authored intent volumes.
**Result:** Both delivered. Final element saved to `final.json`; final render is `probe_final_*.png`. Crown of thorns emerges from the hood (mirrored, ~6 spikes) and branching vessel swells read clearly through the robe (vascular `circ`, coverage 0.37, rooted at `robe:0.6` ≈ chest).

**Usability score: 3 / 5** — the core declare→grow→receipt loop is fast and safe, and the vascular feature is genuinely excellent, but the appendage half of the workflow is effectively *blind*, which broke the tool's central "iterate cheap, render only to confirm" promise exactly where I needed it.

---

## What I actually did (run ledger — 7 of 8 budgeted runs, 2 renders)
1. `parts` + `show` — learn anatomy.
2. Declare thorns + vascular, `grow` — OK/watertight, but no sign of the appendage.
3. `render probe1` — **discovered the crown was completely absent.** Hood top was a smooth cylinder.
4. Fatter thorns, `count 96`, `envelope 2.2`, boosted reinforce — grow OK, coverage 0.16→0.37.
5. 4-thorn variant, `count 110` — grow OK but 16s (render-budget danger).
6. Lighter retune — **`REJECTED: min() arg is an empty sequence`** (cryptic crash; nothing grew).
7. Revert to confirmed-good run-4 config → grow + `save final.json` + `render probe_final` — success.

---

## Top friction points

### 1. Appendages get ZERO receipt feedback (the big one)
The receipt reports `element`, `verdict`, and `conduit <id>` (with faces + vascular tree), but **never says anything about appendages** — no face count, no "grew N segments," no bounding extent, not even a yes/no that structure was produced. My first crown (`count 3`, `envelope 1.8`, thin thorns) silently produced *nothing*, and the receipt looked identically healthy (`components=1 watertight=True verdict OK`) whether the appendage grew or not. I only learned it was a no-op by spending a render.

This directly defeats the advertised workflow ("Growing is cheap... Render only to confirm"). For conduits/vascular the receipt lets me tune blind-free; for appendages I was forced to render to see if anything existed at all. The HELP's "THE RECEIPT" section documents element/violation/conduit/vascular lines and is **silent on appendages** — so I couldn't even tell if the silence was a bug or by design.
**Fix:** add an `appendage <id>: faces=N grew=(bbox / n_segments)` receipt line, and a warning when an appendage resolves to zero added volume.

### 2. A raw exception surfaced as a "rejection" (`min() arg is an empty sequence`)
The HELP promises "Rejections are answers, not crashes." This one was a crash in a rejection's clothing: an un-actionable Python-internal message with no indication of which knob caused it. It appeared when I moved vascular `reinforce` from `0.008,7.0,0.35` (works) to `0.008,6.5,0.42` (crashes) — an invisible, fragile boundary. My best guess: `min_flow` filtered out *all* vessels, so the reinforce step ran `min()` over an empty set. Because I'd changed several things in one edit (forced by run budget), I couldn't localize the culprit.
**Fix:** clamp/guard the empty case with a real operator message, e.g. `reinforce matched 0 vessels — lower min_flow or increase count`.

### 3. Auto-`res` coupling makes receipt numbers non-comparable
The grid `res` auto-changes with the bounding box (I saw 187 → 193 → 191). Adding/moving the thorns silently changed the *vascular* tree (e.g. nodes 1077 vs 1836 at otherwise-similar params) and the base body retessellates, so **face counts can't be diffed across edits** to infer what a change added. That was the natural fallback I reached for to detect the invisible appendage — and it doesn't work.
**Fix:** report a per-feature face delta, or let me pin `res` so receipts are comparable.

---

## Things I reached for that didn't exist
- **A cheap "did the appendage grow / how big is it" check** — anything short of a full render. A `probe`/`bbox <appendage>` would have saved my first render.
- **Grow-time estimate or render-budget guardrail.** Grow swung 3.6s → 16s as I added appendage mass, with no warning; I had to mentally enforce "grow_seconds + ~30s render < 43s timeout." (Also: the *identical* run-4 config grew in 9.2s once and 7.1s another time, so budgeting is guesswork despite the "deterministic" framing.)
- **Documented `count` semantics for appendages.** HELP gives a range for vascular `count` (~200–300 vessels) but nothing for appendage `count`. I guessed by trial: 3 → nothing, ~60–110 → good. One sentence ("appendage count ~60–150 growth particles") would have saved a render.
- **An `undo`/"revert to last good grow."** After the crash left me with no grown state, I had to hand-reconstruct a known-good ops file from memory. The "ops file IS your session" model made this survivable, but a last-known-good marker would help.

## What the receipt failed to tell me
- Whether the appendage produced any geometry (see #1).
- Which parameter triggered the `min()` crash (see #2).
- How to read `coverage` in real terms — is 0.16 "faint" and 0.37 "clear"? I had to render twice to calibrate my eye. A one-line legend ("coverage ≈ fraction of major vessels swelling the surface; >~0.3 reads clearly") would let me hit the target from the receipt alone.
- The vascular tips report `contained by construction` (reassuring), but there's no signal for how much swell actually breaks the silhouette vs stays buried.

## What worked well (credit where due)
- **The declare → grow → receipt loop is fast, and "ops file = session, replayed from pristine" is clean and reproducible.** I never feared corrupting on-disk state, so I experimented freely.
- **`components` / `watertight` / `verdict` are excellent at-a-glance safety signals.**
- **Vascular reinforcement is the star.** The `reinforce(max, coverage)` numbers mapped intuitively to the render, and the surface swells look convincing and organic. Rooting via `<part>:<t>` stations is a nice mental model.
- **Two views + a contact sheet per render** is the right default and was immediately useful.

## Net
I hit the brief within budget, and the vascular tooling is a pleasure. The score is a 3 rather than higher because the appendage workflow is feedback-blind: the one thing the tool tells you to do cheaply (iterate on receipts) is impossible for appendages, so I spent a scarce render to discover a silent no-op, then hit an unactionable crash. Close the appendage-feedback gap and turn the raw exception into a real rejection message, and this jumps to a 4.
