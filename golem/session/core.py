"""Public session harness over the pure algebra, coalgebra, and effects."""

from __future__ import annotations

from golem.session import algebra, coalgebra, effect, ops, outline, receipt
from golem.session.journal import Journal
from golem.session.state import AuthoredState, LoadObstructed, SessionState


class Session:
    def __init__(self, state: SessionState):
        self._model = algebra.initial(state)

    @classmethod
    def open(cls, path: str) -> "Session":
        loaded = effect.load_state(path)
        if isinstance(loaded, LoadObstructed):
            obstruction = loaded.obstructions[0]
            raise ops.Reject(str(obstruction.path), obstruction.reason)
        return cls(loaded)

    @classmethod
    def new(cls, name: str) -> "Session":
        return cls(SessionState.new(name))

    @property
    def state(self) -> SessionState:
        return algebra.current_branch(self._model).state

    @property
    def journal(self) -> Journal:
        return Journal.from_state(algebra.current_branch(self._model).journal)

    @staticmethod
    def _raise_edit_obstruction(obstructed: ops.EditObstructed) -> None:
        raise ops.Reject(obstructed.first.address, obstructed.first.reason)

    @staticmethod
    def _raise_effect_obstruction(obstructed: effect.EffectObstructed) -> None:
        obstruction = obstructed.obstructions[0]
        raise ops.Reject(obstruction.address, obstruction.reason)

    def _accept(self, pending: algebra.PendingTransition) -> str:
        branch = algebra.current_branch(self._model)
        txn = pending.entries[-1].txn if pending.entries else branch.state.txn
        authored = AuthoredState(pending.next_spec, branch.state.spec_dir, txn)
        outcome = effect.compile_authored(authored)
        self._model = algebra.accept_transition(self._model, pending, outcome)
        return receipt.compute(
            pending.previous,
            outcome,
            [entry.to_json() for entry in pending.entries],
        )

    def _interpret_program(self, program: coalgebra.ConstructionResult) -> str:
        if isinstance(program, ops.EditObstructed):
            self._raise_edit_obstruction(program)
        pending = algebra.plan_program(self._model, program.operations)
        if isinstance(pending, ops.EditObstructed):
            self._raise_edit_obstruction(pending)
        return self._accept(pending)

    def do_typed_op(self, operation: ops.Op) -> str:
        pending = algebra.plan_program(self._model, (operation,))
        if isinstance(pending, ops.EditObstructed):
            self._raise_edit_obstruction(pending)
        return self._accept(pending)

    def do_line_op(self, operation: dict) -> str:
        pending = algebra.plan_raw_edit(self._model, operation)
        if isinstance(pending, ops.EditObstructed):
            self._raise_edit_obstruction(pending)
        return self._accept(pending)

    do = do_line_op

    def undo(self) -> str:
        pending = algebra.plan_undo(self._model)
        if isinstance(pending, ops.EditObstructed):
            self._raise_edit_obstruction(pending)
        return self._accept(pending)

    def redo(self) -> str:
        pending = algebra.plan_redo(self._model)
        if isinstance(pending, ops.EditObstructed):
            self._raise_edit_obstruction(pending)
        return self._accept(pending)

    def outline(
        self,
        scope: str = "",
        depth: int = 1,
        aspect: str = "structure",
    ) -> str:
        return outline.render(self.state, scope, depth, aspect, self.journal)

    def recent(self, n: int = 5) -> str:
        entries = self.journal.recent(n)
        return (
            "\n".join(f"txn {entry['txn']}  {entry['label']}" for entry in entries)
            if entries
            else "journal empty"
        )

    def save_spec(self, path: str) -> None:
        result = effect.save_spec(self.state, path)
        if isinstance(result, effect.EffectObstructed):
            self._raise_effect_obstruction(result)

    def save_journal(self, path: str) -> None:
        branch = algebra.current_branch(self._model)
        result = effect.save_journal(
            tuple(entry.to_json() for entry in branch.journal.entries),
            path,
        )
        if isinstance(result, effect.EffectObstructed):
            self._raise_effect_obstruction(result)

    def mesh(self, res: int = 120) -> str:
        result = effect.mesh_summary(self.state, res)
        if isinstance(result, effect.EffectObstructed):
            self._raise_effect_obstruction(result)
        return result.text

    def views(self, path: str, res: int = 96) -> str:
        result = effect.render_views(self.state, path, res)
        if isinstance(result, effect.EffectObstructed):
            self._raise_effect_obstruction(result)
        return result.text

    def attach_bone(self, name: str, anchor: str, params: dict) -> str:
        frames = None
        if "world_dir" in params:
            frame_result = effect.bone_frames(self.state)
            if isinstance(frame_result, effect.EffectObstructed):
                self._raise_effect_obstruction(frame_result)
            frames = frame_result.frames
        return self._interpret_program(
            coalgebra.attach_bone(self.state.spec, frames, name, anchor, params)
        )

    def attach_flesh(self, bone: str, name: str, params: dict) -> str:
        return self._interpret_program(coalgebra.attach_flesh(bone, name, params))

    def reflect(self, bone: str) -> str:
        return self._interpret_program(coalgebra.reflect(bone))

    def array(
        self,
        bone: str,
        name: str,
        n: int,
        along: list,
        template: dict,
    ) -> str:
        return self._interpret_program(coalgebra.array(bone, name, n, along, template))

    def squeeze(
        self,
        name: str,
        anchor_a: str,
        anchor_b: str,
        params: dict | None = None,
    ) -> str:
        frame_result = effect.bone_frames(self.state)
        if isinstance(frame_result, effect.EffectObstructed):
            self._raise_effect_obstruction(frame_result)
        return self._interpret_program(
            coalgebra.squeeze(
                self.state.spec,
                frame_result.frames,
                name,
                anchor_a,
                anchor_b,
                params or {},
            )
        )

    def envelope(self, params: dict) -> str:
        return self._interpret_program(coalgebra.envelope(self.state.spec, params))

    def branch(self, purpose: str) -> str:
        result = algebra.create_branch(self._model, purpose)
        if isinstance(result, ops.EditObstructed):
            self._raise_edit_obstruction(result)
        self._model = result.model
        return result.name

    def branches(self) -> str:
        return algebra.render_branches(self._model)

    def checkout(self, name: str) -> str:
        result = algebra.checkout(self._model, name)
        if isinstance(result, ops.EditObstructed):
            self._raise_edit_obstruction(result)
        self._model = result
        return f"on {name}"

    def compare(self, left: str, right: str) -> str:
        result = algebra.compare(self._model, left, right)
        if isinstance(result, ops.EditObstructed):
            self._raise_edit_obstruction(result)
        return result

    def merge(self, name: str) -> str:
        result = algebra.plan_merge(self._model, name)
        if isinstance(result, ops.EditObstructed):
            self._raise_edit_obstruction(result)
        if isinstance(result, algebra.MergeEmpty):
            return result.message
        return self._accept(result)
