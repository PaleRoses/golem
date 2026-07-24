"""Transactional authored-body sessions with a canonical public protocol.

``protocol`` is the public authoring boundary. ``ops`` owns its typed edit
grammar, while ``journal`` and ``algebra`` own immutable state transitions.
Whole documents descend through ``diff`` into the same journal before protocol
evidence is glued from the existing compiler owners.
"""
