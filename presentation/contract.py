"""THE presentation contract -- every parameter of the judged view, pinned.

Both backends (``blender_contract.py``, ``fallback.py``) read this dict and
nothing else for scene parameters. Changing a value here changes the judged
sense for every future render and belongs in the plan's Decision Log.

Per plan D3: presentation is evaluation infrastructure outside the sealed
environment; it consumes exchange artifacts (GLB + materials sidecar) only
and imports nothing from ``golem``.
"""

CONTRACT = {
    "version": "first-light-1",
    "size": 1024,
    "views_azim_deg": (0, 40, 90, 180),
    "hero_views_azim_deg": (15, 55),
    "elev_deg": 12.0,
    "turntable_frames": 36,
    "turntable_size": 512,
    "key_dir": (0.35, 0.65, 0.85),
    "fill_dir": (-0.7, 0.25, 0.45),
    "key_energy": 1.05,
    "fill_energy": 0.30,
    "ambient": 0.05,
    "rim_color": (0.66, 0.33, 0.97),
    "rim_k": 0.30,
    "bg_top": (34, 34, 44),
    "bg_bottom": (10, 10, 15),
    "blender": {
        "min_version": "4.0",
        "engine": "CYCLES",
        "samples": 64,
        "view_transform": "Filmic",
        "camera_fit_margin": 1.16,
    },
}
