"""Shared infrastructure for the msda-triton labs.

- ``setup``: environment detection; must be imported before triton so the
  CPU interpreter fallback (TRITON_INTERPRET=1) takes effect.
- ``checks``: the labs' "supplied test scripts" — assertion helpers built on
  the repo's reference implementations, plus marimo rendering.
- ``memsim``: scaffold for Lab 3's paper-GPU memory-traffic simulator.
- ``loader``: NAND2Tetris-style "builtin chip" fallback — loads a stage
  artifact from labs/my/ (yours) or labs/solutions/ (reference).
"""
