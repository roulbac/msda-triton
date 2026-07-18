"""NAND2Tetris-style "builtin chips": load a previous lab's artifact.

Each lab ends by telling you to copy your working functions into
``labs/my/labNN.py``. Later labs fetch prior stages through :func:`stage`,
which prefers *your* implementation and silently falls back to the shipped
reference solution in ``labs/solutions/labNN.py`` — so a broken (or skipped)
stage never blocks the next lab, exactly like nand2tetris's builtin chips.
"""

import importlib

_KIND_MY = "yours (labs/my)"
_KIND_REF = "reference (labs/solutions)"


def stage(lab: int, name: str):
    """Return ``(artifact, source)`` for artifact ``name`` of lab ``lab``.

    ``source`` says whether you got your own implementation or the shipped
    reference — labs display it, so there's never doubt about whose code is
    running underneath you.
    """
    for module, kind in (
        (f"labs.my.lab{lab:02d}", _KIND_MY),
        (f"labs.solutions.lab{lab:02d}", _KIND_REF),
    ):
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        obj = getattr(mod, name, None)
        if obj is not None:
            return obj, kind
    raise ImportError(
        f"no implementation of {name!r} found for lab {lab:02d} in labs/my "
        f"or labs/solutions — is the repo checkout intact?"
    )


def stage_banner(mo, used: dict):
    """Render which stages this lab loaded, and from where.

    ``used`` maps artifact name -> source string returned by :func:`stage`.
    """
    lines = [f"- `{name}` — {kind}" for name, kind in used.items()]
    return mo.md(
        "**Building on previous stages:**\n\n" + "\n".join(lines) +
        "\n\n*(To use your own version, copy it into `labs/my/labNN.py` — "
        "see the epilogue of the corresponding lab.)*"
    ).callout(kind="info")
