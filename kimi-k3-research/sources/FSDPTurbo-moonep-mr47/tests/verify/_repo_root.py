"""Import ``fsdp_turbo`` from THIS checkout, not from a pip-installed one.

``python3 tests/verify/<script>.py`` puts ``tests/verify`` on ``sys.path[0]``,
not the repository root, so ``import fsdp_turbo`` resolves to whatever
``pip install -e`` registered. When that points at a different checkout, a
script runs one tree's stand-ins against another tree's adapter and fails with
a confusing ``AttributeError`` instead of naming the mismatch.
"""

from __future__ import annotations

import os
import sys

_VERIFY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_VERIFY_DIR))


def use_repo_checkout() -> str:
    """Put this checkout ahead of site-packages, then report what got loaded."""
    for path in (_VERIFY_DIR, REPO_ROOT):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import fsdp_turbo

    loaded = os.path.realpath(os.path.dirname(fsdp_turbo.__file__))
    expected = os.path.realpath(os.path.join(REPO_ROOT, "fsdp_turbo"))
    if loaded != expected:
        raise RuntimeError(
            "fsdp_turbo was imported from\n"
            f"    {loaded}\n"
            "but this script belongs to\n"
            f"    {expected}\n"
            "An editable install is shadowing this checkout, so the scripts "
            "would check this tree's expectations against another tree's code. "
            "Apply the patches to the shadowing tree, or reinstall with "
            "`pip install -e .` from this one."
        )
    return loaded
