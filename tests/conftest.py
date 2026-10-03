import sys
from pathlib import Path

import pytest

# Add parent directory to path so tests can import modules at root level
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def isolated_registry():
    """Remove components registered during a test; keep built-ins (they are registered once at import)."""
    yield
    from ocr import registry

    for kind, names in registry._REGISTRY.items():
        for name in list(names):
            if (kind, name) not in registry._BUILTIN_MODULES:
                del names[name]
