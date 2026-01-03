import pytest

# If torch isn't installed in the environment (e.g., local quick runs), skip tests that need it.
# CI installs torch from requirements.txt so tests will run there.
pytest.importorskip("torch")
