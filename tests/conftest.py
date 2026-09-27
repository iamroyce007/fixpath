import os

# Fast, download-free embeddings and synchronous warm-up for the test suite.
os.environ.setdefault("EMBED_MODEL", "hash")
os.environ.setdefault("WARM_SYNC", "1")
os.environ.setdefault("LLM_PROVIDER", "none")

import pytest  # noqa: E402

from app.kit import load_catalog, load_siis_rows  # noqa: E402


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def rows():
    return {r.id: r for r in load_siis_rows()}
