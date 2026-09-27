import pytest

from app.kit import load_catalog, load_siis_rows


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def rows():
    return {r.id: r for r in load_siis_rows()}
