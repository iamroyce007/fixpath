from app.kit import DUMMY_DEEPLINK, load_queries, load_sample_output, load_schema


def test_catalog_counts(catalog):
    assert len(catalog) == 578
    assert sum(1 for e in catalog.entries if e.get("validation")) == 570
    assert catalog.get(DUMMY_DEEPLINK)["id"] == "DL-DUMMY"
    assert catalog.get("DL-0001")["deeplink"] == "voiceassist://masked/act/aa73a35e8d"


def test_siis_rows(rows):
    assert len(rows) == 20
    assert len({(r.title, r.content) for r in rows.values()}) == 11


def test_queries_and_sample():
    assert len(load_queries()) == 20
    schema = load_schema()
    schema.ContextDeeplinkResponse.model_validate(load_sample_output()["response"])
