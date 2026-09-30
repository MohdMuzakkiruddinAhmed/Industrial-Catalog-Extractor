from examples.create_sample_catalog import create_catalog


def test_synthetic_catalog_is_byte_stable(tmp_path) -> None:
    first = create_catalog(tmp_path / "first.pdf")
    second = create_catalog(tmp_path / "second.pdf")

    assert first.read_bytes() == second.read_bytes()
