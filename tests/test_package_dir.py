import datenwissenschaften


def test_dir_includes_lazily_resolved_exports():
    names = dir(datenwissenschaften)

    assert "TrainingConfig" in names
    assert "load_config" in names
    assert names == sorted(names)
