def test_src_package_importable():
    import src
    assert src is not None


def test_project_submodules_importable():
    import src.data
    import src.features
    import src.models
    import src.evaluation
    import src.pipelines

    assert True