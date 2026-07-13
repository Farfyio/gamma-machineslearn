from gamma_machineslearn.preprocessing import create_preprocessing_pipeline



def test_create_preprocessing_pipeline_has_scaler_step():
    pipeline = create_preprocessing_pipeline()
    assert [name for name, _ in pipeline.steps] == ["scaler"]
