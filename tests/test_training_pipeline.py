from fish_pipeline.pipelines.training import (
    DensityModelTrainingPipeline,
    TrainingConfig,
)


def test_training_pipeline_runs_and_reports_metrics():
    config = TrainingConfig(num_samples=10, image_size=(32, 32), validation_split=0.3, random_seed=1)
    pipeline = DensityModelTrainingPipeline(config=config)
    report = pipeline.run()

    assert report.training_samples > 0
    assert report.validation_samples > 0
    assert report.count_mae >= 0
    assert report.count_rmse >= 0
    assert report.scale > 0
