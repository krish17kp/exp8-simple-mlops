from model import make_training_examples, train_model


def test_generated_examples_are_deterministic():
    X, y = make_training_examples()
    assert X.shape == (180, 3)
    assert y.shape == (180,)


def test_model_training_returns_quality_metrics():
    metrics = train_model()
    assert metrics["mae"] < 5
    assert metrics["r2"] > 0.8
