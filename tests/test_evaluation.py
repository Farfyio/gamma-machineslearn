from gamma_machineslearn.evaluation import anomaly_detection_metrics



def test_anomaly_detection_metrics_without_scores():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = anomaly_detection_metrics(y_true, y_pred)

    assert set(metrics) == {"precision", "recall", "f1"}



def test_anomaly_detection_metrics_with_scores():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_score = [0.1, 0.8, 0.4, 0.2]

    metrics = anomaly_detection_metrics(y_true, y_pred, y_score=y_score)

    assert "roc_auc" in metrics
