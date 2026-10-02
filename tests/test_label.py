from autocut.label import load_labels, save_labels


def test_labels_roundtrip_sorted(tmp_path):
    p = tmp_path / "x.csv"
    save_labels(str(p), [(30.0, 41.256), (5.5, 12.0), (0.0, 4.0, "warmup")])
    assert load_labels(str(p)) == [(5.5, 12.0), (30.0, 41.26)]
    assert load_labels(str(p), kind="warmup") == [(0.0, 4.0)]
    assert load_labels(str(tmp_path / "missing.csv")) == []
