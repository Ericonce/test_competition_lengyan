import warnings

import matplotlib.pyplot as plt
from scripts import plot_f1_comparison as comparison


def test_plot_style_uses_bundled_english_font():
    with plt.rc_context():
        comparison.configure_plot_style()
        assert plt.rcParams["font.sans-serif"] == ["DejaVu Sans"]


def test_plot_exports_without_chinese_fonts(monkeypatch, tmp_path):
    base = {"macro_f1": 0.75, "macro_f1_percent": 75., "accuracy": 0.8,
            "num_sequences": 4, "train_sequences": 12}
    val = [{**base, "method": "CNN + Softmax", "method_key": "cnn_softmax"}]
    test = [val[0].copy(), {**base, "method": "1-model voting", "method_key": "vote"}]
    audit = {"common": {"num_classes": 2, "sequence_length": 256}}
    monkeypatch.setattr(comparison, "validation_results", lambda path: (val, audit))
    monkeypatch.setattr(comparison, "test_results", lambda *args: (test, {}))
    with plt.rc_context(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison.main(results_dir=tmp_path, predictions_csv=tmp_path / "predictions.csv",
                        labels_file=tmp_path / "labels.txt", output_dir=tmp_path / "plot")
    messages = [str(item.message) for item in caught]
    assert not any("Glyph" in message and "missing" in message for message in messages)
    output = tmp_path / "plot"
    for extension in ("png", "svg", "pdf"):
        assert (output / f"macro_f1_comparison.{extension}").stat().st_size > 0
    svg = (output / "macro_f1_comparison.svg").read_text(encoding="utf-8")
    assert "Synthetic radar experiment comparison" in svg
    assert "1-model voting" in svg
    for name in ("f1_source_data.csv", "comparison_audit.json", "figure_notes.md"):
        assert (output / name).stat().st_size > 0
