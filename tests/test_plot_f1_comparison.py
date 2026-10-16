from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
from scripts import plot_f1_comparison as comparison


def test_supported_chinese_font_keeps_chinese_labels(monkeypatch):
    with plt.rc_context():
        monkeypatch.setattr(comparison.font_manager.fontManager, "ttflist", [SimpleNamespace(name="SimHei")])
        assert comparison.configure_plot_style() is True
        assert plt.rcParams["font.sans-serif"][0] == "SimHei"


def test_plot_exports_without_chinese_fonts(monkeypatch, tmp_path):
    fonts = comparison.font_manager.fontManager.ttflist
    monkeypatch.setattr(comparison.font_manager.fontManager, "ttflist",
                        [font for font in fonts if font.name == "DejaVu Sans"])
    base = {"macro_f1": 0.75, "macro_f1_percent": 75., "accuracy": 0.8,
            "num_sequences": 4, "train_sequences": 12}
    val = [{**base, "method": "CNN + Softmax", "method_key": "cnn_softmax"}]
    test = [val[0].copy(), {**base, "method": "1 模型投票", "method_key": "vote"}]
    audit = {"common": {"num_classes": 2, "sequence_length": 256}}
    monkeypatch.setattr(comparison, "validation_results", lambda path: (val, audit))
    monkeypatch.setattr(comparison, "test_results", lambda *args: (test, {}))
    with plt.rc_context(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison.main(results_dir=tmp_path, predictions_csv=tmp_path / "predictions.csv",
                        labels_file=tmp_path / "labels.txt", output_dir=tmp_path / "plot")
    messages = [str(item.message) for item in caught]
    assert any("using English plot labels" in message for message in messages)
    assert not any("Glyph" in message and "missing" in message for message in messages)
    output = tmp_path / "plot"
    for extension in ("png", "svg", "pdf"):
        assert (output / f"macro_f1_comparison.{extension}").stat().st_size > 0
    svg = (output / "macro_f1_comparison.svg").read_text(encoding="utf-8")
    assert "Synthetic radar experiment comparison" in svg
    assert "1-model voting" in svg
    for name in ("f1_source_data.csv", "comparison_audit.json", "figure_notes.md"):
        assert (output / name).stat().st_size > 0
