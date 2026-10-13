import io
import threading
from unittest.mock import patch

import pytest

from pulsecls.utils.progress import Progress


def test_progress_refreshes_during_blocking_work_and_stops_on_completion():
    refreshed = threading.Event()
    output = io.StringIO()
    with patch("pulsecls.utils.progress.sys.stdout", output):
        with Progress("Blocking I/O") as progress:
            original = progress.bar.refresh
            def refresh(*args, **kwargs):
                original(*args, **kwargs)
                refreshed.set()
            progress.bar.refresh = refresh
            assert refreshed.wait(3), "Elapsed time must refresh even without counter updates"
        assert not progress._thread.is_alive()
    assert "Blocking I/O" in output.getvalue() and "done" in output.getvalue()


def test_failed_progress_preserves_completed_count_and_closes_refresh_thread():
    output = io.StringIO()
    with patch("pulsecls.utils.progress.sys.stdout", output):
        with pytest.raises(RuntimeError, match="bad input"):
            with Progress("Read CSV", total=3) as progress:
                progress.update()
                raise RuntimeError("bad input")
        assert not progress._thread.is_alive()
    assert progress.bar.n == 1
    assert "failed" in output.getvalue() and "100%" not in output.getvalue()


def test_disabled_progress_has_no_output_or_background_thread():
    output = io.StringIO()
    with patch("pulsecls.utils.progress.sys.stdout", output):
        with Progress("Hidden", total=2, enabled=False) as progress:
            progress.update(2)
    assert output.getvalue() == "" and progress._thread is None
