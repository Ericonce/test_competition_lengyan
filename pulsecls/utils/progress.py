"""Terminal progress with periodic refresh during blocking I/O or accelerator work."""
from __future__ import annotations

import sys
import threading

from tqdm import tqdm


class Progress:
    """Count completed work; refresh elapsed/remaining time even between updates."""

    def __init__(self, description, total=None, unit="seq", *, enabled=True, position=None):
        self.description, self.total, self.unit = description, total, unit
        self.enabled, self.position = bool(enabled), position
        self._stop = threading.Event()
        self._thread = None
        self.bar = None

    def __enter__(self):
        self.bar = tqdm(total=self.total, desc=self.description, unit=self.unit,
                        disable=not self.enabled, file=sys.stdout, dynamic_ncols=True,
                        mininterval=0.5, position=self.position, leave=True,
                        bar_format=("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                                    if self.total is not None else "{desc}: working [{elapsed}{postfix}]"))
        if self.enabled:
            self._thread = threading.Thread(target=self._refresh, daemon=True, name="pulsecls-progress")
            self._thread.start()
        return self

    def _refresh(self):
        while not self._stop.wait(1.0):
            self.bar.refresh()

    def update(self, count=1):
        self.bar.update(count)

    def stage(self, description):
        self.bar.set_postfix_str(description, refresh=False)

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if error_type is not None:
            self.bar.set_postfix_str("interrupted" if issubclass(error_type, KeyboardInterrupt) else "failed", refresh=False)
        elif self.total is None:
            self.bar.set_postfix_str("done", refresh=False)
        self.bar.close()
        return False
