from __future__ import annotations

from pulsecls.signals.base import PhaseBuilder
from pulsecls.signals.bpsk import bpsk_phase
from pulsecls.signals.fsk import fsk_phase
from pulsecls.signals.lfm import lfm_phase
from pulsecls.signals.lfm_bpsk import lfm_bpsk_phase
from pulsecls.signals.lfm_fsk import lfm_fsk_phase
from pulsecls.signals.nlfm import nlfm_phase


PHASE_BUILDERS: dict[str, PhaseBuilder] = {
    "lfm": lfm_phase,
    "nlfm": nlfm_phase,
    "fsk": fsk_phase,
    "bpsk": bpsk_phase,
    "lfm_fsk": lfm_fsk_phase,
    "lfm_bpsk": lfm_bpsk_phase,
}


def get_phase_builder(name: str) -> PhaseBuilder:
    key = str(name).lower()
    if key not in PHASE_BUILDERS:
        available = ", ".join(PHASE_BUILDERS)
        raise ValueError(f"Unknown modulation {name!r}; available: {available}")
    return PHASE_BUILDERS[key]
