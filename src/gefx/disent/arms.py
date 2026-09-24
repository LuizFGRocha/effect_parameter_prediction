"""Roster de implementacoes (arms) de distorcao do POC II, lido de um YAML.

Cada arm expoe um unico knob de drive, e os valores dele em cada nivel sao
pareados de ouvido contra a referencia (ver `experiments/disent_roster.yaml`). O
roster e dado, nao codigo: trocar de plugin ou de maquina e editar o arquivo.

O tone e um estagio nosso, identico em todos os arms e aplicado depois da
nao-linearidade, com os tones nativos em neutro: um fator exatamente
compartilhado, que serve de controle positivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from pedalboard import Distortion, LowpassFilter, Pedalboard

DEFAULT_ROSTER = Path("experiments/disent_roster.yaml")

# Tone: passa-baixas de primeira ordem, corte log-espacado. Menos niveis que o
# drive porque e so controle positivo.
TONE_LEVELS = 5
TONE_CUTOFF_HZ: Tuple[float, float] = (500.0, 8000.0)

BACKENDS = ("pedalboard", "vst")


@dataclass(frozen=True)
class Arm:
    """Uma implementacao da distorcao e os seus niveis de drive."""

    key: str
    stratum: str
    drive_param: str
    levels: Tuple[float, ...]
    backend: str = "vst"
    path: Optional[str] = None
    plugin_name: Optional[str] = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    # So enderecaveis por `raw_value`: o `valid_values` publicado nao corresponde
    # ao que o plugin seleciona (o `program` do BYOD).
    raw_fixed: Mapping[str, float] = field(default_factory=dict)
    # Espera depois de `raw_fixed`, para plugins que trocam de estado assincronamente.
    settle_seconds: float = 0.0

    def plugin_spec(self) -> Dict[str, Any]:
        """Formato que `effects.vst_adapter.load_arm` espera."""
        return {"path": self.path, "plugin_name": self.plugin_name, "fixed": dict(self.fixed)}


@dataclass(frozen=True)
class Roster:
    reference: str
    arms: Tuple[Arm, ...]

    @property
    def drive_levels(self) -> int:
        return len(self.arms[0].levels)

    def arm(self, key: str) -> Arm:
        for item in self.arms:
            if item.key == key:
                return item
        raise KeyError(f"arm {key!r} nao esta no roster; ha {self.keys()}")

    def keys(self) -> List[str]:
        return [item.key for item in self.arms]

    def reference_levels(self) -> Tuple[float, ...]:
        """O knob da referencia em cada nivel: a unidade comum do erro da avaliacao."""
        return self.arm(self.reference).levels


def parse_roster(data: Mapping[str, Any]) -> Roster:
    """Valida o roster antes de qualquer render: um erro aqui custa segundos, la horas."""
    arms_data = data.get("arms") or {}
    if not arms_data:
        raise ValueError("roster sem arms")
    arms: List[Arm] = []
    for key, spec in arms_data.items():
        spec = dict(spec)
        unknown = set(spec) - {"stratum", "backend", "drive_param", "levels", "path",
                               "plugin_name", "fixed", "raw_fixed", "settle_seconds"}
        if unknown:
            raise ValueError(f"{key}: campos desconhecidos {sorted(unknown)}")
        for required in ("stratum", "drive_param", "levels"):
            if required not in spec:
                raise ValueError(f"{key}: falta {required!r}")
        backend = spec.get("backend", "vst")
        if backend not in BACKENDS:
            raise ValueError(f"{key}: backend {backend!r} desconhecido; ha {BACKENDS}")
        if backend == "vst" and not spec.get("path"):
            raise ValueError(f"{key}: arm VST3 sem `path`")
        levels = tuple(float(value) for value in spec["levels"])
        if len(levels) < 2:
            raise ValueError(f"{key}: sao precisos ao menos 2 niveis")
        arms.append(Arm(
            key=str(key),
            stratum=str(spec["stratum"]),
            drive_param=str(spec["drive_param"]),
            levels=levels,
            backend=backend,
            path=spec.get("path"),
            plugin_name=spec.get("plugin_name"),
            fixed=dict(spec.get("fixed") or {}),
            raw_fixed={k: float(v) for k, v in (spec.get("raw_fixed") or {}).items()},
            settle_seconds=float(spec.get("settle_seconds", 0.0)),
        ))

    counts = {item.key: len(item.levels) for item in arms}
    if len(set(counts.values())) != 1:
        raise ValueError(f"todo arm precisa do mesmo numero de niveis: {counts}")
    reference = str(data.get("reference", ""))
    if reference not in counts:
        raise ValueError(f"referencia {reference!r} nao esta entre os arms {list(counts)}")
    return Roster(reference=reference, arms=tuple(arms))


def load_roster(path: Path = DEFAULT_ROSTER) -> Roster:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return parse_roster(yaml.safe_load(handle))


# --- estagio de tone compartilhado -------------------------------------------
def tone_cutoff_hz(level: int, n_levels: int = TONE_LEVELS) -> float:
    """Corte do nivel de tone, log-espacado em `TONE_CUTOFF_HZ`."""
    if n_levels < 2:
        raise ValueError("n_levels deve ser >= 2")
    if not 0 <= level < n_levels:
        raise ValueError(f"level {level} fora de [0, {n_levels})")
    lo, hi = TONE_CUTOFF_HZ
    return float(lo * (hi / lo) ** (level / (n_levels - 1)))


def apply_tone(segment: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Aplica o estagio de tone. Identico em todos os arms, por construcao."""
    return Pedalboard([LowpassFilter(cutoff_frequency_hz=float(cutoff_hz))])(
        segment, sr, reset=True
    )


class LoadedArm:
    """Um arm carregado uma vez por processo; so o knob de drive muda entre renders."""

    def __init__(self, arm_spec: Arm, sr: int = 44100) -> None:
        self.arm = arm_spec
        self.sr = sr
        self.plugin = None
        self.stereo = False
        if arm_spec.backend == "vst":
            from gefx.effects.vst_adapter import load_arm

            self.plugin, self.stereo = load_arm(arm_spec.plugin_spec(), sr=sr)
            self._check_drive()
            if arm_spec.raw_fixed:
                self._apply_raw_fixed()

    def _check_drive(self) -> None:
        """O knob de drive falha alto: `set_parameter` so avisaria, e um nome errado
        renderizaria todos os niveis iguais."""
        name = self.arm.drive_param
        if name not in self.plugin.parameters:
            raise ValueError(
                f"{self.arm.key}: o plugin nao tem o parametro {name!r}. Ha: "
                f"{sorted(self.plugin.parameters)}"
            )
        param = self.plugin.parameters[name]
        lo, hi = param.min_value, param.max_value
        fora = [value for value in self.arm.levels if not lo <= value <= hi]
        if fora:
            raise ValueError(f"{self.arm.key}: niveis {fora} fora da faixa [{lo}, {hi}] "
                             f"de {name!r}")

    def _apply_raw_fixed(self) -> None:
        import time

        for name, raw in self.arm.raw_fixed.items():
            if name not in self.plugin.parameters:
                raise ValueError(f"{self.arm.key}: parametro {name!r} nao existe no plugin")
            self.plugin.parameters[name].raw_value = float(raw)

        time.sleep(self.arm.settle_seconds)
        silence = np.zeros((2 if self.stereo else 1, self.sr), dtype=np.float32)
        for _ in range(4):
            self.plugin(silence, self.sr, reset=True)

    def render(self, segment: np.ndarray, sr: int, drive_knob: float) -> np.ndarray:
        """So a nao-linearidade; o tone e aplicado depois, por `apply_tone`."""
        if self.arm.backend == "pedalboard":
            board = Pedalboard([Distortion(drive_db=float(drive_knob))])
            return board(segment, sr, reset=True)

        from gefx.effects.vst_adapter import process_mono, set_parameter

        set_parameter(self.plugin, self.arm.drive_param, float(drive_knob))
        return process_mono(self.plugin, self.stereo, segment, sr)
