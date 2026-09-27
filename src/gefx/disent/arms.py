"""Roster de implementacoes (arms) de distorcao do POC II, lido de dois YAMLs.

- `experiments/disent_roster.yaml`: os plugins, editado a mao. Trocar de plugin
  ou de maquina e editar o arquivo.
- `experiments/disent_levels.yaml`: o knob de drive de cada arm em cada nivel.
  Escrito so por `gefx disent calibrate`, pareando os arms pelo Rnonlin; nao se
  edita a mao.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pedalboard import Distortion, Pedalboard

DEFAULT_ROSTER = Path("experiments/disent_roster.yaml")
DEFAULT_LEVELS = Path("experiments/disent_levels.yaml")

BACKENDS = ("pedalboard", "vst")


@dataclass(frozen=True)
class Arm:
    """Uma implementacao da distorcao e os seus niveis de drive."""

    key: str
    stratum: str
    drive_param: str
    levels: Tuple[float, ...] = ()
    backend: str = "vst"
    path: Optional[str] = None
    plugin_name: Optional[str] = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    # So enderecaveis por `raw_value`: o `valid_values` publicado nao corresponde
    # ao que o plugin seleciona (o `program` do BYOD).
    raw_fixed: Mapping[str, float] = field(default_factory=dict)
    # Espera depois de `raw_fixed`, para plugins que trocam de estado assincronamente.
    settle_seconds: float = 0.0
    # Faixa do knob varrida na calibracao; sem ela, a do proprio parametro.
    sweep: Optional[Tuple[float, float]] = None

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


def parse_roster(data: Mapping[str, Any],
                 levels: Optional[Mapping[str, Sequence[float]]] = None) -> Roster:
    """Valida o roster antes de qualquer render: um erro aqui custa segundos, la horas.

    Sem `levels`, os arms saem sem niveis: e o que a calibracao precisa.
    """
    arms_data = data.get("arms") or {}
    if not arms_data:
        raise ValueError("roster sem arms")
    arms: List[Arm] = []
    for key, spec in arms_data.items():
        spec = dict(spec)
        unknown = set(spec) - {"stratum", "backend", "drive_param", "path", "plugin_name",
                               "fixed", "raw_fixed", "settle_seconds", "sweep"}
        if unknown:
            raise ValueError(f"{key}: campos desconhecidos {sorted(unknown)}")
        for required in ("stratum", "drive_param"):
            if required not in spec:
                raise ValueError(f"{key}: falta {required!r}")
        backend = spec.get("backend", "vst")
        if backend not in BACKENDS:
            raise ValueError(f"{key}: backend {backend!r} desconhecido; ha {BACKENDS}")
        if backend == "vst" and not spec.get("path"):
            raise ValueError(f"{key}: arm VST3 sem `path`")
        if backend == "pedalboard" and "sweep" not in spec:
            raise ValueError(f"{key}: arm pedalboard sem `sweep`")
        sweep = spec.get("sweep")
        if sweep is not None:
            sweep = (float(sweep[0]), float(sweep[1]))
            if not sweep[0] < sweep[1]:
                raise ValueError(f"{key}: `sweep` deve ser [min, max] crescente")
        arms.append(Arm(
            key=str(key),
            stratum=str(spec["stratum"]),
            drive_param=str(spec["drive_param"]),
            levels=arm_levels(str(key), levels),
            backend=backend,
            path=spec.get("path"),
            plugin_name=spec.get("plugin_name"),
            fixed=dict(spec.get("fixed") or {}),
            raw_fixed={k: float(v) for k, v in (spec.get("raw_fixed") or {}).items()},
            settle_seconds=float(spec.get("settle_seconds", 0.0)),
            sweep=sweep,
        ))

    reference = str(data.get("reference", ""))
    if reference not in [item.key for item in arms]:
        raise ValueError(f"referencia {reference!r} nao esta entre os arms "
                         f"{[item.key for item in arms]}")
    if levels is not None:
        counts = {item.key: len(item.levels) for item in arms}
        if len(set(counts.values())) != 1:
            raise ValueError(f"todo arm precisa do mesmo numero de niveis: {counts}")
    return Roster(reference=reference, arms=tuple(arms))


def arm_levels(key: str, levels: Optional[Mapping[str, Sequence[float]]]) -> Tuple[float, ...]:
    if levels is None:
        return ()
    if key not in levels:
        raise ValueError(f"{key}: sem niveis no arquivo de niveis (rode `gefx disent calibrate`)")
    values = tuple(float(value) for value in levels[key])
    if len(values) < 2:
        raise ValueError(f"{key}: sao precisos ao menos 2 niveis")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{key}: os niveis devem ser estritamente crescentes: {values}")
    return values


def load_levels(path: Path = DEFAULT_LEVELS) -> Dict[str, Any]:
    """O arquivo de niveis inteiro: `descriptor`, `targets` e, por arm, `levels` e `unmatched`."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_levels(path: Path, data: Mapping[str, Any]) -> None:
    import yaml

    header = ("# Gerado por `gefx disent calibrate`; nao se edita. `levels` e o knob de cada\n"
              "# nivel; `unmatched`, os niveis (base 1) fora do alcance do arm, na ponta do knob.\n")
    text = yaml.safe_dump(dict(data), sort_keys=False, default_flow_style=None, width=4096)
    Path(path).write_text(header + text, encoding="utf-8")


def load_roster(path: Path = DEFAULT_ROSTER,
                levels_path: Optional[Path] = DEFAULT_LEVELS) -> Roster:
    """`levels_path=None` carrega so os plugins, sem niveis."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if levels_path is None:
        return parse_roster(data)
    arms = load_levels(levels_path)["arms"]
    return parse_roster(data, {key: spec["levels"] for key, spec in arms.items()})


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
        """A nao-linearidade do arm, sem normalizar a loudness."""
        if self.arm.backend == "pedalboard":
            board = Pedalboard([Distortion(drive_db=float(drive_knob))])
            return board(segment, sr, reset=True)

        from gefx.effects.vst_adapter import process_mono, set_parameter

        set_parameter(self.plugin, self.arm.drive_param, float(drive_knob))
        wet = process_mono(self.plugin, self.stereo, segment, sr)
        if not np.all(np.isfinite(wet)):
            # Sem isto o NaN vira um wav corrompido ou um Rnonlin NaN, sem erro. A
            # sondagem do `load_arm` ja troca a instancia que nasce assim.
            raise RuntimeError(f"{self.arm.key}: o plugin devolveu NaN/inf (knob {drive_knob})")
        return wet
