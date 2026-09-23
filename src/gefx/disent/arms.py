"""Roster de implementacoes (arms) de distorcao do POC II.

Cada arm expoe um unico knob de drive; a faixa util sai da calibracao
(`calibrate.py`) e a correspondencia entre arms, do oraculo (`oracle.py`). Os
arms sao estratificados para que a falha possa ser atribuida:

- S1  mesma forma de nao-linearidade e mesma unidade  (pedalboard-tanh, lsp-tanh)
- S2  mesma unidade (`input_gain_db`), outra forma  (lsp-hardclip/arctan/sine)
- S3  unidade e topologia diferentes  (byod-mxr, byod-bigmuff)

O tone e um estagio nosso, identico em todos os arms e aplicado depois da
nao-linearidade, com os tones nativos em neutro: um fator exatamente
compartilhado, que serve de controle positivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from pedalboard import Distortion, LowpassFilter, Pedalboard

LSP_PATH = "plugins/real/lsp-plugins.vst3"
LSP_PLUGIN_NAME = "Clipper Mono"

REFERENCE_ARM = "pedalboard-tanh"

# Tone: passa-baixas de primeira ordem, corte log-espacado. Menos niveis que o
# drive porque e so controle positivo.
TONE_LEVELS = 5
TONE_CUTOFF_HZ: Tuple[float, float] = (500.0, 8000.0)

DRIVE_LEVELS = 8

# A troca de `program` do BYOD e assincrona: sem espera, sai silencio ou o
# circuito anterior.
BYOD_SETTLE_SECONDS = 3.0


@dataclass(frozen=True)
class Arm:
    """Uma implementacao da distorcao.

    `sweep_range` e so o dominio da varredura da calibracao; a faixa util e decidida
    por ela. `drive_in_db` marca os arms cujo knob e o mesmo pre-ganho em dB da
    referencia.
    """

    key: str
    stratum: str
    backend: str
    drive_param: str
    sweep_range: Tuple[float, float]
    path: Optional[str] = None
    plugin_name: Optional[str] = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    # So enderecaveis por `raw_value`: o `valid_values` publicado nao corresponde
    # ao que o plugin seleciona (o `program` do BYOD).
    raw_fixed: Mapping[str, float] = field(default_factory=dict)
    # `audio_signature` do circuito, conferida no carregamento: o nome exibido
    # muda antes de o DSP trocar, entao nao serve de guarda.
    fingerprint: Optional[Tuple[float, float, float]] = None
    drive_in_db: bool = False
    note: str = ""

    def plugin_spec(self) -> Dict[str, Any]:
        """Formato que `effects.vst_adapter.load_arm` espera, sem `raw_fixed` (aplicado pelo `LoadedArm`)."""
        if self.backend != "vst":
            raise ValueError(f"arm {self.key!r} nao e VST3")
        return {"path": self.path, "plugin_name": self.plugin_name, "fixed": dict(self.fixed)}


# Tudo neutro, para que os arms LSP so difiram em `clipper_sigmoid_function`.
_LSP_FIXED: Dict[str, Any] = {
    "bypass": False,
    "clipper_enable": True,
    "enable_input_lufs_limitation": False,
    "overdrive_protection": False,
    "boosting_mode": False,
    "output_gain_db": 0.0,
    "dithering_mode": "None",
    # Em 0 dB o clip de seguranca do plugin satura antes e as sigmoides soam iguais.
    "clipper_sigmoid_threshold_db": -12.0,
    "clipper_dc_offset": 0.0,
    "clipper_sigmoid_pumping_db": 0.0,
}


def _lsp_arm(key: str, stratum: str, sigmoid: str, note: str = "") -> Arm:
    return Arm(
        key=key,
        stratum=stratum,
        backend="vst",
        drive_param="input_gain_db",
        sweep_range=(0.0, 48.0),
        path=LSP_PATH,
        plugin_name=LSP_PLUGIN_NAME,
        fixed={**_LSP_FIXED, "clipper_sigmoid_function": sigmoid},
        drive_in_db=True,
        note=note,
    )


FINGERPRINT_KNOB = 0.0
FINGERPRINT_RTOL = 0.01


def fingerprint_probe(sr: int = 44100) -> np.ndarray:
    """Probe sintetico da assinatura (pilha harmonica de 110 Hz, 0,5 s): nao depende do dataset."""
    t = np.arange(int(0.5 * sr)) / sr
    x = sum(np.sin(2 * np.pi * 110.0 * k * t) / k for k in range(1, 9))
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def audio_signature(segment: np.ndarray, sr: int = 44100) -> Tuple[float, float, float]:
    """(rms, crest em dB, razao de energia acima de 2 kHz).

    Estatisticas e nao um hash dos bytes: o que se quer pegar e circuito trocado,
    nao diferenca de arredondamento entre maquinas.
    """
    a = np.asarray(segment, dtype=np.float64).squeeze()
    rms = float(np.sqrt(np.mean(a**2)))
    crest = 20.0 * np.log10(np.max(np.abs(a)) / (rms + 1e-12) + 1e-12)
    spectrum = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    freqs = np.fft.rfftfreq(len(a), 1.0 / sr)
    return rms, float(crest), float(spectrum[freqs > 2000].sum() / (spectrum.sum() + 1e-20))


def _byod_arm(
    key: str,
    program_raw: float,
    program_name: str,
    fingerprint: Tuple[float, float, float],
    sweep_range: Tuple[float, float],
    note: str = "",
) -> Arm:
    """Um circuito do BYOD, selecionado por `program` (raw = k/40).

    O nome exibido nao identifica o circuito (nomes diferentes dao audio identico);
    quem identifica e a `fingerprint`.
    """
    return Arm(
        key=key,
        stratum="S3",
        backend="vst",
        drive_param="in_gain",
        # Por arm: cada circuito tem regioes patologicas em pontos diferentes.
        sweep_range=sweep_range,
        path="plugins/real/BYOD.vst3",
        fixed={"dry_wet": 100.0, "out_gain": 0.0, "mode": "Mono", "oversampling_factor": 4.0},
        raw_fixed={"program": program_raw},
        fingerprint=fingerprint,
        note=f"{program_name}: {note}" if note else program_name,
    )


ARMS: Tuple[Arm, ...] = (
    Arm(
        key=REFERENCE_ARM,
        stratum="S1",
        backend="pedalboard",
        drive_param="drive_db",
        # A faixa do POC I: o B1 so representa [5, 40] dB.
        sweep_range=(5.0, 40.0),
        drive_in_db=True,
        note="referencia in-domain: tanh(x * 10**(drive_db/20))",
    ),
    _lsp_arm("lsp-tanh", "S1", "Hyperbolic tangent", "mesma forma e mesma unidade da referencia"),
    _lsp_arm("lsp-hardclip", "S2", "Hard clip"),
    _lsp_arm("lsp-arctan", "S2", "Arctangent"),
    _lsp_arm("lsp-sine", "S2", "Sine"),
    _byod_arm(
        "byod-mxr", 0.425, "MXR Distortion",
        (0.146208, 7.0762, 0.003771),
        # Abaixo de -24 dB domina o ruido proprio do circuito.
        sweep_range=(-24.0, 18.0),
        note="op-amp com clipping de diodo para a terra",
    ),
    _byod_arm(
        "byod-bigmuff", 0.275, "Big Muff (Russian)",
        (0.325517, 2.9778, 0.001643),
        # Muito abaixo disso sobra so o fundamental e a planicidade para de variar.
        sweep_range=(-28.0, 18.0),
        note="quatro transistores em cascata, clipping simetrico",
    ),
)

# Candidatos medidos e descartados. O motivo e propriedade do circuito: nao
# adianta tentar de novo sem mudar o desenho.
REJECTED_ARMS: Dict[str, str] = {
    "chowtape": (
        "CHOWTapeModel: teto FISICO de THD 0,172, medido no topo nativo do "
        "`input_gain` (6 dB). Saturacao de fita e suave por construcao. Como a "
        "faixa comum e a intersecao, ele sozinho travava o eixo inteiro em 0,166 "
        "-- audivelmente limpo demais no nivel mais alto."
    ),
    "chowcentaur": (
        "ChowCentaur: teto fisico de THD 0,177 (mesmo problema da chowtape) e piso "
        "de 0,029, por nunca ficar limpo -- ele travava o teto do eixo e levantava "
        "o piso ao mesmo tempo. Essas duas sao as razoes decisivas, ambas duras. "
        "O knob de `gain` tambem funciona como controle de tom (a razao de agudos "
        "varia 36-50x ao longo do drive, em qualquer ajuste de `treble`, contra "
        "1,2x na referencia), mas isso SOZINHO nao desqualifica: a grade e "
        "totalmente cruzada, entao drive e tom continuam separaveis por "
        "construcao, e pedais reais ficam mesmo mais brilhantes com mais ganho. "
        "O byod-bigmuff, que esta no roster, varia 107x e passa nos criterios que "
        "importam."
    ),
    "byod-hotfuzz": (
        "Hot Fuzz: passa em TUDO que a porteira antiga media -- rho 1,000, THD de "
        "0,010 a 0,740, a maior faixa do roster -- e mesmo assim o eixo de drive e "
        "perceptualmente DEGENERADO. Ao longo de toda a faixa do experimento a "
        "compressao da guitarra varia 0,23 dB (os outros variam 1,3 a 7,6) e o "
        "contraste do oraculo cai para 27% no topo, contra ~1000% do lsp-tanh: o "
        "oraculo mal distingue qual nivel casa. Causa: o THD e medido em senoide "
        "casada no PICO da guitarra, e senoide fica no pico o tempo todo enquanto "
        "guitarra tem 15 dB de fator de crista. Num circuito de limiar abrupto a "
        "senoide e ceifada quase sempre e a guitarra so nas pontas -- mesmo THD "
        "medido, quantidades de distorcao percebida incomparaveis."
    ),
    "byod-gainfulclipper": (
        "Melhor variacao de compressao medida (7,57 dB), mas piso de THD 0,0303, "
        "acima do piso do eixo (0,0113): repetiria o defeito do chowcentaur, "
        "puxando o piso comum de todos para cima."
    ),
    "byod-tubescreamer": "Teto de THD 0,168: o mesmo problema de teto baixo da chowtape.",
    "byod-rat": "Piso em THD 0,094 (nunca limpa) e nao monotono (rho -0,21).",
    "byod-kingoftone": "Saida identicamente silenciosa na faixa de `in_gain` sondada.",
    "surge-xt": (
        "Os 13 slots genericos nao mudam com o fx_type; nao da para saber que "
        "valor foi setado nem garantir reprodutibilidade."
    ),
}

# Vies residual medido por escuta cega depois da calibracao, em niveis. Fica
# como metadado, e nao como correcao dos knobs: um avaliador so, IC largo.
#
# No `byod-bigmuff` os niveis 0 a 3 casam todos com o nivel 0 pelo oraculo
# (contraste 0,51-0,65 no audio renderizado): ruido de rotulo conhecido, e limite
# do circuito, nao da calibracao. Ele fica no roster, e todo agregado e
# reportado tambem sem ele.
PERCEPTUAL_BIAS: Dict[str, Dict[str, Any]] = {
    "byod-bigmuff": {
        "bias_levels": -1.17,
        "ci95_levels": [-1.96, -0.38],
        "n_trials": 12,
        "n_listeners": 1,
        "measured_on": "2026-09-07",
        "method": (
            "escuta cega: referencia seguida de candidatos do arm em niveis "
            "vizinhos, ordem sorteada; rodada 1 com 3 alternativas (p-1..p+1), "
            "rodada 2 com 4 (p-2..p+1). Controle lsp-tanh acertou 6/8."
        ),
        "sign": "negativo = soa MAIS distorcido que o nivel nominal",
    },
    "lsp-hardclip": {
        "bias_levels": +0.78,
        "ci95_levels": [0.44, 1.12],
        "n_trials": 9,
        "n_listeners": 1,
        "measured_on": "2026-09-07",
        "method": (
            "mesmo protocolo do byod-bigmuff, 4 alternativas (p-2..p+1), duas "
            "rodadas. So os ensaios em que o avaliador declarou ter base para "
            "julgar entram na conta: metade dos ensaios no extremo limpo foram "
            "chutes, porque tanh e hard clip diferem em ESPECIE e nao em grau, e "
            "a pergunta 'qual tem mais distorcao' pressupoe uma escala comum que "
            "ali nao existe."
        ),
        "sign": "positivo = soa MAIS LIMPO que o nivel nominal",
        "sampling_caveat": (
            "a rodada 4 concentrou-se em p=4,5,6 de proposito; a estimativa vale "
            "para a metade suja do eixo, nao para o eixo inteiro"
        ),
    },
}

ARMS_BY_KEY: Dict[str, Arm] = {item.key: item for item in ARMS}


def arm(key: str) -> Arm:
    if key not in ARMS_BY_KEY:
        raise KeyError(f"arm {key!r} nao existe; conhecidos: {sorted(ARMS_BY_KEY)}")
    return ARMS_BY_KEY[key]


def arm_keys() -> List[str]:
    """Na ordem do roster, que e a ordem dos estratos."""
    return [item.key for item in ARMS]


def arms_in_stratum(stratum: str) -> List[str]:
    return [item.key for item in ARMS if item.stratum == stratum]


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
        from gefx.effects.vst_adapter import load_arm

        self.arm = arm_spec
        self.sr = sr
        self.plugin = None
        self.stereo = False
        if arm_spec.backend == "vst":
            self.plugin, self.stereo = load_arm(arm_spec.plugin_spec(), sr=sr)
            if arm_spec.raw_fixed:
                self._apply_raw_fixed()
        elif arm_spec.backend != "pedalboard":
            raise ValueError(f"backend {arm_spec.backend!r} desconhecido")

    def _apply_raw_fixed(self) -> None:
        """Escreve os parametros de `raw_value`, espera o BYOD trocar de circuito e confere a assinatura."""
        import time

        for name, raw in self.arm.raw_fixed.items():
            if name not in self.plugin.parameters:
                raise ValueError(f"{self.arm.key}: parametro {name!r} nao existe no plugin")
            self.plugin.parameters[name].raw_value = float(raw)

        time.sleep(BYOD_SETTLE_SECONDS)
        silence = np.zeros((2 if self.stereo else 1, self.sr), dtype=np.float32)
        for _ in range(4):
            self.plugin(silence, self.sr, reset=True)

        if self.arm.fingerprint is None:
            return
        got = audio_signature(self.render(fingerprint_probe(self.sr), self.sr, FINGERPRINT_KNOB),
                              self.sr)
        if not np.allclose(got, self.arm.fingerprint, rtol=FINGERPRINT_RTOL):
            raise ValueError(
                f"{self.arm.key}: assinatura {tuple(round(x, 6) for x in got)} nao bate com a "
                f"esperada {self.arm.fingerprint}. O plugin carregou outro circuito -- o mapa "
                f"de `program` mudou, ou a troca nao terminou em {BYOD_SETTLE_SECONDS}s."
            )

    def render(self, segment: np.ndarray, sr: int, drive_knob: float) -> np.ndarray:
        """So a nao-linearidade; o tone e aplicado depois, por `apply_tone`."""
        if self.arm.backend == "pedalboard":
            board = Pedalboard([Distortion(drive_db=float(drive_knob))])
            return board(segment, sr, reset=True)

        from gefx.effects.vst_adapter import process_mono, set_parameter

        set_parameter(self.plugin, self.arm.drive_param, float(drive_knob))
        return process_mono(self.plugin, self.stereo, segment, sr)
