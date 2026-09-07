"""Roster de implementacoes (arms) de distorcao do POC II.

Um arm e uma implementacao da distorcao. A diferenca para `effects/registry.py`
e proposital: la cada parametro de referencia vira um parametro de plugin
preservando a unidade fisica, porque o alvo da regressao precisa ser
interpretavel na mesma escala. Aqui nao ha pretensao de unidade comum -- cada arm
expoe um unico knob de drive, cuja faixa util sai da calibracao
(`disent/calibrate.py`), e a correspondencia entre arms e definida a posteriori
pelo oraculo (`disent/oracle.py`).

Os arms sao estratificados para que a falha possa ser atribuida:

- S1  mesma forma de nao-linearidade e mesma unidade  (pedalboard-tanh, lsp-tanh)
- S2  mesma unidade (`input_gain_db`), forma diferente (lsp-hardclip/arctan/sine)
- S3  unidade e topologia diferentes  (byod-mxr, byod-bigmuff)

O S2 e ablacao controlada da forma da nao-linearidade, nao diversidade de
fabricante: e o mesmo plugin com outra sigmoide. E ele que mostra que
equivalencia nominal e falsa mesmo com unidade identica -- 20 dB em `Hard clip`
distorce muito mais que 20 dB em `Logistic`.

O tone NAO vem dos plugins. E um estagio nosso, identico em todos os arms,
aplicado depois da nao-linearidade, com os controles de tone nativos fixados em
neutro. Isso o torna um fator exatamente compartilhado entre implementacoes, que
serve de **controle positivo** do desemaranhamento -- e nao de teste de
generalizacao.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from pedalboard import Distortion, LowpassFilter, Pedalboard

LSP_PATH = "plugins/real/lsp-plugins.vst3"
LSP_PLUGIN_NAME = "Clipper Mono"

REFERENCE_ARM = "pedalboard-tanh"

# Estagio de tone compartilhado: passa-baixas de primeira ordem (6 dB/oitava),
# corte log-espacado. Identico em todos os arms, por decisao de desenho.
#
# Cinco e nao oito: o tone e **controle positivo** do desemaranhamento, nao teste
# de generalizacao -- ele so precisa de niveis suficientes para mostrar que o
# modelo o recupera. A resolucao que sobra foi para o drive, que e o eixo em
# disputa. A grade continua com 5*8 = 40 configuracoes, ou seja, o mesmo custo
# de render.
TONE_LEVELS = 5
TONE_CUTOFF_HZ: Tuple[float, float] = (500.0, 8000.0)

# Oito e nao cinco. Com chowtape e chowcentaur no roster o teto comum era THD
# 0,166 -- nao por escolha de varredura, mas porque e o limite fisico dos dois
# plugins (medido no topo nativo do parametro: 0,172 e 0,177). Um arm de teto
# baixo achata o eixo de todos, e o resultado era audivelmente limpo demais.
# Sem eles o teto sobe para 0,327 e a faixa medida vai de 9,8 para 22,1 dB
# ([5,00, 27,14] dB equivalentes), o que permite MAIS niveis com passos MAIORES
# ao mesmo tempo: 3,16 dB por nivel contra os 2,45 dB de antes. O POC I errava
# 1,28 dB, entao isso e 2,5x o erro.
DRIVE_LEVELS = 8

# A troca de `program` do BYOD e assincrona: sem espera, circuitos saem em
# silencio absoluto ou com o audio do circuito anterior (verificado). O
# `string_value` ja reporta o nome novo enquanto o DSP ainda e o velho, entao o
# nome NAO serve de guarda -- quem confere e a assinatura de audio.
BYOD_SETTLE_SECONDS = 3.0


@dataclass(frozen=True)
class Arm:
    """Uma implementacao da distorcao.

    `sweep_range` e apenas o dominio em que a calibracao varre o knob; a faixa
    util (`k_lo`, `k_hi`) e decidida por ela e gravada no JSON de calibracao.
    `drive_in_db` marca os arms cujo knob e literalmente o mesmo pre-ganho em dB
    da referencia -- so neles faz sentido reportar erro em dB nativo.
    """

    key: str
    stratum: str
    backend: str
    drive_param: str
    sweep_range: Tuple[float, float]
    path: Optional[str] = None
    plugin_name: Optional[str] = None
    fixed: Mapping[str, Any] = field(default_factory=dict)
    # Parametros que so sao enderecaveis por `raw_value` porque o `valid_values`
    # que o plugin publica nao corresponde ao que ele de fato seleciona (caso do
    # `program` do BYOD).
    raw_fixed: Mapping[str, float] = field(default_factory=dict)
    # Assinatura (rms, crest_db, hf_ratio) do circuito sobre `fingerprint_probe`,
    # conferida no carregamento. Substitui a conferencia pelo nome exibido, que
    # validava a coisa errada: o `string_value` do BYOD reporta o programa novo
    # enquanto o DSP ainda toca o antigo, entao a guarda passava enquanto o audio
    # vinha de outro pedal. Tres estatisticas e nao uma porque o rms sozinho nao
    # separa: MXR e Big Muff diferem 2,2x no rms, mas duas variantes do mesmo Big
    # Muff diferem so 1,9% -- ai quem separa e a razao de agudos, que difere 15%.
    fingerprint: Optional[Tuple[float, float, float]] = None
    drive_in_db: bool = False
    note: str = ""

    def plugin_spec(self) -> Dict[str, Any]:
        """Formato que `effects.vst_adapter.load_arm` espera.

        `raw_fixed` fica de fora: o adaptador escreve por `setattr` e esses
        parametros exigem `raw_value`. Quem aplica e o `LoadedArm`.
        """
        if self.backend != "vst":
            raise ValueError(f"arm {self.key!r} nao e VST3")
        return {"path": self.path, "plugin_name": self.plugin_name, "fixed": dict(self.fixed)}


# Fixos comuns aos arms do LSP Clipper: tudo que colore o som fora a sigmoide e o
# ganho de entrada fica desligado ou neutro, para que a unica diferenca entre os
# arms do S1/S2 seja `clipper_sigmoid_function`.
_LSP_FIXED: Dict[str, Any] = {
    "bypass": False,
    "clipper_enable": True,
    "enable_input_lufs_limitation": False,
    "overdrive_protection": False,
    "boosting_mode": False,
    "output_gain_db": 0.0,
    "dithering_mode": "None",
    # ABAIXO de 0 dBFS de proposito: com o threshold da sigmoide em 0 dB o hard
    # clip de seguranca do proprio plugin chega primeiro e todas as sigmoides
    # produzem exatamente o mesmo audio (verificado: THD identico ate a 4a casa).
    # A -12 dB a sigmoide e quem satura, e as formas se separam.
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
        # As sigmoides duras precisam de mais ganho que a tanh para comecar a
        # distorcer; a faixa util medida fica em torno de 10..30 dB.
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
    """Probe deterministico da assinatura: pilha harmonica de 110 Hz, 0,5 s.

    Sintetico e nao um trecho de guitarra porque a assinatura precisa ser
    reproduzivel sem depender do dataset estar no disco.
    """
    t = np.arange(int(0.5 * sr)) / sr
    x = sum(np.sin(2 * np.pi * 110.0 * k * t) / k for k in range(1, 9))
    return (0.25 * x / np.max(np.abs(x))).astype(np.float32)


def audio_signature(segment: np.ndarray, sr: int = 44100) -> Tuple[float, float, float]:
    """(rms, crest em dB, razao de energia acima de 2 kHz).

    Tres estatisticas de forma e de espectro, nao um hash dos bytes: um hash
    quebraria com qualquer diferenca de arredondamento entre maquinas, e o que
    se quer detectar e circuito trocado, nao ruido de ponto flutuante.
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
    """Um circuito do BYOD.

    O `program` seleciona pedais reais e e o que torna o S3 diverso de verdade.
    Ele so e enderecavel por `raw_value` (o `valid_values` publicado pelo plugin
    nao corresponde ao que ele seleciona) e o mapa medido e raw = k/40.

    `program_name` fica so como documentacao: o nome exibido nao identifica o
    circuito (dos 40 nomes saem 31 audios distintos, e pares como "MXR
    Distortion"/"OctaVerb" produzem audio identico). Quem identifica e a
    `fingerprint`.
    """
    return Arm(
        key=key,
        stratum="S3",
        backend="vst",
        drive_param="in_gain",
        # Por arm, e nao um valor unico: os circuitos do BYOD tem regioes
        # patologicas em pontos diferentes, e varrer dentro delas reprova o arm na
        # monotonicidade por artefato de medicao, nao por propriedade do circuito.
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
        # A faixa do POC I (`effects/catalog.py`), que e o que define a unidade
        # de leitura "drive-tanh equivalente em dB". NAO estender abaixo de 5 dB:
        # o baseline B1 e o regressor do POC I, cuja saida sigmoide so representa
        # [5, 40], e um dataset fora dessa faixa o tornaria incomparavel. Nao ha
        # perda: em 5 dB a referencia esta em THD 0,0113, acima do piso de todos
        # os outros arms, entao e ela que prende o piso e a intersecao cai
        # exatamente na borda do POC I.
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
        # Piso em -24: abaixo de -23 dB o THD para de cair (fica em 0,0006) e
        # volta a subir em -48 (0,0083), que e o ruido proprio do circuito. Varrer
        # ate -48 reprovava o arm com rho 0,85; ate -24 da rho 1,000. O piso do
        # eixo (THD 0,0113) fica bem dentro dessa faixa.
        sweep_range=(-24.0, 18.0),
        note="op-amp com clipping de diodo para a terra",
    ),
    _byod_arm(
        "byod-bigmuff", 0.275, "Big Muff (Russian)",
        (0.325517, 2.9778, 0.001643),
        # Piso em -28 e nao -48: abaixo de -15 dB este circuito e tao escuro que
        # sobra praticamente so o fundamental, e a planicidade despenca para 0,02
        # (contra 30 da guitarra seca) e para de variar -- regiao degenerada que
        # derrubava o rho para 0,80. A partir de -28,8 o rho e 0,998. O piso do
        # eixo em planicidade cai perto de -3 dB, bem dentro dessa faixa.
        sweep_range=(-28.0, 18.0),
        note="quatro transistores em cascata, clipping simetrico",
    ),
)

# Candidatos medidos e descartados, com o motivo. Ficam registrados porque a
# medicao custou caro e porque a razao da rejeicao e propriedade do circuito, nao
# do codigo: nao adianta tentar de novo sem mudar o desenho.
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

# As medicoes que rejeitaram `byod-bigmuff` (piso THD 0,084), `byod-tubescreamer`
# (rho +0,29) e `byod-rat` (rho +0,04) numa versao anterior deste arquivo estavam
# ERRADAS: foram feitas trocando `program` sem esperar o DSP, entao mediram o
# circuito anterior. Refeitas com espera, o Big Muff tem piso 0,0087 e rho 1,000,
# e por isso ele voltou ao roster. Fica registrado para ninguem confiar em
# medicao de BYOD feita sem a espera.

# Vies perceptual RESIDUAL, medido por escuta cega depois da calibracao.
#
# A calibracao casa os 8 niveis pelo descritor combinado e reporta desvio maximo
# de 0,04 dB no byod-bigmuff -- ou seja, pelo descritor ele esta alinhado. O
# ouvido discorda: em 12 ensaios cegos o avaliador escolheu um nivel ABAIXO do
# nominal para casar a referencia, sem uma unica excecao de direcao.
#
# Fica como METADADO e nao como correcao dos knobs, por tres razoes: o IC vai de
# -0,38 a -1,96 (fator de cinco, corrigir por -1,17 seria mais preciso que a
# medida), o numero vem de UM avaliador, e assar isso na grade a tornaria
# irreproduzivel a partir do codigo. Quem quiser corrigir a posteriori tem o
# numero aqui.
#
# Ponto em aberto: a causa nao foi identificada. O avaliador descreveu um
# "chiado" que passa por distorcao mas pode ser caracteristica do circuito, e
# nenhum dos descritores testados (THD, crest, centroide, razao de agudos,
# planicidade, e a combinacao) o captura.
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

# Os dois vieses tem SINAIS OPOSTOS, e e isso que os torna interessantes: o erro
# do descritor de calibracao nao e ruido, e sinalizado e dependente da forma da
# nao-linearidade -- ele erra para cima num arm e para baixo no outro, e reporta
# ambos como casados (0,04 e 0,25 dB de desvio).
#
# O oraculo log-mel (`disent/oracle.py`) foi a UNICA medida automatica a acertar
# os dois sinais: -1,76 contra -1,17 do ouvido no bigmuff, +1,16 contra +0,78 no
# hard clip. Planicidade, crest, centroide e o proprio descritor combinado falham
# em pelo menos um dos dois.

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
    """Corte do nivel de tone, log-espacado em `TONE_CUTOFF_HZ`.

    Log e nao linear porque a percepcao de brilho e logaritmica na frequencia:
    500->1000 Hz e um passo comparavel a 4000->8000 Hz.
    """
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


# --- arm carregado ------------------------------------------------------------
class LoadedArm:
    """Um arm pronto para renderizar.

    Existe porque carregar um VST3 e caro e o plugin guarda estado: ele e
    carregado uma vez por processo e so o knob de drive muda entre renders. O
    `load_arm` do adaptador ja gasta a primeira chamada de processamento, que
    sairia com os parametros antigos.
    """

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
        """Escreve os parametros que so aceitam `raw_value` e confere a assinatura.

        A troca de `program` do BYOD nao e instantanea: o plugin reconstroi o
        grafo de processamento e, ate terminar, renderiza silencio ou o circuito
        anterior. Por isso a espera e a queima de silencio antes de qualquer
        medicao.

        A conferencia e por audio e nao pelo nome exibido. O `string_value` passa
        a reportar o programa novo assim que o parametro e escrito, muito antes
        de o DSP trocar -- conferir o nome deixava passar exatamente a falha que
        se queria pegar.
        """
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
