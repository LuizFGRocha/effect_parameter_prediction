"""Propriedades que mantem a suite rapida e as duas listas de features alinhadas."""
from __future__ import annotations

import subprocess
import sys

from gefx.config import FEATURE_CHOICES
from gefx.data.features import FEATURE_NAMES

CHECK = """
import importlib
import pkgutil
import sys

import gefx

for module in pkgutil.walk_packages(gefx.__path__, prefix="gefx."):
    importlib.import_module(module.name)

heavy = sorted(name for name in ("tensorflow", "keras") if name in sys.modules)
print(",".join(heavy))
"""


def test_importing_the_package_does_not_load_tensorflow():
    # `keras`/`tf` sao importados dentro das funcoes que precisam deles. E isso
    # que deixa a suite inteira rodar em segundos; se alguem subir um import para
    # o topo de um modulo, isso fica vermelho em vez de a suite ficar mais lenta
    # sem ninguem notar. Roda em subprocesso porque outros testes ja podem ter
    # carregado o TensorFlow neste processo.
    result = subprocess.run(
        [sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"import pesado no topo de um modulo: {result.stdout}"


def test_the_two_feature_lists_do_not_drift():
    # `config.FEATURE_CHOICES` (usada pelo `choices` do argparse) e
    # `features.FEATURE_NAMES` (usada pela extracao) sao listas duplicadas.
    assert FEATURE_CHOICES == FEATURE_NAMES


def test_requirements_txt_lists_the_same_dependencies_as_pyproject():
    # `requirements.txt` existe para quem instala sem o pacote; e uma copia da
    # lista do `pyproject.toml`, e uma copia desatualizada falha na instalacao
    # de outra maquina, nao aqui.
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    listed = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert listed == declared["project"]["dependencies"]
