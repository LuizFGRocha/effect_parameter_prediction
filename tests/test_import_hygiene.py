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
