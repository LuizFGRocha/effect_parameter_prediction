"""Ponte entre o catalogo e os plugins embutidos do Pedalboard.

So o que e decisao deste repositorio: a cobertura do catalogo e a rejeicao de um
efeito desconhecido. Se os construtores do Pedalboard aceitam ou nao um kwarg e
comportamento de terceiros.
"""
from __future__ import annotations

import pytest

from gefx.effects.catalog import EFFECT_PARAMETER_RANGES
from gefx.effects.pedalboard_backend import BUILT_IN_PLUGINS, build_chain


def test_every_catalog_effect_has_a_plugin():
    # Um efeito novo no catalogo sem entrada aqui so falharia na hora de
    # renderizar o dataset.
    assert set(BUILT_IN_PLUGINS) == set(EFFECT_PARAMETER_RANGES)


def test_build_chain_rejects_unknown_effects():
    with pytest.raises(ValueError, match="Effect 'nao_existe' not recognized"):
        build_chain(["nao_existe"], {"nao_existe": {}})
