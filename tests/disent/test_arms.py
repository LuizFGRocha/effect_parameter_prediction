"""O roster como dado: o que o YAML precisa ter para o render nao gastar horas a toa."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gefx.disent.arms import (
    DEFAULT_ROSTER,
    LoadedArm,
    load_levels,
    load_roster,
    parse_roster,
    write_levels,
)


def _data(**arms):
    base = {
        "ref": {"stratum": "S1", "backend": "pedalboard", "drive_param": "drive_db",
                "sweep": [0.0, 60.0]},
        "outro": {"stratum": "S2", "path": "x.vst3", "drive_param": "gain"},
    }
    base.update(arms)
    return {"reference": "ref", "arms": base}


LEVELS = {"ref": [5.0, 10.0, 15.0], "outro": [0.1, 0.2, 0.3]}


def test_the_versioned_roster_loads():
    roster = load_roster(DEFAULT_ROSTER)
    assert roster.reference in roster.keys()
    assert roster.drive_levels >= 2


def test_the_roster_loads_without_levels_for_the_calibration():
    roster = load_roster(DEFAULT_ROSTER, levels_path=None)
    assert all(arm.levels == () for arm in roster.arms)


def test_the_reference_levels_are_the_common_scale():
    roster = parse_roster(_data(), LEVELS)
    assert roster.reference_levels() == (5.0, 10.0, 15.0)
    assert roster.drive_levels == 3


def test_arms_with_different_level_counts_are_refused():
    with pytest.raises(ValueError, match="mesmo numero de niveis"):
        parse_roster(_data(), {**LEVELS, "outro": [0.1, 0.2]})


def test_levels_out_of_order_are_refused():
    # Uma curva nao monotona pode inverter dois niveis vizinhos; o render nao aceita.
    with pytest.raises(ValueError, match="crescentes"):
        parse_roster(_data(), {**LEVELS, "outro": [0.1, 0.3, 0.2]})


def test_an_arm_missing_from_the_levels_file_is_refused():
    with pytest.raises(ValueError, match="calibrate"):
        parse_roster(_data(), {"ref": LEVELS["ref"]})


def test_a_reference_outside_the_roster_is_refused():
    with pytest.raises(ValueError, match="referencia"):
        parse_roster({**_data(), "reference": "nao-existe"})


def test_a_vst_arm_without_a_path_is_refused():
    with pytest.raises(ValueError, match="path"):
        parse_roster(_data(outro={"stratum": "S2", "drive_param": "g"}))


def test_a_pedalboard_arm_without_a_sweep_is_refused():
    with pytest.raises(ValueError, match="sweep"):
        parse_roster(_data(ref={"stratum": "S1", "backend": "pedalboard",
                                "drive_param": "drive_db"}))


def test_a_typo_in_a_field_name_is_refused_instead_of_ignored():
    with pytest.raises(ValueError, match="desconhecidos"):
        parse_roster(_data(outro={"stratum": "S2", "path": "x.vst3", "drive_param": "g",
                                  "fxied": {}}))


def test_levels_in_the_roster_are_refused():
    # Os niveis moraram no roster; um roster antigo tem de falhar, nao ser ignorado.
    with pytest.raises(ValueError, match="desconhecidos"):
        parse_roster(_data(outro={"stratum": "S2", "path": "x.vst3", "drive_param": "g",
                                  "levels": [1, 2, 3]}))


@pytest.mark.parametrize("missing", ["stratum", "drive_param"])
def test_a_required_field_missing_is_refused(missing):
    spec = {"stratum": "S2", "path": "x.vst3", "drive_param": "g"}
    del spec[missing]
    with pytest.raises(ValueError, match=missing):
        parse_roster(_data(outro=spec))


def test_the_levels_file_round_trips(tmp_path):
    data = {"descriptor": "crest_drop_db", "targets": [1.0, 2.0, 3.0],
            "arms": {key: {"levels": values, "unmatched": []}
                     for key, values in LEVELS.items()}}
    path = tmp_path / "levels.yaml"
    write_levels(path, data)
    assert load_levels(path) == data


def test_an_unknown_arm_is_refused_by_name():
    with pytest.raises(KeyError, match="nao esta no roster"):
        parse_roster(_data()).arm("z")


def test_a_plugin_that_returns_nan_fails_instead_of_writing_it():
    # Um plugin pode sair NaN sem erro (o Fuzz da VZtec saia); o render gravaria o wav assim.
    loaded = LoadedArm.__new__(LoadedArm)
    loaded.arm = parse_roster(_data()).arm("outro")
    loaded.stereo = False
    loaded.plugin = _NanPlugin()
    with pytest.raises(RuntimeError, match="NaN"):
        loaded.render(np.zeros((1, 64), dtype=np.float32), 44100, 0.5)


class _NanPlugin:
    parameters = {"gain": SimpleNamespace(min_value=0.0, max_value=1.0)}

    def __call__(self, audio, sr, reset):
        out = np.array(audio, dtype=np.float32)
        out[:, 32:] = np.nan
        return out
