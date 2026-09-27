"""Sidecar do POC II e a checagem do pareamento entre arms."""
from __future__ import annotations

import itertools

import pytest

from gefx.disent.sidecar import (
    EFFECT_FOLDER,
    SIDECAR_COLUMNS,
    SIDECAR_FILENAME,
    DisentRecord,
    arm_dirs,
    read_dataset,
    read_sidecar,
    validate_pairing,
    write_sidecar,
)

CONTENTS = ["c0", "c1"]
DRIVES = [0, 1, 2]


def _record(arm, content, config_index, drive, split="train"):
    return DisentRecord(
        file_name=f"{content}__d{drive}__{config_index:05d}.wav",
        arm=arm,
        stratum="S1",
        content_id=content,
        source_audio_id=f"{content}.wav",
        segment_start=1000,
        config_index=config_index,
        config_key=f"d{drive}",
        drive_level=drive,
        drive_knob=1.0 + drive,
        drive_db_equivalente=15.0 + drive,
        split=split,
    )


def _build(root, arms=("m0", "m1"), write_wavs=True, mutate=None):
    """Monta um dataset sintetico usando o proprio codigo de escrita."""
    for arm in arms:
        records = [
            _record(arm, content, drive, drive)
            for content in CONTENTS
            for drive in DRIVES
        ]
        if mutate is not None:
            records = mutate(arm, records)
        folder = root / arm / EFFECT_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        if write_wavs:
            for record in records:
                (folder / record.file_name).touch()
        write_sidecar(root / arm / SIDECAR_FILENAME, records)
    return root


def test_columns_match_the_dataclass_fields():
    record = _record("m0", "c0", 0, 0, 0)
    assert list(record.as_row()) == SIDECAR_COLUMNS


def test_write_then_read_round_trips(tmp_path):
    root = _build(tmp_path / "ds")
    frame = read_sidecar(root / "m0")
    assert list(frame.columns) == SIDECAR_COLUMNS
    assert len(frame) == len(CONTENTS) * len(DRIVES)
    assert set(frame["arm"]) == {"m0"}


def test_read_dataset_concatenates_every_arm(tmp_path):
    root = _build(tmp_path / "ds")
    frame = read_dataset(root)
    assert set(frame["arm"]) == {"m0", "m1"}
    assert len(frame) == 2 * len(CONTENTS) * len(DRIVES)


def test_read_sidecar_reports_a_missing_file(tmp_path):
    (tmp_path / "vazio").mkdir()
    with pytest.raises(FileNotFoundError, match="sidecar ausente"):
        read_sidecar(tmp_path / "vazio")


def test_arm_dirs_ignores_directories_without_a_sidecar(tmp_path):
    root = _build(tmp_path / "ds")
    (root / "cache_solto").mkdir()
    assert [path.name for path in arm_dirs(root)] == ["m0", "m1"]


# --- o invariante do pareamento -----------------------------------------------
def test_validate_accepts_a_crossed_grid(tmp_path):
    root = _build(tmp_path / "ds")
    assert validate_pairing(root) == {
        "arms": 2,
        "rows_per_arm": len(CONTENTS) * len(DRIVES),
        "contents": len(CONTENTS),
        "configs": len(DRIVES),
    }


def test_validate_catches_a_render_missing_in_one_arm(tmp_path):
    # Sem o pareamento o oraculo compararia renders que diferem em mais de uma
    # coisa, e o alvo exato da troca deixaria de existir para alguns pares.
    def drop_one(arm, records):
        return records[:-1] if arm == "m1" else records

    root = _build(tmp_path / "ds", mutate=drop_one)
    with pytest.raises(ValueError, match="nao esta pareado"):
        validate_pairing(root)


def test_validate_catches_a_factor_that_disagrees_between_arms(tmp_path):
    def shift_split(arm, records):
        if arm != "m1":
            return records
        head, *rest = records
        return [DisentRecord(**{**head.as_row(), "split": "query"}), *rest]

    root = _build(tmp_path / "ds", mutate=shift_split)
    with pytest.raises(ValueError, match="difere .* na coluna 'split'"):
        validate_pairing(root)


def test_validate_catches_a_sidecar_row_without_its_wav(tmp_path):
    root = _build(tmp_path / "ds", write_wavs=False)
    with pytest.raises(ValueError, match="sem arquivo"):
        validate_pairing(root)


def test_validate_refuses_an_empty_root(tmp_path):
    (tmp_path / "vazio").mkdir()
    with pytest.raises(FileNotFoundError, match="nenhum arm"):
        validate_pairing(tmp_path / "vazio")
