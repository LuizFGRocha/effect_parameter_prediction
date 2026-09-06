"""Sidecar `metadata.csv`: escrita, leitura, upsert e checagem de integridade.

`validate_sidecar_integrity` roda no inicio de todo treino e e a checagem de
consistencia de facto do dataset — um teste por branch de erro, mais o que ela
deliberadamente nao confere.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from gefx.effects.catalog import EFFECT_PARAMETER_RANGES

from gefx.data.metadata import (
    METADATA_COLUMNS,
    METADATA_FILENAME,
    append_metadata_csv,
    chain_folder,
    ensure_chain_key,
    list_chain_keys,
    read_metadata,
    target_lookup,
    validate_sidecar_integrity,
    write_metadata_csv,
)


def test_as_row_serializes_the_four_json_columns(make_record):
    row = make_record(["distortion"], "a.wav", [0.25]).as_row()

    assert list(row) == METADATA_COLUMNS
    assert json.loads(row["effect_order"]) == ["distortion"]
    assert json.loads(row["normalized_parameter_vector"]) == [0.25]
    assert json.loads(row["raw_parameter_dict"]) == {"distortion": {"drive_db": 13.75}}


def test_as_row_preserves_catalog_order(make_record):
    # Uma convencao so: a ordem do catalogo, a mesma do sufixo binario do nome do
    # arquivo e a mesma do vetor alvo.
    row = make_record(["distortion", "chorus"], "a.wav", [0.0, 0.5, 1.0]).as_row()

    assert list(json.loads(row["effect_presence"])) == list(EFFECT_PARAMETER_RANGES)
    assert list(json.loads(row["raw_parameter_dict"])) == ["distortion", "chorus"]
    assert json.loads(row["normalized_parameter_vector"]) == [0.0, 0.5, 1.0]


def test_as_row_legacy_sorts_the_two_redundant_columns(make_record):
    # `legacy=True` reproduz o sidecar dos datasets ja renderizados.
    row = make_record(["distortion", "chorus"], "a.wav", [0.0, 0.5, 1.0]).as_row(legacy=True)

    assert list(json.loads(row["effect_presence"])) == sorted(EFFECT_PARAMETER_RANGES)
    assert list(json.loads(row["raw_parameter_dict"])) == ["chorus", "distortion"]
    # A ordem que *e* informacao nao muda com a flag.
    assert json.loads(row["effect_order"]) == ["distortion", "chorus"]
    assert json.loads(row["normalized_parameter_vector"]) == [0.0, 0.5, 1.0]


def test_write_metadata_csv_forwards_legacy(make_record, tmp_path):
    records = [make_record(["distortion"], "a.wav", [0.25])]
    for legacy, expected in [(False, list(EFFECT_PARAMETER_RANGES)), (True, sorted(EFFECT_PARAMETER_RANGES))]:
        path = tmp_path / f"{legacy}.csv"
        write_metadata_csv(path, records, legacy=legacy)
        column = pd.read_csv(path).loc[0, "effect_presence"]
        assert list(json.loads(column)) == expected


def test_write_then_read_round_trip(make_record, tmp_path):
    records = [
        make_record(["distortion"], "a.wav", [0.1]),
        make_record(["chorus"], "b.wav", [0.2, 0.3]),
    ]
    write_metadata_csv(tmp_path / METADATA_FILENAME, records)

    frame = read_metadata(tmp_path)
    assert list(frame.columns) == METADATA_COLUMNS
    assert frame["file_name"].tolist() == ["a.wav", "b.wav"]
    assert frame["chain_key"].tolist() == ["distortion", "chorus"]


def test_read_metadata_missing_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing metadata sidecar"):
        read_metadata(tmp_path)


def test_ensure_chain_key_derives_from_effect_order():
    # Sidecars antigos nao tinham a coluna; ela e reconstruida na leitura.
    legacy = pd.DataFrame({"effect_order": [json.dumps(["distortion", "chorus"])]})
    repaired = ensure_chain_key(legacy)

    assert repaired["chain_key"].tolist() == ["distortion__chorus"]
    assert "chain_key" not in legacy.columns  # nao muta a entrada


def test_ensure_chain_key_leaves_modern_frames_untouched():
    frame = pd.DataFrame({"chain_key": ["ja_existe"], "effect_order": ["[]"]})
    assert ensure_chain_key(frame) is frame


def test_target_lookup_filters_by_chain(make_record, tmp_path):
    write_metadata_csv(
        tmp_path / METADATA_FILENAME,
        [
            make_record(["distortion"], "a.wav", [0.1]),
            make_record(["chorus"], "b.wav", [0.2, 0.3]),
        ],
    )
    lookup = target_lookup(read_metadata(tmp_path), "chorus")
    assert lookup == {"b.wav": [0.2, 0.3]}


def test_target_lookup_collapses_duplicate_file_names(make_record, tmp_path):
    # CARACTERIZACAO: o dict de saida guarda a ultima ocorrencia sem avisar. Como
    # os nomes sao unicos por construcao, isso so aparece se o sidecar for
    # concatenado errado.
    write_metadata_csv(
        tmp_path / METADATA_FILENAME,
        [
            make_record(["distortion"], "a.wav", [0.1]),
            make_record(["distortion"], "a.wav", [0.9]),
        ],
    )
    assert target_lookup(read_metadata(tmp_path), "distortion") == {"a.wav": [0.9]}


def test_list_chain_keys_is_sorted_and_unique(make_record, tmp_path):
    write_metadata_csv(
        tmp_path / METADATA_FILENAME,
        [
            make_record(["chorus"], "b.wav", [0.2, 0.3]),
            make_record(["distortion"], "a.wav", [0.1]),
            make_record(["distortion"], "c.wav", [0.4]),
        ],
    )
    assert list_chain_keys(tmp_path) == ["chorus", "distortion"]


def test_chain_folder_requires_the_directory(tmp_path):
    (tmp_path / "distortion").mkdir()
    assert chain_folder(tmp_path, "distortion").name == "distortion"
    with pytest.raises(FileNotFoundError, match="Missing chain folder"):
        chain_folder(tmp_path, "nao_existe")


def test_append_metadata_csv_replaces_rows_with_the_same_file_name(make_record, tmp_path):
    # `cross-impl render` reescreve o mesmo braco varias vezes; o upsert e o que
    # torna isso idempotente.
    append_metadata_csv(
        tmp_path,
        [
            make_record(["distortion"], "a.wav", [0.1]).as_row(),
            make_record(["distortion"], "b.wav", [0.2]).as_row(),
        ],
    )
    append_metadata_csv(
        tmp_path,
        [
            make_record(["distortion"], "b.wav", [0.9]).as_row(),
            make_record(["distortion"], "c.wav", [0.3]).as_row(),
        ],
    )

    frame = pd.read_csv(tmp_path / METADATA_FILENAME)
    assert frame["file_name"].tolist() == ["a.wav", "b.wav", "c.wav"]
    lookup = dict(zip(frame["file_name"], frame["normalized_parameter_vector"].map(json.loads)))
    assert lookup["b.wav"] == [0.9]


# --- validate_sidecar_integrity ---------------------------------------------
def test_validate_happy_path_returns_wav_counts(fake_dataset):
    root = fake_dataset(chains=[["distortion"], ["chorus"]], samples=3)
    assert validate_sidecar_integrity(root) == {"chorus": 3, "distortion": 3}


def test_validate_rejects_missing_column(fake_dataset):
    root = fake_dataset()
    path = root / METADATA_FILENAME
    pd.read_csv(path).drop(columns=["random_seed"]).to_csv(path, index=False)

    with pytest.raises(RuntimeError, match=r"Missing required metadata columns: \['random_seed'\]"):
        validate_sidecar_integrity(root)


def test_validate_tolerates_a_legacy_sidecar_without_chain_key(fake_dataset):
    # `chain_key` e a unica coluna reconstruivel, e `read_metadata` a reconstroi
    # antes da checagem.
    root = fake_dataset()
    path = root / METADATA_FILENAME
    pd.read_csv(path).drop(columns=["chain_key"]).to_csv(path, index=False)

    assert validate_sidecar_integrity(root) == {"chorus": 3, "distortion": 3}


def test_validate_rejects_wav_count_mismatch(fake_dataset):
    root = fake_dataset()
    (root / "distortion" / "intruso.wav").touch()

    with pytest.raises(RuntimeError, match="wav/metadata mismatch wav=4 rows=3"):
        validate_sidecar_integrity(root)


def test_validate_rejects_wav_without_a_metadata_row(fake_dataset):
    root = fake_dataset()
    wav = sorted((root / "distortion").glob("*.wav"))[0]
    wav.rename(wav.with_name("orfao.wav"))

    with pytest.raises(RuntimeError, match="metadata missing for wav orfao.wav"):
        validate_sidecar_integrity(root)


@pytest.mark.parametrize("bad_value", [1.5, -0.1])
def test_validate_rejects_normalized_values_outside_the_unit_interval(fake_dataset, bad_value):
    def mutate(records):
        records[0].normalized_parameter_vector = [bad_value]
        return records

    root = fake_dataset(chains=[["distortion"]], mutate=mutate)
    with pytest.raises(RuntimeError, match=r"normalized value outside \[0,1\]"):
        validate_sidecar_integrity(root)


@pytest.mark.parametrize("edge_value", [0.0, 1.0])
def test_validate_accepts_the_interval_endpoints(fake_dataset, edge_value):
    def mutate(records):
        records[0].normalized_parameter_vector = [edge_value]
        return records

    root = fake_dataset(chains=[["distortion"]], mutate=mutate)
    assert validate_sidecar_integrity(root) == {"distortion": 3}


def test_validate_rejects_a_vector_whose_length_does_not_match_the_chain(fake_dataset):
    # `distortion` preve 1 parametro. Antes, um vetor de 3 valores em [0,1]
    # passava e o erro so aparecia no treino, como incompatibilidade de forma
    # entre y e a saida do modelo.
    def mutate(records):
        records[0].normalized_parameter_vector = [0.1, 0.2, 0.3]
        return records

    root = fake_dataset(chains=[["distortion"]], mutate=mutate)
    with pytest.raises(RuntimeError, match="expected 1 normalized values, got 3"):
        validate_sidecar_integrity(root)


def test_validate_rejects_effect_order_that_contradicts_the_chain_key(fake_dataset):
    def mutate(records):
        records[0].effect_order = ["chorus", "reverb"]
        return records

    root = fake_dataset(chains=[["distortion"]], mutate=mutate)
    with pytest.raises(RuntimeError, match="does not match the chain key"):
        validate_sidecar_integrity(root)


def test_validate_rejects_an_inconsistent_chain_length(fake_dataset):
    def mutate(records):
        records[0].chain_length = 99
        return records

    root = fake_dataset(chains=[["distortion"]], mutate=mutate)
    with pytest.raises(RuntimeError, match="chain_length 99 != 1"):
        validate_sidecar_integrity(root)


def test_validate_rejects_a_chain_key_with_an_effect_outside_the_catalog(fake_dataset):
    # Uma cadeia removida ou renomeada no catalogo deixa o dataset orfao; antes
    # isso passava batido ate a hora de montar o vetor alvo.
    root = fake_dataset(chains=[["distortion"]])
    (root / "distortion").rename(root / "distortion__fantasma")

    path = root / METADATA_FILENAME
    frame = pd.read_csv(path)
    frame["chain_key"] = "distortion__fantasma"
    frame["effect_order"] = json.dumps(["distortion", "fantasma"])
    frame["chain_length"] = 2
    frame.to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="effect not in catalog: fantasma"):
        validate_sidecar_integrity(root)
