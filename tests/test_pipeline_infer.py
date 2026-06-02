from __future__ import annotations

import numpy as np

from readback import store
from readback.data import Clip
from readback.models.base import Audio, Hypothesis
from readback.models.registry import ModelSpec
from readback.pipeline.infer import run_infer
from readback.pipeline.layout import hyps_path, meta_path

SHARDS = {
    0: [("u0", ("N1",)), ("u1", ())],
    1: [("u2", ("N2",))],
}


class FakeSource:
    def list_indices(self):
        return sorted(SHARDS)

    def meta(self, index):
        return [
            {"utterance_id": utt, "airport": "kxxx", "tails": list(tails)}
            for utt, tails in SHARDS[index]
        ]

    def clips(self, index):
        return [
            Clip(utt, Audio(np.zeros(index + 1, np.float32), 16000), tails, "kxxx")
            for utt, tails in SHARDS[index]
        ]


class FakeModel:
    def __init__(self, name, batches):
        self.name = name
        self._batches = batches

    def transcribe(self, audios, bias_terms=None):
        self._batches.append(len(audios))
        return [
            Hypothesis(text=f"{self.name}:{a.array.shape[0]}", no_speech=0.0)
            for a in audios
        ]


def _specs():
    return {"m1": ModelSpec("m1", "fake", "ref"), "m2": ModelSpec("m2", "fake", "ref")}


def _build_factory(loaded, batches):
    def build(spec):
        loaded.append(spec.name)
        return FakeModel(spec.name, batches)

    return build


def test_run_infer_writes_meta_and_hyps(tmp_path):
    loaded, batches = [], []
    run_infer(
        FakeSource(),
        _specs(),
        ["m1", "m2"],
        [0, 1],
        tmp_path,
        build_model=_build_factory(loaded, batches),
        release=lambda: None,
        log=lambda _: None,
    )
    assert loaded == ["m1", "m2"]
    assert [r["utterance_id"] for r in store.read_jsonl(meta_path(tmp_path, 0))] == [
        "u0",
        "u1",
    ]
    rows = list(store.read_jsonl(hyps_path(tmp_path, "m1", 0)))
    assert rows[0] == {
        "utterance_id": "u0",
        "text": "m1:1",
        "confidence": None,
        "no_speech": 0.0,
    }


def test_run_infer_skips_when_complete(tmp_path):
    args = (FakeSource(), _specs(), ["m1", "m2"], [0, 1], tmp_path)
    run_infer(
        *args,
        build_model=_build_factory([], []),
        release=lambda: None,
        log=lambda _: None,
    )

    loaded, batches = [], []
    run_infer(
        *args,
        build_model=_build_factory(loaded, batches),
        release=lambda: None,
        log=lambda _: None,
    )
    assert loaded == []
    assert batches == []


def test_run_infer_continues_when_a_model_fails(tmp_path):
    batches: list[int] = []

    def build(spec):
        if spec.name == "m1":
            raise RuntimeError("cudnn boom")
        return FakeModel(spec.name, batches)

    logs: list[str] = []
    run_infer(
        FakeSource(),
        _specs(),
        ["m1", "m2"],
        [0, 1],
        tmp_path,
        build_model=build,
        release=lambda: None,
        log=logs.append,
    )
    assert any("FAILED m1" in line for line in logs)
    assert not hyps_path(tmp_path, "m1", 0).exists()
    assert hyps_path(tmp_path, "m2", 0).exists()
    assert hyps_path(tmp_path, "m2", 1).exists()


def test_run_infer_resumes_only_missing_shards(tmp_path):
    args = (FakeSource(), _specs(), ["m1", "m2"], [0, 1], tmp_path)
    run_infer(
        *args,
        build_model=_build_factory([], []),
        release=lambda: None,
        log=lambda _: None,
    )

    hyps_path(tmp_path, "m1", 1).unlink()
    loaded, batches = [], []
    run_infer(
        *args,
        build_model=_build_factory(loaded, batches),
        release=lambda: None,
        log=lambda _: None,
    )
    assert loaded == ["m1"]
    assert batches == [1]
