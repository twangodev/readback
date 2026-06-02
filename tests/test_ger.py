from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from readback import store
from readback.fuse.ger import (
    LabelLike,
    build_prompt,
    canonical_callsigns,
    correct,
    transcript_grammar,
)
from readback.pipeline.ger import ger_partial_path, ger_path, ger_shard, run_ger
from readback.pipeline.layout import labels_path, meta_path

HEADING_HYPS = (
    "turn left heading two one zero",
    "turn left heading two one five",
    "turn right heading two one zero",
)


@dataclass
class FakeClient:
    reply: str = ""
    by_grammar: dict[bool, str] = field(default_factory=dict)
    calls: list[dict] = field(default_factory=list)

    def complete(self, prompt, *, choices=None, grammar=None):
        self.calls.append({"prompt": prompt, "choices": choices, "grammar": grammar})
        if self.by_grammar:
            return self.by_grammar[grammar is not None]
        return self.reply


def test_canonical_callsigns_expands_to_spoken_form():
    assert canonical_callsigns(["N172SP"]) == ["november one seven two sierra papa"]
    assert canonical_callsigns([""]) == []


def test_build_prompt_includes_every_hypothesis_and_callsign():
    prompt = build_prompt(list(HEADING_HYPS), HEADING_HYPS[0], ["N172SP", "DAL1234"])
    for hyp in HEADING_HYPS:
        assert hyp in prompt
    assert "november one seven two sierra papa" in prompt
    assert "delta" in prompt
    assert "CANDIDATES" in prompt
    assert "CONSENSUS" in prompt
    assert "CALLSIGNS" in prompt


def test_build_prompt_without_callsigns_omits_callsign_block():
    prompt = build_prompt(["cleared to land"], "cleared to land", [])
    assert "CALLSIGNS" not in prompt


def test_transcript_grammar_pins_single_callsign():
    grammar = transcript_grammar(["N172SP"])
    assert grammar is not None
    assert '"november one seven two sierra papa"' in grammar
    assert "root ::=" in grammar


def test_transcript_grammar_none_for_zero_or_many_callsigns():
    assert transcript_grammar([]) is None
    assert transcript_grammar(["N172SP", "DAL1234"]) is None


def test_correct_returns_model_text_and_marks_corrected():
    client = FakeClient(reply="Turn left heading two one zero.")
    label = LabelLike(
        utterance_id="u1",
        transcript="turn left heading two one five",
        voting_text=HEADING_HYPS,
        callsign_tail=None,
        callsign_matched=False,
        tails=(),
    )
    result = correct(client, label)
    assert result.text == "turn left heading two one zero"
    assert result.corrected is True
    assert result.callsign_enforced is False
    assert client.calls[0]["grammar"] is None


def test_correct_passes_grammar_when_callsign_matched():
    client = FakeClient(reply="november one seven two sierra papa cleared to land")
    label = LabelLike(
        utterance_id="u2",
        transcript="november one seven two sierra papa cleared to land",
        voting_text=("n172sp cleared to land",),
        callsign_tail="N172SP",
        callsign_matched=True,
        tails=("N172SP",),
    )
    result = correct(client, label)
    grammar = client.calls[0]["grammar"]
    assert grammar is not None
    assert "november one seven two sierra papa" in grammar
    assert result.callsign_enforced is True
    assert "november one seven two sierra papa" in result.text


def test_correct_snaps_near_miss_callsign_back_to_canonical():
    client = FakeClient(reply="november one seven two sierra hotel cleared to land")
    label = LabelLike(
        utterance_id="u3",
        transcript="november one seven two sierra papa cleared to land",
        voting_text=("n172sp cleared to land",),
        callsign_tail="N172SP",
        callsign_matched=True,
        tails=("N172SP",),
    )
    result = correct(client, label)
    assert "november one seven two sierra papa" in result.text
    assert result.callsign_enforced is True


def test_correct_falls_back_to_original_on_empty_reply():
    client = FakeClient(reply="")
    label = LabelLike(
        utterance_id="u4",
        transcript="cleared to land",
        voting_text=("cleared to land",),
        callsign_tail=None,
        callsign_matched=False,
        tails=(),
    )
    result = correct(client, label)
    assert result.text == "cleared to land"
    assert result.corrected is False


def _seed_run(run: Path) -> None:
    store.write_jsonl(
        meta_path(run, 0),
        [
            {"utterance_id": "s0", "tails": ["N172SP"]},
            {"utterance_id": "g0", "tails": []},
            {"utterance_id": "t0", "tails": []},
        ],
    )
    store.write_jsonl(
        labels_path(run, 0),
        [
            {
                "utterance_id": "s0",
                "transcript": "november one seven two sierra hotel cleared to land",
                "tier": "silver",
                "callsign_tail": "N172SP",
                "callsign_matched": True,
                "voting_text": ["n172sp cleared to land", "n172sh cleared to land"],
            },
            {
                "utterance_id": "g0",
                "transcript": "cleared to land",
                "tier": "gold",
                "callsign_tail": None,
                "callsign_matched": False,
                "voting_text": ["cleared to land"],
            },
            {
                "utterance_id": "t0",
                "transcript": "garbled",
                "tier": "tail",
                "callsign_tail": None,
                "callsign_matched": False,
                "voting_text": ["garbled"],
            },
        ],
    )


def test_ger_shard_corrects_only_silver_and_passes_through_rest(tmp_path):
    _seed_run(tmp_path)
    client = FakeClient(reply="november one seven two sierra papa cleared to land")
    index, total, corrected = ger_shard(tmp_path, client, 0)
    assert (index, total, corrected) == (0, 3, 1)

    rows = store.index_by(ger_path(tmp_path, 0), "utterance_id")
    assert "november one seven two sierra papa" in rows["s0"]["transcript"]
    assert rows["s0"]["corrected"] is True
    assert rows["s0"]["callsign_enforced"] is True
    assert rows["g0"]["transcript"] == "cleared to land"
    assert rows["g0"]["corrected"] is False
    assert rows["t0"]["transcript"] == "garbled"
    assert rows["t0"]["corrected"] is False


def test_ger_shard_calls_client_only_for_silver(tmp_path):
    _seed_run(tmp_path)
    client = FakeClient(reply="november one seven two sierra papa cleared to land")
    ger_shard(tmp_path, client, 0)
    assert len(client.calls) == 1


def test_run_ger_skips_already_corrected_shards(tmp_path):
    _seed_run(tmp_path)
    client = FakeClient(reply="november one seven two sierra papa cleared to land")
    run_ger(tmp_path, [0], client=client, log=lambda _: None)
    first = len(client.calls)
    logs: list[str] = []
    run_ger(tmp_path, [0], client=client, log=logs.append)
    assert len(client.calls) == first
    assert any("already corrected" in line for line in logs)


@dataclass
class CrashingClient:
    reply: str
    fail_after: int
    calls: list = field(default_factory=list)

    def complete(self, prompt, *, choices=None, grammar=None):
        if len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.append(prompt)
        return self.reply


def _seed_multi_silver(run: Path, n: int) -> None:
    store.write_jsonl(
        meta_path(run, 0),
        [{"utterance_id": f"s{i}", "tails": []} for i in range(n)],
    )
    store.write_jsonl(
        labels_path(run, 0),
        [
            {
                "utterance_id": f"s{i}",
                "transcript": "bravo charlie delta",
                "tier": "silver",
                "callsign_tail": None,
                "callsign_matched": False,
                "voting_text": ["bravo charlie delta", "bravo charlie echo"],
            }
            for i in range(n)
        ],
    )


def test_ger_shard_resumes_after_crash_without_recalling(tmp_path):
    _seed_multi_silver(tmp_path, 2)
    crasher = CrashingClient(reply="bravo charlie echo", fail_after=1)
    with pytest.raises(RuntimeError):
        ger_shard(tmp_path, crasher, 0)

    partial = ger_partial_path(tmp_path, 0)
    assert len(store.read_jsonl_recoverable(partial)) == 1
    assert not ger_path(tmp_path, 0).exists()

    resume = FakeClient(reply="bravo charlie echo")
    index, total, corrected = ger_shard(tmp_path, resume, 0)
    assert len(resume.calls) == 1
    assert (index, total, corrected) == (0, 2, 2)
    assert not partial.exists()
    rows = store.index_by(ger_path(tmp_path, 0), "utterance_id")
    assert all(rows[f"s{i}"]["transcript"] == "bravo charlie echo" for i in range(2))


def test_ger_shard_drops_truncated_wal_tail_and_reruns(tmp_path):
    _seed_multi_silver(tmp_path, 2)
    partial = ger_partial_path(tmp_path, 0)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        '{"utterance_id": "s0", "transcript": "bravo charlie echo", '
        '"corrected": true, "callsign_enforced": false}\n'
        '{"utterance_id": "s1", "transc'
    )
    client = FakeClient(reply="bravo charlie echo")
    index, total, corrected = ger_shard(tmp_path, client, 0)
    assert len(client.calls) == 1
    assert (index, total, corrected) == (0, 2, 2)
    assert not partial.exists()


def test_ger_shard_promotes_from_complete_wal_without_calls(tmp_path):
    _seed_multi_silver(tmp_path, 2)
    partial = ger_partial_path(tmp_path, 0)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(
        '{"utterance_id": "s0", "transcript": "bravo charlie echo", '
        '"corrected": true, "callsign_enforced": false}\n'
        '{"utterance_id": "s1", "transcript": "bravo charlie echo", '
        '"corrected": true, "callsign_enforced": false}\n'
    )
    client = FakeClient(reply="UNUSED")
    index, total, corrected = ger_shard(tmp_path, client, 0)
    assert len(client.calls) == 0
    assert (index, total, corrected) == (0, 2, 2)
    assert not partial.exists()
    rows = store.index_by(ger_path(tmp_path, 0), "utterance_id")
    assert rows["s0"]["transcript"] == "bravo charlie echo"


@pytest.mark.slow
def test_vllm_client_against_live_server():
    from readback.serve import VllmClient, VllmConfig, is_ready

    config = VllmConfig()
    if not is_ready(config):
        pytest.skip("no vLLM server reachable")
    client = VllmClient(base_url=config.base_url)
    label = LabelLike(
        utterance_id="live",
        transcript="november one seven two sierra hotel cleared to land",
        voting_text=(
            "november one seven two sierra papa cleared to land",
            "november one seven two sierra hotel cleared land",
        ),
        callsign_tail="N172SP",
        callsign_matched=True,
        tails=("N172SP",),
    )
    result = correct(client, label)
    assert "november one seven two sierra papa" in result.text
