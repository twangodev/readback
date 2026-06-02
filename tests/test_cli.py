from __future__ import annotations

from typer.testing import CliRunner

from readback import store
from readback.cli import app
from readback.pipeline.layout import hyps_path, labels_path, meta_path

runner = CliRunner()
GOLD_TEXT = "november one seven two sierra papa descend"


def _seed(run):
    store.write_jsonl(
        meta_path(run, 0), [{"utterance_id": "u0", "airport": "k", "tails": ["N172SP"]}]
    )
    for model in ("a", "b", "c", "d"):
        store.write_jsonl(
            hyps_path(run, model, 0),
            [
                {
                    "utterance_id": "u0",
                    "text": GOLD_TEXT,
                    "confidence": None,
                    "no_speech": 0.0,
                }
            ],
        )


def test_cli_fuse_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(
        app,
        [
            "fuse",
            "--run",
            str(tmp_path),
            "--voters",
            "a,b,c",
            "--weights",
            "1,1,2",
            "--advisory",
            "d",
        ],
    )
    assert result.exit_code == 0, result.output
    labels = list(store.read_jsonl(labels_path(tmp_path, 0)))
    assert labels[0]["tier"] == "gold"


def test_cli_fuse_rejects_misaligned_weights(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(
        app, ["fuse", "--run", str(tmp_path), "--voters", "a,b,c", "--weights", "1,2"]
    )
    assert result.exit_code != 0
