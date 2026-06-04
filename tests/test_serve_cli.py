from __future__ import annotations

from typer.testing import CliRunner

from readback.cli import app

runner = CliRunner()


def test_serve_rejects_missing_web_dir(tmp_path):
    result = runner.invoke(
        app,
        ["serve", "--run", str(tmp_path), "--web", str(tmp_path / "nope")],
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
