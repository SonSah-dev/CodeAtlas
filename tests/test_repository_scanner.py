from pathlib import Path

from codeatlas.scanner.repository import scan_repository


def test_scan_repository_ignores_generated_directories(tmp_path: Path):
    source_file = tmp_path / "src" / "main.py"
    source_file.parent.mkdir()
    source_file.write_text("print('hello')")

    ignored_file = tmp_path / ".venv" / "ignored.py"
    ignored_file.parent.mkdir()
    ignored_file.write_text("should not be scanned")

    files = scan_repository(str(tmp_path))

    assert source_file in files
    assert ignored_file not in files