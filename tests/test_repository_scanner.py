from pathlib import Path

from codeatlas.scanner.repository import scan_repository


def test_scan_repository_returns_file_metadata(tmp_path: Path):
    source_file = tmp_path / "src" / "main.py"
    source_file.parent.mkdir()
    source_file.write_text("print('hello')")

    ignored_file = tmp_path / ".venv" / "ignored.py"
    ignored_file.parent.mkdir()
    ignored_file.write_text("should not be scanned")

    files = scan_repository(str(tmp_path))

    assert len(files) == 1

    file = files[0]

    assert file.path == source_file
    assert file.relative_path == Path("src/main.py")
    assert file.extension == ".py"
    assert file.language == "python"
    assert file.size > 0
    assert file.content == "print('hello')"