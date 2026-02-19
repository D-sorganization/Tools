"""Tests for the file renaming engine."""

from pdf_renamer.renamer import Renamer


def test_generate_new_filename() -> None:
    renamer = Renamer()
    name = renamer.generate_new_filename("John Doe", "introduction to python")
    assert name == "Doe - Introduction to Python.pdf"

    name = renamer.generate_new_filename("Jane Smith", "The art of war")
    assert name == "Smith - The Art of War.pdf"


def test_renamer_collision_logic(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    # Setup
    renamer = Renamer(dry_run=False)

    # Create a fake existing file
    (tmp_path / "Doe - Test.pdf").touch()

    # Create the file to be renamed
    source_file = tmp_path / "source.pdf"
    source_file.touch()

    # Should result in "Doe - Test_1.pdf"
    renamer.rename_file(source_file, "Doe - Test.pdf")

    assert (tmp_path / "Doe - Test_1.pdf").exists()
    assert not source_file.exists()
