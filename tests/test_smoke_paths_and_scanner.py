from pathlib import Path

from pathing import build_output_paths
from scanner import collect_pdf_files


def test_collect_pdf_files_filters_and_sorts(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    # Use empty placeholder files. Scanner checks extension and existence only.
    (b / "z.pdf").write_bytes(b"")
    (a / "m.PDF").write_bytes(b"")
    (a / "ignore.txt").write_text("x", encoding="utf-8")

    result = collect_pdf_files(str(tmp_path))
    assert [x.name for x in result] == ["m.PDF", "z.pdf"]


def test_build_output_paths_mirrors_input_tree(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    pdf_path = input_root / "nested" / "doc.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"")

    paths = build_output_paths(pdf_path=pdf_path, input_root=input_root, output_root=output_root)

    expected_doc_dir = output_root / "nested" / "doc"
    assert paths.doc_dir == expected_doc_dir.resolve()
    assert paths.markdown_path.name == "doc.md"
    assert paths.image_dir.name == "images"
