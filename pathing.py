from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    doc_dir: Path
    markdown_path: Path
    image_dir: Path
    layout_pdf_path: Path
    det_markdown_path: Path


def build_output_paths(pdf_path: Path, input_root: Path, output_root: Path) -> OutputPaths:
    if input_root.is_file():
        relative_parent = Path(".")
    else:
        relative_parent = pdf_path.relative_to(input_root).parent
    doc_dir = (output_root / relative_parent / pdf_path.stem).resolve()
    image_dir = doc_dir / "images"
    markdown_path = doc_dir / f"{pdf_path.stem}.md"
    layout_pdf_path = doc_dir / f"{pdf_path.stem}_layouts.pdf"
    det_markdown_path = doc_dir / f"{pdf_path.stem}_det.mmd"
    return OutputPaths(
        doc_dir=doc_dir,
        markdown_path=markdown_path,
        image_dir=image_dir,
        layout_pdf_path=layout_pdf_path,
        det_markdown_path=det_markdown_path,
    )
