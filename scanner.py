from pathlib import Path
from typing import List


def collect_pdf_files(input_path: str) -> List[Path]:
    target = Path(input_path)
    if not target.exists():
        raise FileNotFoundError(f"Input path not found: {target}")
    if target.is_file():
        if target.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {target}")
        return [target.resolve()]

    pdfs = []
    for path in target.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path.resolve())
    return sorted(
        pdfs,
        key=lambda p: (
            str(p.parent).lower(),
            p.name.lower(),
        ),
    )
