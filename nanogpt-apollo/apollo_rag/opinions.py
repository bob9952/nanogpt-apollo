"""Export the selected Finance Ministry opinions for the separate RAG pilot."""

import argparse
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory


RECORDS = (
    "000501114-2025-10520-004-000-011-004",
    "001855973-2025-10520-004-000-011-004",
    "003002091-2025-10520-004-000-011-004",
)


def read_opinions(repository):
    results = {}
    for record in RECORDS:
        source = repository / "misljenja" / "ministarstvo-finansija" / "2025" / record
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata["kind"] != "opinion" or metadata["is_law"] is not False:
            raise ValueError(f"Expected a Ministry opinion for {record}")
        issued = date.fromisoformat(metadata["date"])
        pages = "–".join(map(str, metadata["pdf_pages"]))
        body = (source / "opinion.md").read_text(encoding="utf-8")
        if not body.strip():
            raise ValueError(f"Empty opinion text for {record}")
        header = (
            f"Mišljenje Ministarstva finansija Republike Srbije — {issued}\n"
            f"Broj mišljenja: {metadata['number']}\n"
            f"Naslov: {metadata['title']}\n"
            f"Originalni PDF: {metadata['source']}\n"
            f"Strane izvornog PDF-a: {pages}\n"
            f"Stranica biltena: {metadata['source_page']}\n"
            f"Zapis u serbia-law: {source.relative_to(repository).as_posix()}\n"
        )
        results[f"mfin-{record}.txt"] = f"{header}\n{body}"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    if output == repository or repository in output.parents:
        parser.error("--output must be outside the serbia-law checkout")
    if output.exists():
        parser.error("--output already exists; choose a new directory")

    try:
        opinions = read_opinions(repository)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temp:
            staged = Path(temp) / "opinions"
            staged.mkdir()
            for name, text in opinions.items():
                (staged / name).write_text(text, encoding="utf-8", newline="\n")
            staged.rename(output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Opinion export failed: {exc}\n")
    print(f"Exported {len(opinions)} Ministry opinions to {output}")


if __name__ == "__main__":
    main()
