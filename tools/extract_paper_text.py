#!/usr/bin/env python3
"""Extract page-marked PDF text or paragraph/table text from a DOCX file."""

import argparse
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = "{%s}" % WORD_NAMESPACE


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(source: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(source))
    sections = []
    for page_number, page in enumerate(reader.pages, 1):
        text = normalize(page.extract_text() or "")
        sections.append(f"\n===== PDF PAGE {page_number} =====\n{text}")
    return "\n".join(sections).strip() + "\n"


def element_text(element) -> str:
    chunks = []
    for node in element.iter():
        if node.tag == W + "t" and node.text:
            chunks.append(node.text)
        elif node.tag in (W + "tab",):
            chunks.append("\t")
        elif node.tag in (W + "br", W + "cr"):
            chunks.append("\n")
    return normalize("".join(chunks))


def extract_docx(source: Path) -> str:
    with zipfile.ZipFile(source) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    body = document.find(W + "body")
    if body is None:
        return ""
    output = []
    paragraph_number = 0
    table_number = 0
    for child in body:
        if child.tag == W + "p":
            text = element_text(child)
            if text:
                paragraph_number += 1
                output.append(f"[P{paragraph_number}] {text}")
        elif child.tag == W + "tbl":
            table_number += 1
            output.append(f"\n===== DOCX TABLE {table_number} =====")
            for row in child.findall(".//" + W + "tr"):
                cells = [element_text(cell) for cell in row.findall(W + "tc")]
                if any(cells):
                    output.append(" | ".join(cells))
    return "\n".join(output).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    source = Path(args.source)
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        text = extract_pdf(source)
    elif suffix == ".docx":
        text = extract_docx(source)
    else:
        raise ValueError(f"unsupported document type: {suffix}")
    destination.write_text(text, encoding="utf-8")
    print(f"{source.name}: {len(text)} characters -> {destination}")


if __name__ == "__main__":
    main()
