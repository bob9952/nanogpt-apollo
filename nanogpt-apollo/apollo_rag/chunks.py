"""Passages with original line/column mappings and exact E5 input accounting."""

import hashlib
import json
import re

from .inputs import clean_manual, original_lines


def embedding_text(text: str, section: str | None) -> str:
    """Phase 2 must embed this exact string, including E5's passage prefix."""
    return "passage: " + (section + "\n\n" if section else "") + text


def heading(text):
    stripped = text.strip()
    if not stripped or len(stripped) > 160:
        return False
    return bool(
        re.match(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+\S)", stripped)
        or (len(stripped) >= 4 and stripped.isupper() and any(char.isalpha() for char in stripped))
        or re.match(r"^(?:create\s+(?:or\s+replace\s+)?(?:package|procedure|function|trigger)|"
                    r"(?:procedure|function)\s+\w+)", stripped, re.I)
    )


def paragraphs(document):
    lines = original_lines(document)
    removed = []
    if document.kind == "manual":
        lines, removed = clean_manual(lines)
    result, pending = [], []
    section = None

    def flush():
        if not pending:
            return
        text, spans = "", []
        for line in pending:
            part = line.text.strip() if document.kind == "manual" else line.text
            column_start = line.column_start + (line.text.find(part) if part else 0)
            if text:
                text += " " if document.kind == "manual" else "\n"
            start = len(text)
            text += part
            spans.append({
                "text_start": start, "text_end": len(text),
                "line_start": line.number, "line_end": line.number,
                "column_start": column_start, "column_end": column_start + len(part),
                "page": line.page,
            })
        result.append((text, spans, section))
        pending.clear()

    for line in lines:
        if not line.text.strip():
            flush()
            continue
        if pending and line.page != pending[-1].page:
            flush()
        if heading(line.text):
            flush()
            section = line.text.strip()
        pending.append(line)
    flush()
    return result, removed


def safe_section(section, tokenizer, max_tokens):
    if section is None:
        return None
    # Reserve at least half the budget for body text, measured by the real tokenizer.
    while section and tokenizer.count(embedding_text("", section)) > max_tokens // 2:
        section = section[:len(section) // 2]
    return section or None


def passage_windows(text, section, tokenizer, max_tokens=384, overlap_tokens=32):
    """Keep all characters; overlap uses true tokenizer counts, never estimates."""
    start = 0
    while start < len(text):
        low, high = start + 1, min(len(text), start + max_tokens * 32)
        best = start
        while low <= high:
            mid = (low + high) // 2
            if tokenizer.count(embedding_text(text[start:mid], section)) <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best == start:
            raise ValueError("Token budget cannot fit a source character plus the passage prefix and section")
        end = best
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + (end - start) // 2:
                end = boundary + 1
        yield start, end
        if end == len(text):
            break
        if not overlap_tokens:
            start = end
            continue
        low, high, overlap_start = start + 1, end, end
        while low <= high:
            mid = (low + high) // 2
            if tokenizer.count(text[mid:end]) <= overlap_tokens:
                overlap_start = mid
                high = mid - 1
            else:
                low = mid + 1
        start = max(start + 1, overlap_start)


def clipped_spans(spans, start, end):
    result = []
    for span in spans:
        left, right = max(start, span["text_start"]), min(end, span["text_end"])
        if left >= right:
            continue
        result.append({
            **span, "text_start": left - start, "text_end": right - start,
            "column_start": span["column_start"] + left - span["text_start"],
            "column_end": span["column_start"] + right - span["text_start"],
        })
    return result


def document_chunks(document, tokenizer, svn, max_tokens=384, overlap_tokens=32):
    passages, removed = paragraphs(document)
    chunks = []
    for paragraph_index, (text, spans, original_section) in enumerate(passages):
        section = safe_section(original_section, tokenizer, max_tokens)
        for start, end in passage_windows(text, section, tokenizer, max_tokens, overlap_tokens):
            body = text[start:end]
            mapped = clipped_spans(spans, start, end)
            if not mapped:
                continue
            embedding = embedding_text(body, section)
            count = tokenizer.count(embedding)
            if count > max_tokens:
                raise ValueError("Tokenizer produced a passage over the configured token limit")
            chunk = {
                "kind": document.kind, "path": document.path,
                "source_hash": document.source_hash,
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "line_start": mapped[0]["line_start"], "line_end": mapped[-1]["line_end"],
                "page_start": mapped[0]["page"], "page_end": mapped[-1]["page"],
                "section": original_section, "embedding_section": section,
                "paragraph_index": paragraph_index, "passage_start": start, "passage_end": end,
                "text": body, "embedding_text": embedding, "token_count": count,
                "spans": mapped, "svn": svn,
            }
            identity = [document.kind, document.path, document.source_hash,
                        paragraph_index, start, end, embedding]
            chunk["id"] = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
            chunks.append(chunk)
    return chunks, removed
