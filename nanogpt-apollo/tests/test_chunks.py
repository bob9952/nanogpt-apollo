from apollo_rag.chunks import document_chunks, embedding_text, paragraphs
from apollo_rag.inputs import SourceDocument, sha256


def document(text, kind="manual"):
    return SourceDocument(kind, "guide.txt", "/local/guide.txt",
                          sha256(text.encode()), "utf-8", text)


def test_header_toc_cleanup_preserves_stock_account_instructions_and_lines(tokenizer):
    text = (
        "Apollo ERP User Guide\nContents\nStock accounts ......... 4\n\n"
        "1. Stock accounts\nSelect the stock account before posting.\n"
        "The selected account receives the stock movement.\n\nPage 1\n\f"
        "Apollo ERP User Guide\n\n2. Warehouse\nOpen the warehouse document.\n"
        "Choose its organisation.\n\nPage 2\n\f"
        "Apollo ERP User Guide\n\n3. Posting\nReview the document totals.\n"
        "Post the prepared document.\n\nPage 3"
    )
    source = document(text)
    chunks, removed = document_chunks(source, tokenizer, {"availability": "not-applicable"})
    body = "\n".join(chunk["text"] for chunk in chunks)
    assert "Select the stock account before posting." in body
    assert "The selected account receives the stock movement." in body
    assert "........." not in body
    assert "Apollo ERP User Guide" not in body
    assert {item["reason"] for item in removed} >= {"repeated-header-footer", "table-of-contents-entry"}
    guidance = next(chunk for chunk in chunks if "Select the stock account" in chunk["text"])
    assert guidance["line_start"] <= 6 <= guidance["line_end"]
    assert guidance["page_start"] == 1
    for chunk in chunks:
        for span in chunk["spans"]:
            original = text.split("\n")[span["line_start"] - 1]
            assert chunk["text"][span["text_start"]:span["text_end"]] == original[span["column_start"]:span["column_end"]]


def test_long_passage_retains_every_character_with_bounded_overlap_and_exact_count(tokenizer):
    source = document("Stock account detail " * 200 + "last instruction.")
    chunks, _ = document_chunks(source, tokenizer, {}, max_tokens=80, overlap_tokens=12)
    assert len(chunks) > 10
    rebuilt = ""
    previous_end = 0
    for chunk in chunks:
        assert chunk["token_count"] == tokenizer.count(chunk["embedding_text"])
        assert chunk["token_count"] <= 80
        assert chunk["embedding_text"] == embedding_text(chunk["text"], chunk["embedding_section"])
        assert chunk["passage_start"] <= previous_end
        overlap = previous_end - chunk["passage_start"]
        if overlap:
            assert tokenizer.count(chunk["text"][:overlap]) <= 12
        rebuilt += chunk["text"][overlap:]
        previous_end = chunk["passage_end"]
    assert rebuilt == source.text
    assert all(left["passage_end"] > right["passage_start"] for left, right in zip(chunks, chunks[1:]))


def test_comments_and_indentation_preserved_and_ids_repeat(tokenizer):
    source = document("-- stock movement comment\n  procedure post_stock is\n    begin\n    null;\n  end;", "source")
    chunks, _ = document_chunks(source, tokenizer, {"revision": "177", "local_status": "modified"})
    assert "-- stock movement comment" in "\n".join(chunk["text"] for chunk in chunks)
    assert any("    null;" in chunk["text"] for chunk in chunks)
    assert [chunk["id"] for chunk in chunks] == [chunk["id"] for chunk in document_chunks(source, tokenizer, {"revision": "177", "local_status": "modified"})[0]]
    assert all(chunk["svn"]["local_status"] == "modified" for chunk in chunks)


def test_long_unicode_heading_has_budgeted_embedding_context_without_losing_body(tokenizer):
    source = document("# " + "Ž" * 120 + "\n\n" + "汉字 čć " * 100)
    chunks, _ = document_chunks(source, tokenizer, {}, max_tokens=64, overlap_tokens=8)
    assert all(chunk["token_count"] <= 64 for chunk in chunks)
    assert chunks[0]["section"] == "# " + "Ž" * 120
    assert len(chunks[0]["embedding_section"]) < len(chunks[0]["section"])
    assert chunks[0]["text"].startswith("# ")


def test_paragraph_wrapping_keeps_each_original_line_mapping():
    passages, _ = paragraphs(document("Heading\n\nFirst wrapped line\nsecond wrapped line.\n\nNext paragraph."))
    text, spans, _ = passages[1]
    assert text == "First wrapped line second wrapped line."
    assert [span["line_start"] for span in spans] == [3, 4]


def test_business_numeric_line_is_preserved_without_page_footer_evidence(tokenizer):
    source = document("Overview\n\nAvailable account balances\n\n12345")
    chunks, removed = document_chunks(source, tokenizer, {})
    assert any(chunk["text"] == "12345" for chunk in chunks)
    assert not removed


def test_bare_page_footer_needs_multiple_pages_and_matching_page_number(tokenizer):
    source = document("First account details\n\n1\n\fSecond account details\n\n2")
    chunks, removed = document_chunks(source, tokenizer, {})
    assert [item["reason"] for item in removed] == ["standalone-page-number"] * 2
    assert all(chunk["text"] not in {"1", "2"} for chunk in chunks)


def test_inline_formfeed_spans_map_to_full_original_physical_line(tokenizer):
    text = "Before break.\fAfter break.\nNext physical line."
    chunks, _ = document_chunks(document(text), tokenizer, {})
    after = next(chunk for chunk in chunks if "After break." in chunk["text"])
    span = after["spans"][0]
    assert span["page"] == 2
    assert span["line_start"] == 1
    assert span["column_start"] == len("Before break.\f")
    for chunk in chunks:
        for span in chunk["spans"]:
            original = text.split("\n")[span["line_start"] - 1]
            assert chunk["text"][span["text_start"]:span["text_end"]] == original[span["column_start"]:span["column_end"]]
