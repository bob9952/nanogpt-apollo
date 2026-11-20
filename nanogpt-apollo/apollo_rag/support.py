"""Reviewed guidance for stock-account and posting-balance support tickets."""

from pathlib import PurePosixPath


STOCK_FOCUS = "Konto zaliha"
BALANCE_FOCUS = "Neravnoteža dokumenta"
_GUIDE = "Apollo G4 ERP Users Guide v2.txt"
_LOVCEN = "Lovcen Auto v1.0.pdf.txt"

# Match the relevant wording in the indexed body. A heading or form code
# alone does not establish the rule or its lookup fields.
_RULES = (
    ("stock_default", _GUIDE, STOCK_FOCUS, (
        "default ledger accounts", "lin001m001",
        "warehouse (organizational unit) of stock", "item type of items in warehouse",
        "specific ledger account", "lit002m001",
    )),
    ("document_balance", _GUIDE, BALANCE_FOCUS, (
        "financial balance, debit side is not equal credit side",
    )),
    ("stock_fallback", _LOVCEN, STOCK_FOCUS, (
        "konto izostavi sistem sam koristi konto zaliha",
        "artikal, o.j., datum važnosti i partner",
        "automatski otvara prema podrazumevanim vrednostima konta",
        "lin001m001",
    )),
)


def reviewed_passages(chunks):
    """Find reviewed rules within the selected corpus scope."""
    by_name = {_GUIDE: [], _LOVCEN: []}
    for chunk in chunks:
        if chunk["kind"] == "manual":
            name = PurePosixPath(chunk["path"]).name
            if name in by_name:
                by_name[name].append(chunk)

    matches = {}
    for rule, name, focus, phrases in _RULES:
        for chunk in by_name[name]:
            words = " ".join(chunk["text"].casefold().split())
            if all(phrase in words for phrase in phrases):
                matches[chunk["id"]] = (rule, focus)
                break
    return matches


def reviewed_answer(results):
    """Render the checklist when both primary rules have been verified."""
    sources = {}
    for passage in results:
        rule = passage.get("support_rule")
        if rule and rule not in sources:
            sources[rule] = {**passage, "citation": f"P{len(sources) + 1}"}
    if not {"stock_default", "document_balance"} <= sources.keys():
        return None

    stock = sources["stock_default"]["citation"]
    balance = sources["document_balance"]["citation"]
    sections = [
        f"**Konto zaliha.** U LIN001M001 proverite podrazumevano konto za kombinaciju "
        f"magacina (O.J.) i tipa artikla [{stock}]. Za poseban konto artikla proverite LIT002M001 [{stock}]."
    ]
    if fallback := sources.get("stock_fallback"):
        sections.append(
            "**Pravilo knjiženja.** Proverite da li je polje Konto popunjeno. "
            "Ako je prazno i vaša instalacija koristi dokumentovano pravilo, proverite konto "
            "na stanju za artikal, magacin, datum važnosti i partnera. Ako stanje nedostaje, "
            f"proverite podrazumevani konto u LIN001M001 [{fallback['citation']}][{stock}]."
        )
    sections.append(
        f"**Neravnoteža.** Uporedite zbirove Duguje i Potražuje [{balance}]. "
        "Pošaljite tip dokumenta i finansijske stavke sa dugovnim i potražnim iznosima. "
        "Ti podaci su potrebni da se utvrdi uzrok razlike i moguća veza sa kontom zaliha."
    )
    return "\n\n".join(sections), list(sources.values())
