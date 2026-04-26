"""
Conservative OCR line ↔ JSON classification_hints matching for synthetic export.

Prefers clean labels over aggressive matching. Uses normalization only; no ML.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def normalize_basic(s: str) -> str:
    if not s or not str(s).strip():
        return ""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = re.sub(r"\s+", " ", t)
    return t.casefold()


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _strip_currency_tokens(s: str) -> str:
    t = normalize_basic(s)
    t = re.sub(r"\b(pln|zł|zl|eur|usd)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _norm_currency_token_to_iso(tok: str) -> Optional[str]:
    """Map one OCR word (after basic normalize) to PLN / EUR / USD, or None."""
    if not tok:
        return None
    t = normalize_basic(tok)
    if t in ("pln", "zł", "zl"):
        return "PLN"
    if t == "eur":
        return "EUR"
    if t == "usd":
        return "USD"
    return None


def _is_standalone_currency_line(variant: str) -> bool:
    """
    True if the line is only one or more currency tokens (and whitespace).
    False if a monetary amount (digits) remains after removing currency words.
    """
    n = normalize_basic(variant)
    if not n or not re.search(
        r"(?i)(?:\bpln\b|\bzł\b|\bzl\b|\beur\b|\busd\b)", n
    ):
        return False
    if try_parse_amount(variant) is not None:
        rem = re.sub(
            r"(?i)(?:\bpln\b|\bzł\b|\bzl\b|\beur\b|\busd\b)", "", n
        )
        rem = re.sub(r"\s+", "", rem)
        if re.search(r"\d", rem):
            return False
    toks = [t for t in n.split() if t]
    for tok in toks:
        if _norm_currency_token_to_iso(tok) is None:
            return False
    return True


def _hint_to_iso_currencies(canonical: str, rendered: str) -> Set[str]:
    """Invoice CURRENCY: canonical is PLN/EUR/USD; rendered may be zł or the code."""
    s: Set[str] = set()
    c = (canonical or "").strip().upper()
    if c in ("PLN", "EUR", "USD"):
        s.add(c)
    r_iso = _norm_currency_token_to_iso(rendered or "")
    if r_iso:
        s.add(r_iso)
    if not s and rendered:
        ru = (rendered or "").strip().upper()
        if ru in ("PLN", "EUR", "USD"):
            s.add(ru)
    return s


def _ocr_line_to_iso_currencies(variant: str) -> Set[str]:
    n = normalize_basic(variant)
    out: Set[str] = set()
    for tok in n.split():
        iso = _norm_currency_token_to_iso(tok)
        if iso:
            out.add(iso)
    return out


def try_parse_amount(s: str) -> Optional[Decimal]:
    """Parse Polish-style or plain amounts from a fragment."""
    if not s:
        return None
    t = str(s).strip()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"(?i)(pln|zł|zl|eur|usd)$", "", t)
    if not re.search(r"\d", t):
        return None
    # Polish: 1.234,56
    if "," in t and "." in t:
        t = t.replace(".", "").replace(",", ".")
    elif "," in t and "." not in t:
        parts = t.split(",")
        if len(parts[-1]) == 2 and parts[-1].isdigit():
            t = "".join(parts[:-1]) + "." + parts[-1]
        else:
            t = t.replace(",", ".")
    try:
        return Decimal(t).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def amount_key(s: str) -> Optional[str]:
    d = try_parse_amount(s)
    if d is None:
        return None
    return format(d, "f")


def try_parse_date_iso(s: str) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d.%m.%y"):
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# Normalized caption phrases (caption-only lines -> OTHER). Merged with JSON `render` values.
_STATIC_CAPTION_PREFIXES: Tuple[str, ...] = (
    "sprzedawca",
    "wystawca",
    "nabywca",
    "odbiorca",
    "nr faktury",
    "numer faktury",
    "faktura nr",
    "data wystawienia",
    "data wystawienia faktury",
    "wystawiono dnia",
    "data sprzedaży",
    "data zakończenia dostawy/usług",
    "nip sprzedawcy",
    "nip nabywcy",
    "kwota netto",
    "kwota vat",
    "kwota brutto",
    "wartość netto",
    "wartość brutto",
    "razem netto",
    "razem vat",
    "razem brutto",
    "waluta",
    "kod waluty",
    "ilość",
    "il.",
    "jednostka",
    "j.m.",
    "cena netto",
    "cena jedn. netto",
    "stawka vat",
    "vat %",
    "pozycje faktury",
    "wykaz towarów i usług",
    "lp.",
    "lp",
    "nazwa",
    "forma płatności",
    "sposób zapłaty",
    "metoda płatności",
    "termin płatności",
    "płatne do",
    "wart. netto",
    "wart. brutto",
    "netto",
    "brutto",
    "vat",
)


def build_caption_sets(render_block: Optional[Dict[str, Any]]) -> Tuple[Set[str], List[str]]:
    """Exact-match caption norms + prefix list (longest-first) for caption stripping."""
    norms: Set[str] = set()
    for c in _STATIC_CAPTION_PREFIXES:
        n = normalize_basic(c)
        if n:
            norms.add(n)
    if render_block:
        for _k, v in render_block.items():
            if v and isinstance(v, str):
                n = normalize_basic(v)
                if n:
                    norms.add(n)
    # Prefixes: longest first for stripping
    prefixes = sorted((p for p in norms if len(p) >= 2), key=len, reverse=True)
    return norms, prefixes


def line_variants(nline: str, caption_prefixes: Iterable[str]) -> List[str]:
    """Original line plus remainder after stripping a leading caption (caption + value)."""
    seen: Set[str] = set()
    out: List[str] = []
    if nline:
        if nline not in seen:
            seen.add(nline)
            out.append(nline)
    for cap in caption_prefixes:
        if not cap or not nline.startswith(cap):
            continue
        rest = nline[len(cap) :].lstrip(" \t:.-—")
        if rest and rest not in seen:
            seen.add(rest)
            out.append(rest)
    return out


@dataclass
class MatchDecision:
    semantic: Optional[str]
    matched_value: str
    match_type: str
    reason: str


def _tier_for_hint(
    variant: str,
    semantic: str,
    rendered: str,
    canonical: str,
) -> Optional[Tuple[int, str, str, int]]:
    """Return (tier, match_type, matched_value, tie_weight) or None. tie_weight: higher = stronger substring."""
    nr = normalize_basic(rendered)
    nc = normalize_basic(canonical)

    if nr and variant == nr:
        return 0, "exact_rendered", rendered, 0
    if nc and variant == nc:
        return 1, "exact_canonical", canonical, 0

    if "NIP" in semantic:
        dv = digits_only(variant)
        if len(dv) >= 10 and dv == digits_only(canonical):
            return 2, "nip_digits_canonical", canonical, 0
        if len(dv) >= 10 and dv == digits_only(rendered):
            return 2, "nip_digits_rendered", rendered, 0

    if semantic in ("NET_AMOUNT", "VAT_AMOUNT", "TOTAL_PRICE"):
        vk = amount_key(variant)
        if vk:
            if amount_key(canonical) == vk:
                return 3, "amount_canonical", canonical, 0
            if amount_key(rendered) == vk:
                return 3, "amount_rendered", rendered, 0

    if semantic in ("INVOICE_DATE", "SALE_DATE"):
        dk = try_parse_date_iso(variant)
        if dk and try_parse_date_iso(canonical) == dk:
            return 4, "date_canonical", canonical, 0
        if dk and try_parse_date_iso(rendered) == dk:
            return 4, "date_rendered", rendered, 0

    if semantic == "CURRENCY":
        # Line is only e.g. "PLN" / "zł" — v2 is empty; match via token map + hint
        if _is_standalone_currency_line(variant):
            ocr_iso = _ocr_line_to_iso_currencies(variant)
            hint_iso = _hint_to_iso_currencies(canonical, rendered)
            if ocr_iso and hint_iso and (ocr_iso & hint_iso):
                mval = rendered if (rendered or "").strip() else canonical
                return 2, "currency_standalone_token", mval, 0
        v2 = _strip_currency_tokens(variant)
        c2 = normalize_basic(canonical)
        r2 = normalize_basic(rendered)
        if c2 and v2 == c2:
            return 3, "currency_canonical", canonical, 0
        if r2 and v2 == r2:
            return 3, "currency_rendered", rendered, 0
        if c2 == "pln" and v2 in ("pln", "zł", "zl"):
            return 3, "currency_pln_variant", canonical, 0
        if r2 in ("zł", "zl", "pln") and v2 in ("zł", "zl", "pln"):
            return 3, "currency_rendered_pln", rendered, 0

    # Strong substring: prefer longer embedded value
    best_len = 0
    best_label = ""
    for label, cand in (("rendered", nr), ("canonical", nc)):
        if len(cand) < 4:
            continue
        if cand in variant and len(cand) > best_len:
            best_len = len(cand)
            best_label = label
    if best_len > 0:
        mval = rendered if best_label == "rendered" else canonical
        return 5, f"substring_{best_label}", mval, best_len

    return None


def match_ocr_line(
    ocr_text: str,
    hints: List[Dict[str, Any]],
    consumed_semantics: Set[str],
    caption_norms: Set[str],
    caption_prefixes: List[str],
) -> MatchDecision:
    """
    Map one OCR line to a semantic or OTHER (semantic None -> caller maps to OTHER id).
    Conservative: ties -> OTHER. Consumed semantics are excluded.
    """
    nline = normalize_basic(ocr_text)
    if not nline:
        return MatchDecision(None, "", "NONE", "empty_line")

    if nline in caption_norms:
        return MatchDecision(None, "", "NONE", "caption_only_exact")

    variants = line_variants(nline, caption_prefixes)
    candidates: List[Tuple[int, str, str, str, str, int]] = []
    # tier, semantic, match_type, matched_value, variant, tie_weight

    for variant in variants:
        for hint in hints:
            sem = hint.get("semantic")
            if not isinstance(sem, str) or sem in consumed_semantics or sem == "OTHER":
                continue
            rendered = str(hint.get("rendered_value") or "")
            canonical = str(hint.get("canonical_value") or "")
            got = _tier_for_hint(variant, sem, rendered, canonical)
            if got:
                tier, mtype, mval, tw = got
                candidates.append((tier, sem, mtype, mval, variant, tw))

    if not candidates:
        return MatchDecision(None, "", "NONE", "no_hint_match")

    # Keep best (lowest tier, then highest tie_weight for substring) per semantic
    by_sem: Dict[str, Tuple[int, str, str, str, int]] = {}
    for tier, sem, mtype, mval, variant, tw in candidates:
        if sem not in by_sem:
            by_sem[sem] = (tier, mtype, mval, variant, tw)
        else:
            old_t, _om, _ov, _ovr, otw = by_sem[sem]
            if tier < old_t or (tier == old_t and tw > otw):
                by_sem[sem] = (tier, mtype, mval, variant, tw)

    reduced = [(t, s, m, v, vr, tw) for s, (t, m, v, vr, tw) in by_sem.items()]
    best_tier = min(c[0] for c in reduced)
    top = [c for c in reduced if c[0] == best_tier]
    sems = {c[1] for c in top}
    if len(sems) > 1:
        return MatchDecision(
            None,
            "",
            "NONE",
            f"ambiguous_tie_tier_{best_tier}:{','.join(sorted(sems))}",
        )

    _tier, sem, mtype, mval, _var, _tw = top[0]
    return MatchDecision(sem, mval, mtype, "ok")
