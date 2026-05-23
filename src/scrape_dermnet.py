"""Scrape DermNet NZ to build the Dermagemma knowledge base.

Reads the v2 classifier's id2label from vit_skin_classifier_v2/config.json,
resolves each label to a DermNet topic slug via the public sitemap, fetches
the topic page, and parses it into one KB entry per <h2> section.

Each entry is tagged with `soc_relevant: bool` if either the section heading
matches a SOC-specific pattern (e.g. "How do clinical features vary in
differing types of skin?") or the body contains SOC keywords.

Output: knowledge_base.json at the repo root.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

DERMNET_BASE = "https://dermnetnz.org"
SITEMAP_URL = f"{DERMNET_BASE}/sitemap.xml"

# Sections whose heading alone indicates the content is SOC-specific.
SOC_SECTION_PATTERNS = [
    r"differing types of skin",
    r"in skin of colou?r",
    r"in darker skin",
    r"in dark skin",
    r"ethnic differences",
    r"vary.*skin",
]

# Substrings that mark a section body as SOC-relevant.
SOC_KEYWORDS = [
    "skin of color", "skin of colour",
    "darker skin", "dark skin", "deeply pigmented",
    "fitzpatrick iv", "fitzpatrick v", "fitzpatrick vi",
    "type iv skin", "type v skin", "type vi skin",
    "ethnic skin", "pigmented skin",
    "black skin", "african", "afro-",
    "hispanic", "latino", "latina",
    "south asian", "south-east asian", "asian skin",
    "post-inflammatory hyperpigmentation",
    "post-inflammatory hypopigmentation",
    "hyperpigmentation", "hypopigmentation",
]

# Hand-curated slug overrides for labels the fuzzy matcher gets wrong.
# Tried first; if the override slug isn't in the sitemap, falls back to scoring.
MANUAL_SLUG_OVERRIDES = {
    "Acne_Keloidalis_Nuchae": "acne-keloidalis",
    "Dermatomyositis": "dermatomyositis",
    "Factitial_Dermatitis": "dermatitis-artefacta",
    "Hailey_Hailey_Disease": "hailey-hailey-disease",
    "Lichen_Amyloidosis": "lichen-amyloidosis",
    "Lupus_Erythematosus": "cutaneous-lupus-erythematosus",
    "Milia": "milium-cyst",
    "Mucous_Cyst": "digital-myxoid-cyst",
    "Naevus_Comedonicus": "naevus-comedonicus",
    "Nematode_Infection": "cutaneous-larva-migrans",
    "Neurotic_Excoriations": "neurotic-excoriations",
    "Pediculosis_Lids": "pediculosis",
    "Photodermatoses": "photodermatoses",
    "Pilar_Cyst": "trichilemmal-cyst",
    "Prurigo_Nodularis": "prurigo-nodularis",
    "Scleroderma": "systemic-sclerosis",
    "Scleromyxedema": "scleromyxoedema",
    "Seborrheic_Dermatitis": "seborrhoeic-dermatitis",
    "Squamous_Cell_Carcinoma": "cutaneous-squamous-cell-carcinoma",
    "Telangiectases": "telangiectasia",
    "Urticaria": "urticaria",
}

# DermNet has parallel sub-pages whose slugs end in these suffixes — they're
# usually image galleries or histopathology pages with no clinical prose.
# Penalize them so the scoring matcher prefers the main topic page.
SUB_PAGE_SUFFIXES = (
    "-images", "-pathology", "-dermoscopy",
    "-in-children", "-questions", "-management",
)

# Map heading shape -> entry "type" used by main.py's retriever boosts.
TYPE_RULES = [
    (re.compile(r"treatment|manage", re.I), "treatment"),
    (re.compile(r"complications|outlook|outcome|prognosis|prevent", re.I), "pitfall"),
    (re.compile(r"clinical features|differing types of skin|signs", re.I), "feature"),
    (re.compile(r"causes|who gets|risk", re.I), "feature"),
    (re.compile(r"diagnos|differential|tests|investigat", re.I), "feature"),
]


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def fetch_sitemap_slugs(client: httpx.Client) -> List[str]:
    print("Fetching DermNet sitemap...")
    r = client.get(SITEMAP_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns_match = re.match(r"\{(.*?)\}", root.tag)
    tag = f"{{{ns_match.group(1)}}}loc" if ns_match else "loc"

    slugs: List[str] = []
    for loc in root.iter(tag):
        url = (loc.text or "").strip()
        if "/topics/" in url:
            slug = url.rsplit("/topics/", 1)[1].rstrip("/")
            if slug:
                slugs.append(slug)
    print(f"  {len(slugs)} topic slugs in sitemap.")
    return slugs


# ---------------------------------------------------------------------------
# Label -> slug resolution
# ---------------------------------------------------------------------------

def _label_forms(label: str) -> List[str]:
    base = label.replace("_", " ").lower().strip()
    forms = [base, base.replace(" ", "-")]
    if base.endswith("s"):
        sing = base[:-1]
        forms.extend([sing, sing.replace(" ", "-")])
    return forms


def _word_forms(word: str) -> List[str]:
    """Return word + light singularization variants for fuzzy matching."""
    forms = [word]
    if word.endswith("ies") and len(word) > 5:
        forms.append(word[:-3] + "y")
    elif word.endswith("s") and len(word) > 4:
        forms.append(word[:-1])
    return forms


def match_slug(label: str, slugs: List[str]) -> Optional[str]:
    slug_set = set(slugs)

    # 0. Manual override (curated for known-bad fuzzy matches).
    override = MANUAL_SLUG_OVERRIDES.get(label)
    if override and override in slug_set:
        return override

    # 1. Exact slug match in any normalized form.
    for f in _label_forms(label):
        if f in slug_set:
            return f

    # 2. Scored partial match across label words (with singularization).
    words = [w for w in label.replace("_", " ").lower().split() if len(w) > 2]
    if not words:
        return None
    word_forms = [_word_forms(w) for w in words]

    scored = []
    for s in slugs:
        score = sum(1 for forms in word_forms if any(f in s for f in forms))
        if score == 0:
            continue
        starts = 1 if any(s.startswith(f) for f in word_forms[0]) else 0
        sub_page_penalty = -1 if any(s.endswith(suf) for suf in SUB_PAGE_SUFFIXES) else 0
        # Sort by: more matches, no sub-page suffix, starts-with-first-word, shorter slug.
        scored.append((score, sub_page_penalty, starts, -len(s), s))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][-1]


# ---------------------------------------------------------------------------
# Page parsing
# ---------------------------------------------------------------------------

def parse_sections(html: str) -> List[tuple]:
    """Return [(heading, body_text), ...] split by <h2> boundaries."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    sections: List[tuple] = []
    current_h: Optional[str] = None
    current_body: List[str] = []

    for el in soup.find_all(["h2", "h3", "p", "ul", "ol", "li"]):
        if el.name == "h2":
            if current_h and current_body:
                sections.append((current_h, " ".join(current_body).strip()))
            current_h = el.get_text(" ", strip=True)
            current_body = []
        elif current_h:
            text = el.get_text(" ", strip=True)
            if text:
                current_body.append(text)

    if current_h and current_body:
        sections.append((current_h, " ".join(current_body).strip()))
    return sections


def classify_type(heading: str) -> str:
    for pattern, label in TYPE_RULES:
        if pattern.search(heading):
            return label
    return "definition"


def is_soc_section(heading: str) -> bool:
    h = heading.lower()
    return any(re.search(p, h) for p in SOC_SECTION_PATTERNS)


def has_soc_keywords(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in SOC_KEYWORDS)


def build_entries(pathology: str, sections: list, url: str, max_chars: int) -> List[dict]:
    entries = []
    pid = re.sub(r"[^a-z0-9]+", "_", pathology.lower()).strip("_")
    for i, (heading, body) in enumerate(sections):
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < 60:
            continue
        if len(body) > max_chars:
            body = body[:max_chars].rsplit(" ", 1)[0] + "..."
        soc = is_soc_section(heading) or has_soc_keywords(body)
        entries.append({
            "id": f"{pid}__{i}",
            "pathology": pathology,
            "type": classify_type(heading),
            "source": "DermNet NZ",
            "source_url": url,
            "heading": heading,
            "text": body,
            "soc_relevant": soc,
        })
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_labels(config_path: str) -> List[str]:
    with open(config_path) as f:
        cfg = json.load(f)
    labels = list(cfg["id2label"].values())

    # Collapse near-duplicate labels (e.g. v2 has both "Keloid" and "Keloids").
    seen = {}
    for l in labels:
        key = l.lower().rstrip("s")
        seen.setdefault(key, l)
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description="Scrape DermNet to build knowledge_base.json.")
    parser.add_argument("--labels", default="vit_skin_classifier_v2/config.json",
                        help="Path to classifier config containing id2label.")
    parser.add_argument("--out", default="knowledge_base.json")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds between requests.")
    parser.add_argument("--max-chars", type=int, default=700, help="Max chars per section body.")
    parser.add_argument("--limit", type=int, default=None, help="Cap conditions for testing.")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]
    print(f"Loaded {len(labels)} unique labels from {args.labels}.")

    headers = {"User-Agent": "Dermagemma-Research/0.1 (educational; contact: github.com/saifxyzyz)"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        slugs = fetch_sitemap_slugs(client)

        all_entries: List[dict] = []
        slug_map: dict = {}
        unmatched: List[str] = []

        for i, label in enumerate(labels, start=1):
            slug = match_slug(label, slugs)
            if not slug:
                unmatched.append(label)
                print(f"[{i:2d}/{len(labels)}] {label}: NO MATCH")
                continue

            url = f"{DERMNET_BASE}/topics/{slug}"
            try:
                r = client.get(url, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"[{i:2d}/{len(labels)}] {label} ({slug}): fetch failed — {e}")
                continue

            sections = parse_sections(r.text)
            pathology = label.replace("_", " ")
            entries = build_entries(pathology, sections, url, args.max_chars)
            all_entries.extend(entries)
            slug_map[label] = slug
            soc_count = sum(1 for e in entries if e["soc_relevant"])
            print(f"[{i:2d}/{len(labels)}] {label} -> {slug}  ({len(entries)} entries, {soc_count} SOC)")
            time.sleep(args.sleep)

    soc_total = sum(1 for e in all_entries if e["soc_relevant"])
    print(f"\n=== Scrape complete ===")
    print(f"Total entries: {len(all_entries)}")
    print(f"SOC-tagged:    {soc_total} ({soc_total / max(1, len(all_entries)) * 100:.1f}%)")
    print(f"Unmatched ({len(unmatched)}): {unmatched}")

    payload = {
        "_meta": {
            "source": "DermNet NZ",
            "labels_source": args.labels,
            "label_count": len(labels),
            "entry_count": len(all_entries),
            "soc_tagged": soc_total,
            "slug_map": slug_map,
            "unmatched": unmatched,
        },
        "entries": all_entries,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
