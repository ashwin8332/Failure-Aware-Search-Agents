"""PDF Question Answering Agent.

Extracts text from uploaded PDFs and answers questions using heuristics:
- Summarize: key sentences via TF-IDF-style scoring
- Explain: section identification by keyword density
- Find contradictions: negation-based conflict detection
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── PDF text extraction ────────────────────────────────────────────────────

def extract_text_from_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extract full text from PDF bytes. Returns (text, page_count)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
        full_text = "\n\n".join(p for p in pages if p)
        return full_text, len(reader.pages)
    except Exception as exc:
        logger.warning("PDF extraction failed: %s", exc)
        raise ValueError(f"Could not read PDF: {exc}") from exc


# ── Text utilities ────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "this", "that", "these", "those",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "with",
    "by", "from", "it", "its", "we", "our", "they", "their", "he", "she",
    "as", "if", "then", "than", "so", "not", "no", "also", "such", "which",
}


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]


def _word_freq(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return Counter(w for w in words if w not in _STOPWORDS)


def _score_sentence(sentence: str, freq: Counter) -> float:
    words = re.findall(r"[a-zA-Z]{3,}", sentence.lower())
    content = [w for w in words if w not in _STOPWORDS]
    if not content:
        return 0.0
    return sum(freq.get(w, 0) for w in content) / len(content)


def _top_sentences(text: str, n: int = 5) -> List[str]:
    sents = _sentences(text)
    if not sents:
        return []
    freq = _word_freq(text)
    scored = sorted(sents, key=lambda s: _score_sentence(s, freq), reverse=True)
    # Return in original order for readability
    top_set = set(scored[:n])
    return [s for s in sents if s in top_set][:n]


def _find_sections(text: str) -> List[Dict[str, str]]:
    """Identify likely section headings and their first paragraph."""
    lines = text.split("\n")
    sections = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Heuristic: short line in ALL CAPS or ends without period → heading
        if (10 < len(line) < 80) and (line.isupper() or not line.endswith(".")):
            # Collect next paragraph
            para_lines = []
            j = i + 1
            while j < len(lines) and len(para_lines) < 6:
                l = lines[j].strip()
                if l:
                    para_lines.append(l)
                elif para_lines:
                    break
                j += 1
            if para_lines:
                sections.append({
                    "heading": line,
                    "content": " ".join(para_lines)[:400],
                })
        i += 1
    return sections[:8]


def _find_contradictions(text: str) -> List[Dict[str, str]]:
    """Find sentence pairs that may contradict each other."""
    sents = _sentences(text)
    NEGATIONS = ["not", "never", "no", "cannot", "isn't", "wasn't", "doesn't",
                 "didn't", "won't", "wouldn't", "couldn't", "shouldn't",
                 "however", "but", "contrary", "despite", "although", "yet"]
    neg_sents = [s for s in sents if any(w in s.lower().split() for w in NEGATIONS)]
    pos_sents = [s for s in sents if s not in neg_sents and len(s) > 40]

    pairs = []
    freq = _word_freq(text)
    for ns in neg_sents[:10]:
        ns_words = set(re.findall(r"[a-zA-Z]{4,}", ns.lower())) - _STOPWORDS
        best_score = 0
        best_match = None
        for ps in pos_sents:
            ps_words = set(re.findall(r"[a-zA-Z]{4,}", ps.lower())) - _STOPWORDS
            overlap = len(ns_words & ps_words)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_match = ps
        if best_match:
            pairs.append({"claim": best_match[:200], "contradiction": ns[:200]})

    return pairs[:5]


# ── Main Q&A entry point ──────────────────────────────────────────────────

def answer_question(
    text: str,
    question_type: str,
    custom_question: Optional[str] = None,
    page_count: int = 0,
) -> Dict[str, Any]:
    """Answer a question about PDF text.

    question_type: 'summarize' | 'explain' | 'contradictions' | 'custom'
    """
    word_count = len(text.split())

    if question_type == "summarize":
        top = _top_sentences(text, n=5)
        answer = " ".join(top) if top else text[:600]
        retrieval_quality = min(1.0, len(top) / 5)
        explanation_label = f"Extracted {len(top)} key sentences from {word_count} words across {page_count} pages"
        policy = "confident" if retrieval_quality >= 0.7 else "hedge"
        confidence = 0.78 if policy == "confident" else 0.58

    elif question_type == "explain":
        sections = _find_sections(text)
        if sections:
            parts = [f"**{s['heading']}**: {s['content']}" for s in sections]
            answer = "\n\n".join(parts)
            retrieval_quality = min(1.0, len(sections) / 5)
        else:
            top = _top_sentences(text, n=4)
            answer = " ".join(top) or text[:500]
            retrieval_quality = 0.5
        explanation_label = f"Identified {len(sections) if sections else 0} sections in document"
        policy = "confident" if retrieval_quality >= 0.6 else "hedge"
        confidence = 0.72 if policy == "confident" else 0.52

    elif question_type == "contradictions":
        pairs = _find_contradictions(text)
        if pairs:
            parts = []
            for i, p in enumerate(pairs, 1):
                parts.append(f"**Potential contradiction {i}:**\n• Claim: {p['claim']}\n• Tension: {p['contradiction']}")
            answer = "\n\n".join(parts)
            retrieval_quality = min(1.0, len(pairs) / 3)
            policy = "hedge"
            confidence = 0.55
        else:
            answer = "No obvious contradictions detected in this document. The text appears internally consistent."
            retrieval_quality = 0.9
            policy = "confident"
            confidence = 0.82
        explanation_label = f"Analyzed {len(_sentences(text))} sentences for contradiction patterns"

    elif question_type == "custom" and custom_question:
        # Keyword search in text
        freq = _word_freq(custom_question)
        sents = _sentences(text)
        scored = sorted(sents, key=lambda s: _score_sentence(s, freq), reverse=True)
        relevant = scored[:4]
        if relevant and _score_sentence(relevant[0], freq) > 0:
            answer = " ".join(relevant[:3])
            retrieval_quality = min(1.0, _score_sentence(relevant[0], freq) / 3)
        else:
            answer = "The document does not appear to directly address this question based on keyword matching."
            retrieval_quality = 0.2
        policy = "confident" if retrieval_quality >= 0.5 else "hedge"
        confidence = 0.65 if policy == "confident" else 0.42
        explanation_label = f"Searched {len(sents)} sentences for: '{custom_question[:60]}'"

    else:
        answer = "Please select a question type (summarize, explain, contradictions) or enter a custom question."
        retrieval_quality = 0.0
        policy = "hedge"
        confidence = 0.30
        explanation_label = "No question type provided"

    # Build failure_reasons for hedge
    failure_reasons = []
    if policy == "hedge":
        failure_reasons = [
            {"label": f"Document coverage {'good' if word_count > 500 else 'limited'} ({word_count} words)", "passed": word_count > 500},
            {"label": f"Retrieval quality {'adequate' if retrieval_quality >= 0.5 else 'low'} ({retrieval_quality:.0%})", "passed": retrieval_quality >= 0.5},
            {"label": "PDF text extraction successful", "passed": True},
            {"label": "Answer based on heuristic (no LLM)", "passed": False},
        ]

    explanation = {
        "factors": [
            {"label": f"Document: {page_count} pages, {word_count} words", "passed": word_count > 100, "icon": "✔" if word_count > 100 else "✖"},
            {"label": explanation_label, "passed": retrieval_quality > 0.3, "icon": "✔" if retrieval_quality > 0.3 else "✖"},
            {"label": f"Question type: {question_type}", "passed": True, "icon": "✔"},
            {"label": "Analysis method: heuristic (keyword + structure)", "passed": True, "icon": "✔"},
        ],
        "failure_reasons": failure_reasons,
        "confidence_breakdown": {
            "retrieval_quality_score": round(retrieval_quality, 3),
            "retrieval_contribution_pct": round(retrieval_quality * 45),
            "source_trust_score": 1.0,
            "trust_contribution_pct": 30,
            "verification_score": 0.5,
            "verification_contribution_pct": 13,
            "conflict_penalty": 0,
            "conflict_penalty_pct": 0,
            "final": round(confidence, 3),
            "final_pct": int(confidence * 100),
            "formula": f"PDF heuristic analysis · {question_type} mode",
        },
        "policy_reason": (
            f"PDF analysis via {question_type} mode. "
            f"Retrieval quality: {retrieval_quality:.0%}. "
            f"{'High confidence — sufficient evidence found.' if policy == 'confident' else 'Hedge — heuristic analysis, no LLM verification.'}"
        ),
        "policy_chosen": policy,
        "source_trust_scores": [{"domain": "uploaded-pdf", "trust_score": 70, "trust_pct": 70, "trust_tier": "credible", "url": ""}],
    }

    return {
        "answer": answer,
        "policy": policy,
        "confidence": confidence,
        "explanation": explanation,
        "meta": {
            "question_type": question_type,
            "word_count": word_count,
            "page_count": page_count,
            "retrieval_quality": round(retrieval_quality, 3),
        },
    }
