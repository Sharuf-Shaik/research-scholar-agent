from __future__ import annotations

import re
from typing import Iterable, Set


_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


def normalize_title(text: str) -> str:
	text = text.lower()
	text = _PUNCT_RE.sub(" ", text)
	text = _WS_RE.sub(" ", text)
	return text.strip()


def tokenize(text: str) -> Set[str]:
	return set([t for t in normalize_title(text).split(" ") if t])


def jaccard_similarity(a_tokens: Iterable[str], b_tokens: Iterable[str]) -> float:
	a = set(a_tokens)
	b = set(b_tokens)
	if not a and not b:
		return 1.0
	if not a or not b:
		return 0.0
	inter = len(a & b)
	union = len(a | b)
	return inter / union