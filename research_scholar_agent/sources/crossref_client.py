from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from ..config import get_settings

CROSSREF_API = "https://api.crossref.org/works"


def _strip_html(text: Optional[str]) -> str:
	if not text:
		return ""
	text = html.unescape(text)
	text = re.sub(r"<[^>]+>", " ", text)
	return re.sub(r"\s+", " ", text).strip()


def _extract_year(item: Dict[str, Any]) -> Optional[int]:
	issued = item.get("issued", {})
	parts = issued.get("date-parts") or []
	if parts and parts[0]:
		try:
			return int(str(parts[0][0]))
		except Exception:
			return None
	return None


async def search_crossref(query: str, rows: int = 25, offset: int = 0) -> List[Dict[str, Any]]:
	settings = get_settings()
	params = {"query": query, "rows": rows, "offset": offset}
	if settings.crossref_mailto:
		params["mailto"] = settings.crossref_mailto
	async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
		resp = await client.get(CROSSREF_API, params=params)
		resp.raise_for_status()
		data = resp.json()
	items = data.get("message", {}).get("items", [])
	results: List[Dict[str, Any]] = []
	for it in items:
		authors = []
		for a in it.get("author", []) or []:
			name_parts = [a.get("given"), a.get("family")]
			authors.append(" ".join([p for p in name_parts if p]))
		results.append(
			{
				"source": "crossref",
				"id": it.get("DOI") or it.get("URL"),
				"title": (it.get("title") or [""])[0],
				"abstract": _strip_html(it.get("abstract")),
				"authors": authors,
				"year": _extract_year(it),
				"url": it.get("URL") or "",
				"pdf_url": None,
				"doi": it.get("DOI"),
				"venue": it.get("container-title", [None])[0] or it.get("publisher"),
			}
		)
	logger.debug(f"Crossref returned {len(results)} results for query='{query}'")
	return results