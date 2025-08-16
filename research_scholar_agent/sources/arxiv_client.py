from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import feedparser
import httpx
from loguru import logger

ARXIV_API = "https://export.arxiv.org/api/query"


def _extract_year(dt: Optional[str]) -> Optional[int]:
	if not dt:
		return None
	m = re.match(r"(\d{4})-\d{2}-\d{2}", dt)
	return int(m.group(1)) if m else None


async def search_arxiv(query: str, max_results: int = 25, start: int = 0) -> List[Dict[str, Any]]:
	params = {
		"search_query": query,
		"start": start,
		"max_results": max_results,
	}
	async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), follow_redirects=True) as client:
		resp = await client.get(ARXIV_API, params=params)
		resp.raise_for_status()
		feed_text = resp.text
	feed = feedparser.parse(feed_text)
	results: List[Dict[str, Any]] = []
	for entry in feed.entries:
		authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
		links = entry.get("links", [])
		link_pdf = None
		link_abs = None
		for l in links:
			rel = l.get("rel")
			href = l.get("href")
			if rel == "related" and href and href.endswith(".pdf"):
				link_pdf = href
			if rel == "alternate" and href:
				link_abs = href
		doi = getattr(entry, "arxiv_doi", None)
		results.append(
			{
				"source": "arxiv",
				"id": entry.get("id"),
				"title": entry.get("title", "").strip(),
				"abstract": entry.get("summary", "").strip(),
				"authors": authors,
				"year": _extract_year(entry.get("published")),
				"url": link_abs or entry.get("id"),
				"pdf_url": link_pdf,
				"doi": doi,
				"venue": None,
			}
		)
	logger.debug(f"arXiv returned {len(results)} results for query='{query}'")
	return results