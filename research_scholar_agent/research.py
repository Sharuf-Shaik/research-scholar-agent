from __future__ import annotations

import asyncio
import math
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .config import get_settings
from .llm.providers import ChatMessage, LLMProvider, get_default_llm_provider
from .schemas import Citation, Paper, ResearchResponse
from .sources.arxiv_client import search_arxiv
from .sources.crossref_client import search_crossref
from .utils.text import jaccard_similarity, normalize_title, tokenize


async def retrieve_sources(query: str, max_per_source: int) -> List[Paper]:
	arxiv_coro = search_arxiv(query=query, max_results=max_per_source)
	crossref_coro = search_crossref(query=query, rows=max_per_source)
	results = await asyncio.gather(arxiv_coro, crossref_coro, return_exceptions=True)
	papers: List[Paper] = []
	for res in results:
		if isinstance(res, Exception):
			logger.warning(f"Retrieval error: {res}")
			continue
		for item in res:
			try:
				papers.append(Paper(**item))
			except Exception as e:
				logger.debug(f"Paper parse skipped: {e}")
	return papers


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
	# Prefer DOI as key, fallback to normalized title similarity
	seen_by_doi: Dict[str, Paper] = {}
	without_doi: List[Paper] = []
	for p in papers:
		if p.doi:
			key = p.doi.lower().strip()
			if key not in seen_by_doi:
				seen_by_doi[key] = p
			else:
				# merge heuristics: prefer longer abstract and venue if missing
				existing = seen_by_doi[key]
				if len(p.abstract) > len(existing.abstract):
					seen_by_doi[key] = p
			continue
		without_doi.append(p)
	# Now dedup remaining by Jaccard on titles
	kept: List[Paper] = list(seen_by_doi.values())
	title_tokens: List[Tuple[Paper, set]] = [(p, tokenize(p.title)) for p in without_doi]
	for i, (p, toks) in enumerate(title_tokens):
		is_dup = False
		for q, qtoks in title_tokens[:i]:
			if jaccard_similarity(toks, qtoks) >= 0.9:
				is_dup = True
				break
		if not is_dup:
			kept.append(p)
	return kept


def score_paper(query: str, paper: Paper) -> float:
	q_tokens = tokenize(query)
	tokens_title = tokenize(paper.title)
	tokens_abs = tokenize(paper.abstract)
	match_title = len(q_tokens & tokens_title)
	match_abs = len(q_tokens & tokens_abs)
	relevance = match_title * 2 + match_abs * 1
	# recency weight: newer is better (year in [1990, 2030])
	year = paper.year or 1990
	recency = (max(min(year, 2030), 1990) - 1990) / (2030 - 1990)
	# Normalize relevance
	norm_rel = 1.0 - math.exp(-0.3 * relevance)
	return 0.75 * norm_rel + 0.25 * recency


def rank_and_select(query: str, papers: List[Paper], top_k: int) -> List[Paper]:
	scored = [(score_paper(query, p), p) for p in papers]
	scored.sort(key=lambda x: x[0], reverse=True)
	return [p for _, p in scored[:top_k]]


def build_citations(selected: List[Paper]) -> List[Citation]:
	citations: List[Citation] = []
	for idx, p in enumerate(selected, start=1):
		citations.append(
			Citation(
				index=idx,
				title=p.title,
				authors=p.authors,
				year=p.year,
				doi=p.doi,
				url=p.url or (p.pdf_url or ""),
				source=p.source,
			)
		)
	return citations


def build_synthesis_messages(query: str, citations: List[Citation]) -> List[ChatMessage]:
	records_block_lines: List[str] = []
	for c in citations:
		authors = ", ".join(c.authors) if c.authors else "Unknown"
		rec = f"[{c.index}] {c.title} — {authors} ({c.year or 'n.d.'}); DOI: {c.doi or 'n/a'}; URL: {c.url}"
		records_block_lines.append(rec)
	records_block = "\n".join(records_block_lines)
	instructions = (
		"You are Research Scholar Agent, an expert academic writer. "
		"Write a rigorous, concise, and well-structured report that synthesizes findings across disciplines. "
		"Use inline numeric citations like [1], [2] whenever you state a claim or refer to a paper. "
		"Only cite from the provided sources. Do not invent citations. If evidence is limited, state that explicitly. "
		"Prefer recent and high-quality sources."
	)
	outline = (
		"Required sections: 1) Executive Summary; 2) Background & Related Work; 3) Methods (if applicable); "
		"4) Key Findings & Themes; 5) Cross-Disciplinary Insights; 6) Limitations & Risks; 7) Open Questions; 8) Future Directions; 9) References. "
		"In References, list each source with its numeric index and full metadata."
	)
	system: ChatMessage = {
		"role": "system",
		"content": (
			"You are a careful research assistant. Follow instructions exactly. "
			"Do not fabricate citations. Keep the tone formal and academic."
		),
	}
	user: ChatMessage = {
		"role": "user",
		"content": (
			f"Research question: {query}\n\n"
			f"Sources:\n{records_block}\n\n"
			f"{instructions}\n{outline}\n"
			"Write 800-1200 words."
		),
	}
	return [system, user]


def synthesize_fallback(query: str, selected: List[Paper]) -> str:
	# Compose a structured report without LLM using abstracts and metadata
	citations = build_citations(selected)
	lines: List[str] = []
	lines.append(f"# {query}\n")
	lines.append("## Executive Summary\n")
	if selected:
		lines.append(
			"This report summarizes key findings from the most relevant recent publications on the topic. "
			"It integrates evidence across sources and highlights areas of consensus and uncertainty.\n"
		)
	else:
		lines.append("No sources were retrieved. The summary below is limited.\n")
	lines.append("## Background & Related Work\n")
	for idx, p in enumerate(selected[: max(3, len(selected) // 3)], start=1):
		snippet = (p.abstract or p.title).strip()
		lines.append(f"- {p.title} [{idx}]: {snippet[:600]}{'...' if len(snippet)>600 else ''}\n")
	lines.append("## Key Findings & Themes\n")
	for idx, p in enumerate(selected[:5], start=1):
		lines.append(f"- Finding linked to [{idx}] derived from its abstract and title.\n")
	lines.append("## Limitations & Risks\n")
	lines.append("- This non-LLM fallback uses abstracts and may miss methodological nuance.\n")
	lines.append("## References\n")
	for c in citations:
		authors = ", ".join(c.authors) if c.authors else "Unknown"
		lines.append(f"[{c.index}] {c.title} — {authors} ({c.year or 'n.d.'}); DOI: {c.doi or 'n/a'}; URL: {c.url}\n")
	return "\n".join(lines)


async def synthesize_report(query: str, selected: List[Paper], provider_choice: Optional[str] = None) -> str:
	try:
		llm: LLMProvider = get_default_llm_provider(provider_choice)
		citations = build_citations(selected)
		messages = build_synthesis_messages(query, citations)
		return await llm.generate(messages=messages, temperature=0.2)
	except Exception as e:
		logger.warning(f"LLM synthesis failed; using fallback. Error: {e}")
		return synthesize_fallback(query, selected)


async def run_research(query: str, top_k: Optional[int] = None, provider_choice: Optional[str] = None) -> ResearchResponse:
	settings = get_settings()
	max_per_source = settings.max_results_per_source
	top_k_effective = top_k or settings.top_k_synthesis
	all_papers = await retrieve_sources(query, max_per_source=max_per_source)
	if not all_papers:
		logger.warning("No papers retrieved; generating a general background without citations")
		text = await synthesize_report(query, selected=[], provider_choice=provider_choice)
		return ResearchResponse(query=query, report_markdown=text, citations=[], num_sources=0, selected=[])
	unique = deduplicate_papers(all_papers)
	selected = rank_and_select(query, unique, top_k=top_k_effective)
	text = await synthesize_report(query, selected=selected, provider_choice=provider_choice)
	return ResearchResponse(query=query, report_markdown=text, citations=build_citations(selected), num_sources=len(selected), selected=selected)