from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
	model_config = ConfigDict(extra="ignore", populate_by_name=True)
	# Normalized metadata across sources
	source: str = Field(description="Origin source, e.g., arxiv, crossref")
	id: Optional[str] = Field(default=None)
	title: str
	abstract: str = ""
	authors: List[str] = []
	year: Optional[int] = None
	url: str = ""
	pdf_url: Optional[str] = None
	doi: Optional[str] = None
	venue: Optional[str] = None


class Citation(BaseModel):
	index: int
	title: str
	authors: List[str]
	year: Optional[int]
	doi: Optional[str]
	url: str
	source: str


class ResearchRequest(BaseModel):
	query: str
	top_k: Optional[int] = None
	provider: Optional[str] = None


class ResearchResponse(BaseModel):
	query: str
	report_markdown: str
	citations: List[Citation]
	num_sources: int
	selected: List[Paper]