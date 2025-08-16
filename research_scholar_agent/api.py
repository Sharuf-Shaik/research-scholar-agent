from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from . import __version__
from .research import run_research
from .schemas import ResearchRequest, ResearchResponse

app = FastAPI(title="Research Scholar Agent API", version=__version__, default_response_class=ORJSONResponse)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
	return {"status": "ok", "version": __version__}


@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(req: ResearchRequest) -> ResearchResponse:
	return await run_research(query=req.query, top_k=req.top_k, provider_choice=req.provider)