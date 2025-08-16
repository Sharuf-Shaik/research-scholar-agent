from __future__ import annotations

import asyncio

import streamlit as st

from .research import run_research

st.set_page_config(page_title="Research Scholar Agent", layout="wide")

st.title("Research Scholar Agent")

with st.form("research_form"):
	query = st.text_area("Research question or topic", height=120, placeholder="e.g., Large Language Models for clinical decision support")
	top_k = st.slider("Number of sources to synthesize", min_value=3, max_value=20, value=8)
	provider = st.selectbox("LLM Provider (auto-detect if empty)", options=["", "openai", "together", "ollama"], index=0)
	submitted = st.form_submit_button("Run Research")

if submitted and query.strip():
	with st.spinner("Retrieving sources and synthesizing report..."):
		resp = asyncio.run(run_research(query=query.strip(), top_k=top_k, provider_choice=(provider or None)))
		st.subheader("Report")
		st.markdown(resp.report_markdown)
		st.subheader("Citations")
		for c in resp.citations:
			authors = ", ".join(c.authors) if c.authors else "Unknown"
			st.write(f"[{c.index}] {c.title} — {authors} ({c.year or 'n.d.'}); DOI: {c.doi or 'n/a'}; URL: {c.url}")
		st.download_button("Download Markdown", data=resp.report_markdown, file_name="report.md", mime="text/markdown")
		st.success(f"Synthesized from {resp.num_sources} sources")