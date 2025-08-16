from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
	# LLM provider settings
	default_llm_provider: Optional[str] = Field(
		default=None,
		alias="DEFAULT_LLM_PROVIDER",
		description="One of: openai, together, ollama. If None, auto-detect based on available env/host.",
	)
	openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
	together_api_key: Optional[str] = Field(default=None, alias="TOGETHER_API_KEY")
	ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

	# Search/backends
	crossref_mailto: str = Field(
		default="your-email@example.com",
		alias="CROSSREF_MAILTO",
		description="Email used for polite Crossref requests",
	)

	# Pipeline defaults
	max_results_per_source: int = Field(default=25)
	top_k_synthesis: int = Field(default=10)

	class Config:
		env_file = ".env"
		case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
	return AppSettings()  # type: ignore[arg-type]