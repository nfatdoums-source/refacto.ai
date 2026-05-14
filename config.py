"""Configuration centrale — Refacto.ai (mode contexte limité)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str        = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    api_key: str         = os.getenv("LM_STUDIO_API_KEY",  "lm-studio")
    model: str           = os.getenv("LM_STUDIO_MODEL",    "local-model")
    temperature: float   = float(os.getenv("LM_STUDIO_TEMPERATURE", "0.2"))
    max_tokens: int      = int(os.getenv("LM_STUDIO_MAX_TOKENS", "1500"))
    streaming: bool      = True
    request_timeout: int = int(os.getenv("LM_STUDIO_TIMEOUT", "180"))
    max_workers: int     = 1
    connect_timeout: int = 10


@dataclass(frozen=True)
class AppConfig:
    title: str       = "🔧 Refacto.ai"
    description: str = "Refactorisez votre code legacy avec un LLM local."
    host: str   = "0.0.0.0"
    port: int   = 7860
    share: bool = False
    max_file_size_mb: int = 1


@dataclass(frozen=True)
class RefactoringConfig:
    supported_languages: tuple[str, ...] = (
        "Python", "JavaScript", "TypeScript", "Java",
        "C#", "C++", "Go", "Rust", "PHP", "Ruby",
    )
    refactoring_modes: tuple[str, ...] = (
        "Refactorisation complète",
        "Lisibilité uniquement",
        "Performance uniquement",
        "Sécurité uniquement",
        "Ajout de documentation",
        "Conversion vers patterns modernes",
    )
    max_input_chars: int = 4000   # ~1000 tokens — laisse 3000 pour la réponse


lm_studio_cfg   = LMStudioConfig()
app_cfg         = AppConfig()
refactoring_cfg = RefactoringConfig()