"""
Moteur de refactorisation — LangChain + LM Studio.

Ce module encapsule toute la logique LangChain :
- Connexion au modèle LM Studio (OpenAI-compatible)
- Construction de la chain de refactorisation
- Streaming de la réponse
- Gestion des erreurs et retry
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.callbacks import StreamingStdOutCallbackHandler

from config import lm_studio_cfg, refactoring_cfg
from prompts import build_refactoring_prompt, get_mode_instruction

logger = logging.getLogger(__name__)


class RefactorerError(Exception):
    """Erreur spécifique au moteur de refactorisation."""


class CodeRefactorer:
    """
    Moteur de refactorisation de code via LangChain + LM Studio.

    Utilise l'API OpenAI-compatible de LM Studio pour interroger
    le modèle local chargé dans LM Studio.

    Example:
        >>> refactorer = CodeRefactorer()
        >>> for chunk in refactorer.refactor_stream(code, "Python", "Refactorisation complète"):
        ...     print(chunk, end="", flush=True)
    """

    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self) -> ChatOpenAI:
        """Instancie le client LLM pointant vers LM Studio."""
        return ChatOpenAI(
            base_url=lm_studio_cfg.base_url,
            api_key=lm_studio_cfg.api_key,
            model=lm_studio_cfg.model,
            temperature=lm_studio_cfg.temperature,
            max_tokens=lm_studio_cfg.max_tokens,
            streaming=lm_studio_cfg.streaming,
            request_timeout=lm_studio_cfg.request_timeout,
        )

    def _build_chain(self, language: str, mode: str):
        """
        Construit la LCEL chain : prompt | llm | parser.

        Args:
            language: Langage de programmation.
            mode: Mode de refactorisation.

        Returns:
            Une Runnable chain LangChain.
        """
        prompt = build_refactoring_prompt(language, mode)
        return prompt | self._llm | StrOutputParser()

    def _validate_input(self, code: str, language: str, mode: str) -> None:
        """Valide les entrées avant l'appel au modèle."""
        if not code or not code.strip():
            raise RefactorerError("Le code source ne peut pas être vide.")
        if len(code) > refactoring_cfg.max_input_chars:
            raise RefactorerError(
                f"Le code dépasse la limite de {refactoring_cfg.max_input_chars:,} caractères. "
                f"Veuillez diviser le fichier en sections."
            )
        if language not in refactoring_cfg.supported_languages:
            raise RefactorerError(f"Langage '{language}' non supporté.")
        if mode not in refactoring_cfg.refactoring_modes:
            raise RefactorerError(f"Mode '{mode}' non reconnu.")

    def refactor_stream(
        self,
        code: str,
        language: str,
        mode: str,
    ) -> Generator[str, None, None]:
        """
        Refactorise le code et retourne la réponse en streaming.

        Args:
            code: Le code source legacy à refactoriser.
            language: Le langage de programmation.
            mode: Le mode de refactorisation.

        Yields:
            Des chunks de texte au fur et à mesure que le modèle génère.

        Raises:
            RefactorerError: En cas d'erreur de validation ou de connexion.
        """
        self._validate_input(code, language, mode)

        chain = self._build_chain(language, mode)
        inputs = {
            "language": language,
            "mode": mode,
            "mode_instruction": get_mode_instruction(mode),
            "code": code,
        }

        logger.info("Démarrage refactorisation — langue=%s mode=%s taille=%d chars",
                    language, mode, len(code))

        start = time.perf_counter()
        try:
            for chunk in chain.stream(inputs):
                yield chunk
        except ConnectionRefusedError as exc:
            raise RefactorerError(
                "Impossible de se connecter à LM Studio. "
                "Vérifiez que LM Studio est lancé et qu'un modèle est chargé."
            ) from exc
        except Exception as exc:
            logger.exception("Erreur inattendue lors de la refactorisation")
            raise RefactorerError(f"Erreur du modèle : {exc}") from exc
        finally:
            elapsed = time.perf_counter() - start
            logger.info("Refactorisation terminée en %.2fs", elapsed)

    def refactor_sync(self, code: str, language: str, mode: str) -> str:
        """
        Version synchrone (non-streaming) — utile pour les tests unitaires.

        Returns:
            La réponse complète du modèle sous forme de chaîne.
        """
        return "".join(self.refactor_stream(code, language, mode))

    def health_check(self) -> dict[str, Any]:
        """
        Vérifie la connectivité avec LM Studio et récupère le modèle chargé.
        """
        try:
            import httpx
            response = httpx.get(
                f"{lm_studio_cfg.base_url}/models",
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            model_name = models[0]["id"] if models else "Aucun modèle chargé"
            return {
                "status": "ok",
                "base_url": lm_studio_cfg.base_url,
                "model": model_name,
                "model_count": len(models),
            }
        except Exception as exc:
            return {
                "status": "error",
                "base_url": lm_studio_cfg.base_url,
                "model": "Inconnu",
                "model_count": 0,
                "error": str(exc),
            }
