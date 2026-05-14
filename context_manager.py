"""
Gestionnaire de contexte partagé entre fichiers.

Après chaque fichier refactorisé, extrait les changements importants
(symboles renommés, nouvelles signatures, types créés, conventions adoptées)
et les injecte dans les prompts des fichiers suivants.

C'est le cœur de l'intelligence multi-fichiers : chaque fichier
"connaît" ce qui a été fait dans les fichiers dont il dépend.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class SymbolChange:
    """Représente un symbole qui a changé de nom ou de signature."""
    old_name: str
    new_name: str
    kind: str           # "function" | "class" | "variable" | "type"
    source_file: str    # Fichier d'origine
    signature: str = ""


@dataclass
class FileContext:
    """Contexte extrait d'un fichier après refactorisation."""
    relative_path: str
    language: str
    # Symboles exportés (fonctions, classes, types publics)
    exported_symbols: list[str] = field(default_factory=list)
    # Changements de noms détectés
    symbol_changes: list[SymbolChange] = field(default_factory=list)
    # Conventions adoptées (ex: snake_case, type hints, etc.)
    conventions: list[str] = field(default_factory=list)
    # Extrait du code refactorisé (signatures publiques uniquement)
    public_api_snippet: str = ""


@dataclass
class SharedContext:
    """
    Mémoire partagée de toute la session de refactorisation.
    Accumulée au fil des fichiers traités.
    """
    language: str
    mode: str
    # Historique par fichier
    file_contexts: dict[str, FileContext] = field(default_factory=dict)
    # Index global des changements de symboles : ancien_nom → SymbolChange
    global_renames: dict[str, SymbolChange] = field(default_factory=dict)
    # Conventions globales adoptées
    global_conventions: list[str] = field(default_factory=list)

    def add_file_context(self, ctx: FileContext) -> None:
        """Enregistre le contexte d'un fichier traité."""
        self.file_contexts[ctx.relative_path] = ctx
        for change in ctx.symbol_changes:
            self.global_renames[change.old_name] = change
        # Déduplication des conventions
        for conv in ctx.conventions:
            if conv not in self.global_conventions:
                self.global_conventions.append(conv)

    def get_relevant_context(
        self,
        file_path: str,
        dependencies: set[str],
    ) -> str:
        """
        Génère le bloc de contexte à injecter dans le prompt d'un fichier.

        Inclut uniquement le contexte des fichiers dont ce fichier dépend
        directement, pour éviter le bruit.

        Args:
            file_path: Chemin du fichier en cours de traitement.
            dependencies: Ensemble des fichiers dont ce fichier dépend.

        Returns:
            Texte de contexte formaté à injecter dans le prompt.
        """
        if not self.file_contexts:
            return ""

        lines: list[str] = []

        # Contexte des fichiers dépendants
        relevant = {
            path: ctx
            for path, ctx in self.file_contexts.items()
            if path in dependencies
        }

        if relevant:
            lines.append("=== CONTEXTE DES FICHIERS DÉPENDANTS ===")
            lines.append(
                "Ces fichiers ont déjà été refactorisés. "
                "Respecte leurs interfaces et noms de symboles."
            )
            for dep_path, ctx in relevant.items():
                lines.append(f"\n--- {dep_path} ---")
                if ctx.public_api_snippet:
                    lines.append(ctx.public_api_snippet)
                if ctx.exported_symbols:
                    lines.append(f"Symboles exportés : {', '.join(ctx.exported_symbols)}")

        # Renames globaux applicables
        if self.global_renames:
            lines.append("\n=== SYMBOLES RENOMMÉS DANS LE PROJET ===")
            lines.append(
                "Si ce fichier utilise ces anciens noms, utilise les nouveaux :"
            )
            for old, change in self.global_renames.items():
                lines.append(
                    f"  {change.kind}: '{old}' → '{change.new_name}' "
                    f"(défini dans {change.source_file})"
                )

        # Conventions globales
        if self.global_conventions:
            lines.append("\n=== CONVENTIONS ADOPTÉES DANS CE PROJET ===")
            for conv in self.global_conventions[:8]:  # Max 8 pour éviter le bruit
                lines.append(f"  - {conv}")

        return "\n".join(lines) if lines else ""

    def summary(self) -> str:
        return (
            f"{len(self.file_contexts)} fichier(s) traité(s) · "
            f"{len(self.global_renames)} rename(s) · "
            f"{len(self.global_conventions)} convention(s)"
        )


# ──────────────────────────────────────────────
# Extracteur de contexte
# ──────────────────────────────────────────────

class ContextExtractor:
    """
    Extrait le contexte utile d'une réponse de refactorisation.

    Analyse la section "Changements effectués" et le code refactorisé
    pour identifier les renames, nouvelles signatures, et conventions.
    """

    # Patterns pour détecter les renames dans la section "Changements effectués"
    RENAME_PATTERNS = [
        r"`(\w+)`\s*(?:renommé|renamed|→|->)\s*`(\w+)`",
        r"(?:Renamed?|Renommé)\s+`(\w+)`\s+(?:to|en|→|->)\s+`(\w+)`",
        r"`(\w+)`\s+(?:devient|becomes)\s+`(\w+)`",
    ]

    # Patterns pour extraire les signatures publiques Python
    PYTHON_PUBLIC_PATTERNS = [
        r"^((?:async )?def [a-z][^:]+:)",          # fonctions publiques
        r"^(class \w+[^:]*:)",                      # classes
    ]

    # Patterns pour JS/TS exports
    JS_EXPORT_PATTERNS = [
        r"^(export (?:default )?(?:function|class|const|let|var) \w+[^\{]*)",
        r"^(export (?:interface|type) \w+[^\{]*)",
    ]

    def extract(
        self,
        refactored_code: str,
        full_response: str,
        source_file_path: str,
        language: str,
    ) -> FileContext:
        """
        Extrait le contexte complet d'un fichier refactorisé.

        Args:
            refactored_code: Code refactorisé (extrait du bloc markdown).
            full_response: Réponse complète du LLM (inclut l'analyse).
            source_file_path: Chemin relatif du fichier.
            language: Langage de programmation.

        Returns:
            FileContext prêt à être ajouté au SharedContext.
        """
        ctx = FileContext(
            relative_path=source_file_path,
            language=language,
        )

        # 1. Extraire les renames depuis la section "Changements effectués"
        changes_section = self._extract_section(full_response, "Changements effectués")
        ctx.symbol_changes = self._extract_renames(changes_section, source_file_path)

        # 2. Extraire les symboles publics et l'API publique
        ctx.exported_symbols, ctx.public_api_snippet = self._extract_public_api(
            refactored_code, language
        )

        # 3. Extraire les conventions depuis la réponse
        ctx.conventions = self._extract_conventions(changes_section, language)

        return ctx

    def _extract_section(self, text: str, section_name: str) -> str:
        """Extrait le contenu d'une section markdown."""
        pattern = rf"##\s+{re.escape(section_name)}\s*\n(.*?)(?=##|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_renames(self, changes_text: str, source_file: str) -> list[SymbolChange]:
        """Détecte les renames dans la section changements."""
        renames: list[SymbolChange] = []
        for pattern in self.RENAME_PATTERNS:
            for match in re.finditer(pattern, changes_text, re.IGNORECASE):
                old, new = match.group(1), match.group(2)
                if old != new:
                    kind = self._guess_kind(old, new)
                    renames.append(SymbolChange(
                        old_name=old,
                        new_name=new,
                        kind=kind,
                        source_file=source_file,
                    ))
        return renames

    def _extract_public_api(
        self, code: str, language: str
    ) -> tuple[list[str], str]:
        """Extrait les symboles publics et un snippet de l'API publique."""
        symbols: list[str] = []
        snippet_lines: list[str] = []

        if language == "Python":
            for pattern in self.PYTHON_PUBLIC_PATTERNS:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    sig = match.group(1).strip()
                    snippet_lines.append(sig)
                    # Extraire le nom du symbole
                    name_match = re.search(r"(?:def|class)\s+(\w+)", sig)
                    if name_match:
                        symbols.append(name_match.group(1))

        elif language in ("JavaScript", "TypeScript"):
            for pattern in self.JS_EXPORT_PATTERNS:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    sig = match.group(1).strip()
                    snippet_lines.append(sig)
                    name_match = re.search(
                        r"(?:function|class|const|let|var|interface|type)\s+(\w+)", sig
                    )
                    if name_match:
                        symbols.append(name_match.group(1))

        else:
            # Fallback : chercher des patterns de fonctions/classes génériques
            for match in re.finditer(
                r"^(?:public|export|func|def)\s+\w+\s+(\w+)\s*\(", code, re.MULTILINE
            ):
                symbols.append(match.group(1))

        # Limiter le snippet à 20 lignes max pour ne pas saturer le contexte
        snippet = "\n".join(snippet_lines[:20])
        return symbols[:30], snippet

    def _extract_conventions(self, changes_text: str, language: str) -> list[str]:
        """Extrait les conventions adoptées mentionnées dans les changements."""
        conventions: list[str] = []

        CONVENTION_KEYWORDS = [
            "snake_case", "camelCase", "PascalCase",
            "type hints", "annotations de types", "docstrings",
            "async/await", "f-strings", "list comprehension",
            "dataclass", "pydantic", "const", "readonly",
            "single responsibility", "dependency injection",
        ]

        text_lower = changes_text.lower()
        for keyword in CONVENTION_KEYWORDS:
            if keyword.lower() in text_lower:
                conventions.append(keyword)

        # Conventions spécifiques au langage
        if language == "Python" and "type hint" in text_lower:
            conventions.append("annotations de types Python")

        return conventions

    @staticmethod
    def _guess_kind(old_name: str, new_name: str) -> str:
        """Devine le type de symbole à partir de la casse."""
        if old_name[0].isupper():
            return "class"
        if old_name.isupper():
            return "variable"
        return "function"
