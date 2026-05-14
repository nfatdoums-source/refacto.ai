"""
Agent de refactorisation multi-fichiers.

Orchestre les 4 phases du pipeline :
  1. Scan du repository
  2. Analyse des dépendances + tri topologique
  3. Refactorisation ordonnée avec contexte partagé
  4. Propagation des renames + génération du ZIP

C'est ce module qui assure la cohérence entre fichiers :
chaque fichier est refactorisé en connaissant les changements
effectués dans les fichiers dont il dépend.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
import shutil
import time
import zipfile
from collections.abc import Generator
from dataclasses import dataclass, field, asdict
from pathlib import Path

from refactorer import CodeRefactorer, RefactorerError
from repo_scanner import RepoScanner, ScanResult, SourceFile
from dependency_analyzer import DependencyAnalyzer, DependencyGraph
from context_manager import SharedContext, ContextExtractor, FileContext
from prompts import build_refactoring_prompt, get_mode_instruction

logger = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "refactored_output"


# ──────────────────────────────────────────────
# Dataclasses de rapport
# ──────────────────────────────────────────────

@dataclass
class AgentFileResult:
    relative_path: str
    language: str
    status: str                 # "success" | "error"
    error_message: str = ""
    duration_seconds: float = 0.0
    renames_detected: int = 0
    had_dependency_context: bool = False


@dataclass
class AgentReport:
    repo_path: str
    mode: str
    started_at: str
    processing_order: list[str] = field(default_factory=list)
    dependency_edges: int = 0
    total_files: int = 0
    succeeded: int = 0
    failed: int = 0
    total_duration_seconds: float = 0.0
    global_renames: int = 0
    file_results: list[AgentFileResult] = field(default_factory=list)
    output_dir: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def summary(self) -> str:
        return (
            f"✅ **{self.succeeded}** succès  |  "
            f"❌ **{self.failed}** erreurs\n"
            f"🔗 **{self.dependency_edges}** dépendance(s) analysée(s)  |  "
            f"🔄 **{self.global_renames}** rename(s) propagé(s)\n"
            f"⏱️ Durée totale : **{self.total_duration_seconds:.1f}s**"
        )


# ──────────────────────────────────────────────
# ProgressEvent (réutilisé depuis batch_refactorer)
# ──────────────────────────────────────────────

@dataclass
class ProgressEvent:
    current: int
    total: int
    file_path: str
    status: str
    message: str = ""
    log_line: str = ""

    @property
    def fraction(self) -> float:
        return self.current / max(self.total, 1)

    @property
    def percent(self) -> int:
        return int(self.fraction * 100)


# ──────────────────────────────────────────────
# Prompt avec contexte partagé
# ──────────────────────────────────────────────

def build_context_aware_prompt(
    code: str,
    language: str,
    mode: str,
    shared_context_text: str,
) -> str:
    """
    Construit le prompt complet pour un fichier avec contexte injecté.

    Args:
        code: Code source à refactoriser.
        language: Langage de programmation.
        mode: Mode de refactorisation.
        shared_context_text: Contexte des fichiers dépendants déjà traités.

    Returns:
        Prompt utilisateur complet.
    """
    context_block = ""
    if shared_context_text:
        context_block = f"""
{shared_context_text}

IMPORTANT : Respecte impérativement les noms et interfaces définis ci-dessus.
Si ce fichier importe des symboles renommés, utilise leurs nouveaux noms.

---
"""

    return f"""Langage : {language}
Mode : {mode}
Instruction spécifique : {get_mode_instruction(mode)}
{context_block}
Code legacy à refactoriser :
```{language.lower()}
{code}
```"""


# ──────────────────────────────────────────────
# Agent principal
# ──────────────────────────────────────────────

class MultiFileAgent:
    """
    Agent de refactorisation multi-fichiers cohérent.

    Contrairement au BatchRefactorer simple, cet agent :
    - Analyse les dépendances entre fichiers
    - Trie les fichiers dans l'ordre optimal (dépendances en premier)
    - Maintient un contexte partagé (renames, signatures, conventions)
    - Injecte ce contexte dans le prompt de chaque fichier dépendant
    - Propage les renames dans les fichiers déjà écrits si nécessaire

    Example:
        >>> agent = MultiFileAgent()
        >>> for event in agent.run("/path/to/repo", "Refactorisation complète"):
        ...     print(f"{event.percent}% — {event.log_line}")
    """

    def __init__(self) -> None:
        self._refactorer = CodeRefactorer()
        self._scanner = RepoScanner()
        self._analyzer = DependencyAnalyzer()
        self._extractor = ContextExtractor()

    def run(
        self,
        repo_path: str,
        mode: str,
        output_base: str | None = None,
    ) -> Generator[ProgressEvent, None, None]:
        """
        Lance la refactorisation multi-fichiers complète.

        Args:
            repo_path: Chemin vers la racine du repository.
            mode: Mode de refactorisation.
            output_base: Dossier de sortie (défaut : parent du repo).

        Yields:
            ProgressEvent à chaque étape.
        """
        root = Path(repo_path).resolve()
        output_dir = Path(output_base or root.parent) / OUTPUT_DIR_NAME

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1 : Scan ──────────────────────
        yield ProgressEvent(0, 1, "", "processing", "🔍 Scan du repository...")
        try:
            scan: ScanResult = self._scanner.scan(root)
        except ValueError as exc:
            yield ProgressEvent(0, 1, "", "error", str(exc))
            return

        if scan.total_files == 0:
            yield ProgressEvent(0, 1, "", "error", "Aucun fichier source trouvé.")
            return

        # ── Phase 2 : Analyse des dépendances ───
        yield ProgressEvent(0, scan.total_files, "", "processing",
                            f"🔗 Analyse des dépendances ({scan.total_files} fichiers)...")
        graph: DependencyGraph = self._analyzer.analyze(scan.files)
        processing_order: list[str] = graph.topological_sort()

        edges_count = sum(len(v) for v in graph.edges.values())
        yield ProgressEvent(
            0, scan.total_files, "", "processing",
            f"📊 Graphe : {edges_count} dépendance(s) · "
            f"Ordre : {' → '.join(Path(p).name for p in processing_order[:5])}"
            + (" → ..." if len(processing_order) > 5 else ""),
            log_line=(
                f"[Analyse] {scan.total_files} fichiers · "
                f"{edges_count} dépendances détectées\n"
                f"[Ordre] {' → '.join(Path(p).name for p in processing_order)}"
            ),
        )

        # ── Phase 3 : Refactorisation ordonnée ──
        shared_ctx = SharedContext(language=scan.files[0].language if scan.files else "", mode=mode)
        report = AgentReport(
            repo_path=str(root),
            mode=mode,
            started_at=datetime.datetime.now().isoformat(),
            total_files=scan.total_files,
            processing_order=processing_order,
            dependency_edges=edges_count,
            output_dir=str(output_dir),
        )

        global_start = time.perf_counter()
        file_index = {str(f.relative_path): f for f in scan.files}

        for idx, rel_path in enumerate(processing_order, start=1):
            source_file = file_index.get(rel_path)
            if not source_file:
                continue

            dependencies = graph.get_dependencies(rel_path)
            context_text = shared_ctx.get_relevant_context(rel_path, dependencies)
            has_context = bool(context_text)

            yield ProgressEvent(
                current=idx,
                total=scan.total_files,
                file_path=rel_path,
                status="processing",
                message=f"Refactorisation de `{rel_path}`"
                        + (" (avec contexte)" if has_context else "") + "...",
                log_line=(
                    f"[{idx}/{scan.total_files}] ⏳ {rel_path}"
                    + (f" | contexte: {len(dependencies)} dép." if has_context else "")
                ),
            )

            file_start = time.perf_counter()
            try:
                full_response = self._refactor_with_context(
                    source_file=source_file,
                    mode=mode,
                    shared_context_text=context_text,
                )
                duration = time.perf_counter() - file_start

                # Extraire le code propre
                refactored_code = self._extract_code(full_response, source_file.language)

                # Écrire le fichier
                dest = output_dir / source_file.relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(refactored_code, encoding="utf-8")

                # Mettre à jour le contexte partagé
                file_ctx: FileContext = self._extractor.extract(
                    refactored_code=refactored_code,
                    full_response=full_response,
                    source_file_path=rel_path,
                    language=source_file.language,
                )
                shared_ctx.add_file_context(file_ctx)

                renames = len(file_ctx.symbol_changes)
                report.succeeded += 1
                report.file_results.append(AgentFileResult(
                    relative_path=rel_path,
                    language=source_file.language,
                    status="success",
                    duration_seconds=round(duration, 2),
                    renames_detected=renames,
                    had_dependency_context=has_context,
                ))

                yield ProgressEvent(
                    current=idx,
                    total=scan.total_files,
                    file_path=rel_path,
                    status="success",
                    message=f"`{rel_path}` ✓ ({duration:.1f}s"
                            + (f" · {renames} rename(s)" if renames else "") + ")",
                    log_line=(
                        f"[{idx}/{scan.total_files}] ✅ {rel_path} "
                        f"({duration:.1f}s"
                        + (f" · {renames} rename(s) détecté(s)" if renames else "")
                        + ")"
                    ),
                )

            except Exception as exc:
                duration = time.perf_counter() - file_start
                logger.error("Erreur sur %s : %s", rel_path, exc)

                # Copier le fichier original en fallback
                dest = output_dir / source_file.relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source_file.content, encoding="utf-8")

                report.failed += 1
                report.file_results.append(AgentFileResult(
                    relative_path=rel_path,
                    language=source_file.language,
                    status="error",
                    error_message=str(exc),
                    duration_seconds=round(duration, 2),
                ))

                yield ProgressEvent(
                    current=idx,
                    total=scan.total_files,
                    file_path=rel_path,
                    status="error",
                    message=f"Erreur sur `{rel_path}` : {exc}",
                    log_line=f"[{idx}/{scan.total_files}] ❌ {rel_path} — {exc}",
                )

        # ── Phase 4 : Propagation des renames ───
        if shared_ctx.global_renames:
            yield ProgressEvent(
                scan.total_files, scan.total_files, "", "processing",
                f"🔄 Propagation de {len(shared_ctx.global_renames)} rename(s)...",
                log_line=f"\n[Phase 4] Propagation de {len(shared_ctx.global_renames)} rename(s)...",
            )
            self._propagate_renames(output_dir, shared_ctx)

        # ── Rapport final ────────────────────────
        report.total_duration_seconds = round(time.perf_counter() - global_start, 2)
        report.global_renames = len(shared_ctx.global_renames)

        report_path = output_dir / "agent_report.json"
        report_path.write_text(report.to_json(), encoding="utf-8")

        # Générer le contexte lisible
        context_summary_path = output_dir / "refactoring_context.md"
        context_summary_path.write_text(
            self._build_context_summary(shared_ctx, report), encoding="utf-8"
        )

        # ZIP
        zip_path = output_dir.parent / "refactored_output.zip"
        self._create_zip(output_dir, zip_path)

        yield ProgressEvent(
            current=scan.total_files,
            total=scan.total_files,
            file_path="",
            status="done",
            message=report.summary(),
            log_line=f"\n🏁 Agent terminé\n{report.summary()}\n📦 ZIP : {zip_path}",
        )

    def _refactor_with_context(
        self,
        source_file: SourceFile,
        mode: str,
        shared_context_text: str,
    ) -> str:
        """Appelle le LLM avec le contexte partagé injecté dans le prompt."""
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        from prompts import SYSTEM_BASE
        from config import lm_studio_cfg

        llm = ChatOpenAI(
            base_url=lm_studio_cfg.base_url,
            api_key=lm_studio_cfg.api_key,
            model=lm_studio_cfg.model,
            temperature=lm_studio_cfg.temperature,
            max_tokens=lm_studio_cfg.max_tokens,
            streaming=False,
            request_timeout=lm_studio_cfg.request_timeout,
        )

        system = SYSTEM_BASE.format(language=source_file.language.lower())
        human_content = build_context_aware_prompt(
            code=source_file.content,
            language=source_file.language,
            mode=mode,
            shared_context_text=shared_context_text,
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template("{human}"),
        ])

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"human": human_content})

    @staticmethod
    def _extract_code(response: str, language: str) -> str:
        """Extrait le bloc de code de la réponse markdown."""
        pattern = rf"```{re.escape(language.lower())}\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        fallback = re.search(r"```\w*\n(.*?)```", response, re.DOTALL)
        return fallback.group(1).strip() if fallback else response

    @staticmethod
    def _propagate_renames(output_dir: Path, shared_ctx: SharedContext) -> None:
        """
        Phase 4 : Applique les renames détectés dans tous les fichiers de sortie.

        Parcourt tous les fichiers générés et remplace les anciens noms
        par les nouveaux dans les appels, imports et références.
        """
        if not shared_ctx.global_renames:
            return

        for file_path in output_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix == ".json":
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                modified = content
                for old_name, change in shared_ctx.global_renames.items():
                    # Remplacement avec boundaries pour éviter les faux positifs
                    pattern = rf"\b{re.escape(old_name)}\b"
                    modified = re.sub(pattern, change.new_name, modified)
                if modified != content:
                    file_path.write_text(modified, encoding="utf-8")
                    logger.debug("Renames propagés dans %s", file_path.name)
            except Exception as exc:
                logger.warning("Impossible de propager les renames dans %s : %s", file_path, exc)

    @staticmethod
    def _build_context_summary(shared_ctx: SharedContext, report: AgentReport) -> str:
        """Génère un rapport markdown lisible de la session."""
        lines = [
            "# Rapport de refactorisation multi-fichiers",
            "",
            f"- **Mode** : {report.mode}",
            f"- **Démarré** : {report.started_at}",
            f"- **Durée** : {report.total_duration_seconds}s",
            f"- **Fichiers traités** : {report.succeeded}/{report.total_files}",
            f"- **Dépendances analysées** : {report.dependency_edges}",
            f"- **Renames propagés** : {report.global_renames}",
            "",
            "## Ordre de traitement",
            "",
        ]
        for i, path in enumerate(report.processing_order, 1):
            lines.append(f"{i}. `{path}`")

        if shared_ctx.global_renames:
            lines += ["", "## Symboles renommés", ""]
            for old, change in shared_ctx.global_renames.items():
                lines.append(
                    f"- `{old}` → `{change.new_name}` "
                    f"({change.kind}, défini dans `{change.source_file}`)"
                )

        if shared_ctx.global_conventions:
            lines += ["", "## Conventions adoptées", ""]
            for conv in shared_ctx.global_conventions:
                lines.append(f"- {conv}")

        return "\n".join(lines)

    @staticmethod
    def _create_zip(source_dir: Path, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in source_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(source_dir.parent))
        logger.info("ZIP créé : %s", zip_path)
