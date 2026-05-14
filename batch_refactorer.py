"""
Moteur de refactorisation en batch pour un repository complet.

Traite les fichiers un par un, génère les fichiers refactorisés
dans un dossier de sortie et produit un rapport JSON détaillé.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import zipfile
from collections.abc import Generator
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from refactorer import CodeRefactorer, RefactorerError
from repo_scanner import RepoScanner, ScanResult, SourceFile

logger = logging.getLogger(__name__)

OUTPUT_DIR_NAME = "refactored_output"


# ──────────────────────────────────────────────
# Dataclasses de rapport
# ──────────────────────────────────────────────

@dataclass
class FileResult:
    """Résultat du traitement d'un fichier."""
    relative_path: str
    language: str
    size_bytes: int
    status: str          # "success" | "error" | "skipped"
    error_message: str = ""
    duration_seconds: float = 0.0


@dataclass
class BatchReport:
    """Rapport complet d'une session de refactorisation en batch."""
    repo_path: str
    mode: str
    started_at: str
    total_files: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_seconds: float = 0.0
    file_results: list[FileResult] = field(default_factory=list)
    output_dir: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def summary(self) -> str:
        return (
            f"✅ **{self.succeeded}** succès  |  "
            f"❌ **{self.failed}** erreurs  |  "
            f"⏭️ **{self.skipped}** ignorés\n"
            f"⏱️ Durée totale : **{self.total_duration_seconds:.1f}s**"
        )


# ──────────────────────────────────────────────
# Progress event
# ──────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Événement de progression émis pendant le traitement."""
    current: int
    total: int
    file_path: str
    status: str          # "processing" | "success" | "error" | "done"
    message: str = ""
    log_line: str = ""

    @property
    def fraction(self) -> float:
        return self.current / max(self.total, 1)

    @property
    def percent(self) -> int:
        return int(self.fraction * 100)


# ──────────────────────────────────────────────
# BatchRefactorer
# ──────────────────────────────────────────────

class BatchRefactorer:
    """
    Refactorise un repository complet fichier par fichier.

    Yields des ProgressEvent pour alimenter l'interface en temps réel.

    Example:
        >>> batch = BatchRefactorer()
        >>> for event in batch.run("/path/to/repo", "Refactorisation complète"):
        ...     print(f"{event.percent}% — {event.file_path}")
    """

    def __init__(self) -> None:
        self._refactorer = CodeRefactorer()
        self._scanner = RepoScanner()

    def run(
        self,
        repo_path: str,
        mode: str,
        output_base: str | None = None,
    ) -> Generator[ProgressEvent, None, None]:
        """
        Lance la refactorisation du repository.

        Args:
            repo_path: Chemin vers la racine du repository.
            mode: Mode de refactorisation (ex: "Refactorisation complète").
            output_base: Dossier de sortie (défaut: repo_path/../refactored_output).

        Yields:
            ProgressEvent à chaque étape du traitement.
        """
        root = Path(repo_path).resolve()

        # Dossier de sortie
        if output_base:
            output_dir = Path(output_base) / OUTPUT_DIR_NAME
        else:
            output_dir = root.parent / OUTPUT_DIR_NAME

        # Nettoyer et recréer le dossier de sortie
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan
        yield ProgressEvent(0, 1, "", "processing", "🔍 Scan du repository en cours...")
        try:
            scan: ScanResult = self._scanner.scan(root)
        except ValueError as exc:
            yield ProgressEvent(0, 1, "", "error", str(exc))
            return

        total = scan.total_files
        if total == 0:
            yield ProgressEvent(0, 1, "", "error",
                                "Aucun fichier source trouvé dans ce repository.")
            return

        # Rapport
        import datetime
        report = BatchReport(
            repo_path=str(root),
            mode=mode,
            started_at=datetime.datetime.now().isoformat(),
            total_files=total,
            output_dir=str(output_dir),
        )

        global_start = time.perf_counter()

        # Traitement fichier par fichier
        for idx, source_file in enumerate(scan.files, start=1):
            rel = str(source_file.relative_path)

            yield ProgressEvent(
                current=idx,
                total=total,
                file_path=rel,
                status="processing",
                message=f"Refactorisation de `{rel}`...",
                log_line=f"[{idx}/{total}] ⏳ {rel}",
            )

            file_start = time.perf_counter()
            try:
                refactored_code = self._refactorer.refactor_sync(
                    code=source_file.content,
                    language=source_file.language,
                    mode=mode,
                )
                duration = time.perf_counter() - file_start

                # Écrire le fichier refactorisé en préservant la structure
                dest = output_dir / source_file.relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(
                    self._extract_code(refactored_code, source_file.language),
                    encoding="utf-8",
                )

                report.succeeded += 1
                report.file_results.append(FileResult(
                    relative_path=rel,
                    language=source_file.language,
                    size_bytes=source_file.size_bytes,
                    status="success",
                    duration_seconds=round(duration, 2),
                ))

                yield ProgressEvent(
                    current=idx,
                    total=total,
                    file_path=rel,
                    status="success",
                    message=f"`{rel}` refactorisé en {duration:.1f}s",
                    log_line=f"[{idx}/{total}] ✅ {rel} ({duration:.1f}s)",
                )

            except (RefactorerError, Exception) as exc:
                duration = time.perf_counter() - file_start
                err_msg = str(exc)
                logger.error("Erreur sur %s : %s", rel, err_msg)

                # Copier le fichier original en cas d'erreur
                dest = output_dir / source_file.relative_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source_file.content, encoding="utf-8")

                report.failed += 1
                report.file_results.append(FileResult(
                    relative_path=rel,
                    language=source_file.language,
                    size_bytes=source_file.size_bytes,
                    status="error",
                    error_message=err_msg,
                    duration_seconds=round(duration, 2),
                ))

                yield ProgressEvent(
                    current=idx,
                    total=total,
                    file_path=rel,
                    status="error",
                    message=f"Erreur sur `{rel}` : {err_msg}",
                    log_line=f"[{idx}/{total}] ❌ {rel} — {err_msg}",
                )

            report.processed += 1

        # Rapport final
        report.total_duration_seconds = round(time.perf_counter() - global_start, 2)
        report_path = output_dir / "refactoring_report.json"
        report_path.write_text(report.to_json(), encoding="utf-8")

        # Créer le ZIP
        zip_path = output_dir.parent / "refactored_output.zip"
        self._create_zip(output_dir, zip_path)

        yield ProgressEvent(
            current=total,
            total=total,
            file_path="",
            status="done",
            message=report.summary(),
            log_line=f"\n🏁 Terminé — {report.summary()}\n📦 ZIP : {zip_path}",
        )

    @staticmethod
    def _extract_code(response: str, language: str) -> str:
        """Extrait le bloc de code de la réponse markdown du LLM."""
        import re
        pattern = rf"```{re.escape(language.lower())}\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        fallback = re.search(r"```\w*\n(.*?)```", response, re.DOTALL)
        if fallback:
            return fallback.group(1).strip()
        return response

    @staticmethod
    def _create_zip(source_dir: Path, zip_path: Path) -> None:
        """Compresse le dossier de sortie dans un ZIP téléchargeable."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in source_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(source_dir.parent))
        logger.info("ZIP créé : %s", zip_path)
