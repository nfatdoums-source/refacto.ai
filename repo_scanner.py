"""
Scanner de repository — découverte et filtrage des fichiers source.

Responsabilités :
- Parcourir récursivement un dossier
- Filtrer les fichiers pertinents (langage, taille, exclusions)
- Retourner une liste ordonnée de fichiers à traiter
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────

# Extension → Langage
EXT_TO_LANG: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "JavaScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".cs": "C#",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".rb": "Ruby",
}

# Dossiers toujours exclus
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "env", ".env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "out", "target", "bin", "obj",
    ".idea", ".vscode", ".vs",
    "coverage", ".coverage", "htmlcov",
    "vendor", "third_party", "extern",
})

# Fichiers toujours exclus
EXCLUDED_FILES: frozenset[str] = frozenset({
    "setup.py", "conftest.py",
    "manage.py",  # Django boilerplate
})

MAX_FILE_SIZE_BYTES = 100_000   # 100 Ko — fichiers trop gros ignorés
MAX_FILES_PER_REPO = 500        # Limite de sécurité


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class SourceFile:
    """Représente un fichier source à refactoriser."""
    path: Path
    language: str
    size_bytes: int
    relative_path: Path  # Relatif à la racine du repo

    @property
    def content(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024


@dataclass
class ScanResult:
    """Résultat d'un scan de repository."""
    root: Path
    files: list[SourceFile] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_size_kb(self) -> float:
        return sum(f.size_kb for f in self.files)

    @property
    def languages(self) -> dict[str, int]:
        """Compte de fichiers par langage."""
        result: dict[str, int] = {}
        for f in self.files:
            result[f.language] = result.get(f.language, 0) + 1
        return result

    def summary(self) -> str:
        langs = ", ".join(f"{lang} ({n})" for lang, n in sorted(self.languages.items()))
        return (
            f"{self.total_files} fichier(s) trouvé(s) "
            f"({self.total_size_kb:.1f} Ko total)\n"
            f"Langages : {langs or 'aucun'}\n"
            f"{len(self.skipped_paths)} fichier(s) ignoré(s)"
        )


# ──────────────────────────────────────────────
# Scanner
# ──────────────────────────────────────────────

class RepoScanner:
    """
    Parcourt un repository et retourne la liste des fichiers source à traiter.

    Example:
        >>> scanner = RepoScanner()
        >>> result = scanner.scan("/path/to/repo")
        >>> print(result.summary())
    """

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        max_files: int = MAX_FILES_PER_REPO,
        extra_excluded_dirs: set[str] | None = None,
    ) -> None:
        self._max_file_size = max_file_size
        self._max_files = max_files
        self._excluded_dirs = EXCLUDED_DIRS | (extra_excluded_dirs or set())

    def scan(self, repo_path: str | Path) -> ScanResult:
        """
        Scanne un dossier et retourne les fichiers source éligibles.

        Args:
            repo_path: Chemin vers la racine du repository.

        Returns:
            ScanResult avec la liste des fichiers et les exclusions.

        Raises:
            ValueError: Si le chemin n'existe pas ou n'est pas un dossier.
        """
        root = Path(repo_path).resolve()
        if not root.exists():
            raise ValueError(f"Dossier introuvable : {root}")
        if not root.is_dir():
            raise ValueError(f"Le chemin n'est pas un dossier : {root}")

        result = ScanResult(root=root)
        logger.info("Scan du repository : %s", root)

        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue

            # Vérifier les dossiers exclus
            if self._is_in_excluded_dir(file_path, root):
                continue

            # Vérifier l'extension
            lang = EXT_TO_LANG.get(file_path.suffix.lower())
            if not lang:
                continue

            # Vérifier le nom de fichier
            if file_path.name in EXCLUDED_FILES:
                result.skipped_paths.append(str(file_path.relative_to(root)))
                result.skipped_reasons.append("fichier boilerplate exclu")
                continue

            # Vérifier la taille
            size = file_path.stat().st_size
            if size == 0:
                result.skipped_paths.append(str(file_path.relative_to(root)))
                result.skipped_reasons.append("fichier vide")
                continue
            if size > self._max_file_size:
                result.skipped_paths.append(str(file_path.relative_to(root)))
                result.skipped_reasons.append(f"trop volumineux ({size / 1024:.0f} Ko > {self._max_file_size / 1024:.0f} Ko)")
                continue

            # Limite de sécurité
            if len(result.files) >= self._max_files:
                result.skipped_paths.append(str(file_path.relative_to(root)))
                result.skipped_reasons.append(f"limite de {self._max_files} fichiers atteinte")
                continue

            result.files.append(SourceFile(
                path=file_path,
                language=lang,
                size_bytes=size,
                relative_path=file_path.relative_to(root),
            ))

        logger.info("Scan terminé : %s", result.summary())
        return result

    def _is_in_excluded_dir(self, file_path: Path, root: Path) -> bool:
        """Retourne True si le fichier est dans un dossier exclu."""
        try:
            relative = file_path.relative_to(root)
        except ValueError:
            return True
        return any(part in self._excluded_dirs for part in relative.parts)
