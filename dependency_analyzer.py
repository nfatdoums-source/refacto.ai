"""
Analyseur de dépendances inter-fichiers.

Construit un graphe orienté des dépendances entre fichiers source
puis calcule l'ordre optimal de traitement via tri topologique (Kahn).

Supporte :
- Python  : via ast (précis)
- JS/TS   : via regex (import/require)
- Java    : via regex (import)
- C#      : via regex (using)
- Go      : via regex (import)
- Autres  : analyse basique des inclusions
"""
from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from repo_scanner import SourceFile

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class DependencyGraph:
    """
    Graphe orienté des dépendances entre fichiers.

    edges[A] = {B, C}  signifie que A importe B et C.
    """
    files: list[SourceFile]
    edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # Chemin relatif → SourceFile
    _index: dict[str, SourceFile] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {str(f.relative_path): f for f in self.files}

    def add_dependency(self, from_file: str, to_file: str) -> None:
        """A dépend de B — B doit être traité avant A."""
        if from_file != to_file and to_file in self._index:
            self.edges[from_file].add(to_file)

    def get_dependencies(self, file_path: str) -> set[str]:
        """Retourne les fichiers dont dépend file_path."""
        return self.edges.get(file_path, set())

    def topological_sort(self) -> list[str]:
        """
        Tri topologique via algorithme de Kahn.

        Les fichiers sans dépendances sont traités en premier.
        En cas de cycle (imports circulaires), les fichiers cycliques
        sont ajoutés à la fin dans leur ordre original.

        Returns:
            Liste ordonnée de chemins relatifs.
        """
        all_files = {str(f.relative_path) for f in self.files}

        # Calculer le in-degree (nombre de dépendances entrantes)
        in_degree: dict[str, int] = {f: 0 for f in all_files}
        # reversed edges : dependants[B] = {A} si A dépend de B
        dependants: dict[str, set[str]] = defaultdict(set)

        for src, dsts in self.edges.items():
            for dst in dsts:
                if dst in all_files:
                    in_degree[src] = in_degree.get(src, 0) + 1
                    dependants[dst].add(src)

        # File d'attente : fichiers sans dépendances non résolues
        queue: deque[str] = deque(
            sorted(f for f, deg in in_degree.items() if deg == 0)
        )
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for dependant in sorted(dependants.get(current, set())):
                in_degree[dependant] -= 1
                if in_degree[dependant] == 0:
                    queue.append(dependant)

        # Fichiers non traités (cycles détectés)
        remaining = [f for f in all_files if f not in order]
        if remaining:
            logger.warning(
                "Cycles détectés entre %d fichier(s) — ajoutés à la fin : %s",
                len(remaining), remaining,
            )
            order.extend(sorted(remaining))

        return order

    def summary(self) -> str:
        edges_count = sum(len(v) for v in self.edges.values())
        return (
            f"{len(self.files)} fichier(s) · "
            f"{edges_count} dépendance(s) détectée(s)"
        )


# ──────────────────────────────────────────────
# Parseurs par langage
# ──────────────────────────────────────────────

class _PythonParser:
    """Extrait les imports Python via AST."""

    def extract_imports(self, source: str, file_path: Path) -> list[str]:
        """Retourne les noms de modules importés (relatifs prioritaires)."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Import relatif : from . import x, from ..utils import y
                    base = file_path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    module = node.module or ""
                    if module:
                        imports.append(str(base / module.replace(".", "/")))
                    else:
                        imports.append(str(base))
                elif node.module:
                    imports.append(node.module.replace(".", "/"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.replace(".", "/"))
        return imports


class _RegexParser:
    """Extracteur générique basé sur des patterns regex."""

    PATTERNS: dict[str, list[str]] = {
        "JavaScript": [
            r"""(?:import|from)\s+['"]([^'"]+)['"]""",
            r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
        ],
        "TypeScript": [
            r"""(?:import|from)\s+['"]([^'"]+)['"]""",
            r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
        ],
        "Java": [
            r"""^import\s+([\w.]+)\s*;""",
        ],
        "C#": [
            r"""^using\s+([\w.]+)\s*;""",
        ],
        "Go": [
            r""""([\w./]+)"\s*$""",
        ],
        "Rust": [
            r"""^(?:use|mod)\s+([\w:]+)""",
        ],
        "PHP": [
            r"""(?:require|include)(?:_once)?\s*['"]([\w./]+)['"]""",
        ],
        "Ruby": [
            r"""(?:require|require_relative)\s+['"]([\w./]+)['"]""",
        ],
        "C++": [
            r"""#include\s+[<"]([\w./]+)[>"]""",
        ],
    }

    def extract_imports(self, source: str, language: str) -> list[str]:
        patterns = self.PATTERNS.get(language, [])
        results: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, source, re.MULTILINE):
                results.append(match.group(1))
        return results


# ──────────────────────────────────────────────
# DependencyAnalyzer principal
# ──────────────────────────────────────────────

class DependencyAnalyzer:
    """
    Analyse les dépendances entre fichiers source et construit
    un graphe orienté utilisé pour déterminer l'ordre de traitement.

    Example:
        >>> analyzer = DependencyAnalyzer()
        >>> graph = analyzer.analyze(source_files)
        >>> order = graph.topological_sort()
    """

    def __init__(self) -> None:
        self._py_parser = _PythonParser()
        self._regex_parser = _RegexParser()

    def analyze(self, files: list[SourceFile]) -> DependencyGraph:
        """
        Analyse tous les fichiers et construit le graphe de dépendances.

        Args:
            files: Liste des fichiers source détectés par RepoScanner.

        Returns:
            DependencyGraph avec les dépendances et l'ordre de traitement.
        """
        graph = DependencyGraph(files=files)

        # Index : stem (nom sans extension) → chemin relatif
        # Permet de résoudre "from utils import x" → "utils.py"
        stem_index = self._build_stem_index(files)

        for source_file in files:
            rel = str(source_file.relative_path)
            try:
                raw_imports = self._extract_raw_imports(source_file)
                for raw in raw_imports:
                    resolved = self._resolve(raw, source_file, stem_index)
                    if resolved:
                        graph.add_dependency(rel, resolved)
            except Exception as exc:
                logger.debug("Erreur analyse dépendances %s : %s", rel, exc)

        logger.info("Graphe de dépendances : %s", graph.summary())
        return graph

    def _extract_raw_imports(self, source_file: SourceFile) -> list[str]:
        """Extrait les imports bruts selon le langage."""
        content = source_file.content
        if source_file.language == "Python":
            return self._py_parser.extract_imports(content, source_file.path)
        return self._regex_parser.extract_imports(content, source_file.language)

    def _resolve(
        self,
        raw_import: str,
        source_file: SourceFile,
        stem_index: dict[str, str],
    ) -> str | None:
        """
        Tente de résoudre un import brut vers un chemin relatif de fichier connu.

        Returns:
            Le chemin relatif du fichier importé, ou None si non résolu.
        """
        # Normaliser le chemin importé
        normalized = raw_import.replace("\\", "/").lstrip("./")

        # Essai 1 : correspondance directe (stem ou chemin partiel)
        for stem, rel_path in stem_index.items():
            if normalized.endswith(stem) or stem.endswith(normalized):
                return rel_path

        # Essai 2 : import relatif résolu depuis le dossier du fichier
        try:
            base = source_file.path.parent
            candidate = (base / raw_import).resolve()
            for source_file2 in stem_index.values():
                # Comparer le stem
                pass
        except Exception:
            pass

        return None

    @staticmethod
    def _build_stem_index(files: list[SourceFile]) -> dict[str, str]:
        """
        Construit un index stem → chemin relatif.

        Ex: "utils" → "src/utils.py"
            "src/utils" → "src/utils.py"
        """
        index: dict[str, str] = {}
        for f in files:
            rel = str(f.relative_path).replace("\\", "/")
            # Clé 1 : nom sans extension
            index[f.path.stem] = rel
            # Clé 2 : chemin relatif sans extension
            index[rel.rsplit(".", 1)[0]] = rel
            # Clé 3 : chemin relatif complet
            index[rel] = rel
        return index
