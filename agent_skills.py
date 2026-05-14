"""
Skills (tools) de l'agent ReAct.

Chaque skill est une fonction décorée avec @tool de LangChain.
Le LLM choisit lui-même quels skills appeler, dans quel ordre,
et avec quels arguments — en fonction de ce qu'il observe.

Skills disponibles :
  - scan_repo          : lister les fichiers source d'un dossier
  - read_file          : lire le contenu d'un fichier
  - analyze_deps       : analyser les dépendances entre fichiers
  - get_shared_context : récupérer le contexte accumulé
  - refactor_file      : refactoriser un fichier (appel LLM interne)
  - write_file         : écrire le résultat refactorisé
  - update_context     : mettre à jour la mémoire partagée
  - create_zip         : créer le ZIP final
  - get_processing_order : obtenir l'ordre optimal de traitement
"""
from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

from repo_scanner import RepoScanner
from dependency_analyzer import DependencyAnalyzer
from context_manager import ContextExtractor, SharedContext
from refactorer import CodeRefactorer
from config import lm_studio_cfg

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# État partagé de la session agent
# (le LLM y accède via les skills get/update)
# ──────────────────────────────────────────────
_session: dict = {
    "shared_context": None,
    "scan_result": None,
    "graph": None,
    "output_dir": None,
    "refactorer": None,
    "extractor": None,
}


def init_session(output_dir: str, mode: str) -> None:
    """Initialise la session avant de lancer l'agent."""
    _session["shared_context"] = SharedContext(language="", mode=mode)
    _session["output_dir"] = Path(output_dir)
    _session["output_dir"].mkdir(parents=True, exist_ok=True)
    _session["refactorer"] = CodeRefactorer()
    _session["extractor"] = ContextExtractor()
    _session["scan_result"] = None
    _session["graph"] = None


# ──────────────────────────────────────────────
# SKILL 1 — scan_repo
# ──────────────────────────────────────────────

@tool
def scan_repo(repo_path: Annotated[str, "Chemin absolu vers la racine du repository"]) -> str:
    """
    Scanne un repository et retourne la liste des fichiers source détectés.
    Filtre automatiquement node_modules, .git, __pycache__, binaires, etc.
    Utilise ce skill EN PREMIER pour découvrir les fichiers à traiter.
    """
    try:
        scanner = RepoScanner()
        result = scanner.scan(repo_path)
        _session["scan_result"] = result

        files_info = [
            {
                "path": str(f.relative_path),
                "language": f.language,
                "size_kb": round(f.size_kb, 1),
            }
            for f in result.files
        ]

        return json.dumps({
            "status": "ok",
            "total_files": result.total_files,
            "total_size_kb": round(result.total_size_kb, 1),
            "languages": result.languages,
            "files": files_info,
            "skipped": len(result.skipped_paths),
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 2 — analyze_deps
# ──────────────────────────────────────────────

@tool
def analyze_deps(repo_path: Annotated[str, "Chemin absolu vers la racine du repository"]) -> str:
    """
    Analyse les dépendances entre les fichiers du repository.
    Construit un graphe orienté (qui importe quoi).
    Utilise ce skill APRÈS scan_repo pour comprendre les relations entre fichiers.
    """
    if not _session.get("scan_result"):
        return json.dumps({"status": "error", "error": "Lance d'abord scan_repo."})

    try:
        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze(_session["scan_result"].files)
        _session["graph"] = graph

        edges = {k: list(v) for k, v in graph.edges.items() if v}

        return json.dumps({
            "status": "ok",
            "total_dependencies": sum(len(v) for v in graph.edges.values()),
            "dependency_edges": edges,
            "summary": graph.summary(),
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 3 — get_processing_order
# ──────────────────────────────────────────────

@tool
def get_processing_order() -> str:
    """
    Retourne l'ordre optimal de traitement des fichiers (tri topologique).
    Les fichiers sans dépendances viennent en premier.
    Utilise ce skill APRÈS analyze_deps.
    """
    if not _session.get("graph"):
        return json.dumps({"status": "error", "error": "Lance d'abord analyze_deps."})

    try:
        order = _session["graph"].topological_sort()
        return json.dumps({
            "status": "ok",
            "processing_order": order,
            "total": len(order),
            "explanation": "Traite les fichiers dans cet ordre exact — les dépendances sont résolues en premier.",
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 4 — read_file
# ──────────────────────────────────────────────

@tool
def read_file(
    relative_path: Annotated[str, "Chemin relatif du fichier (depuis la racine du repo)"],
    repo_path: Annotated[str, "Chemin absolu vers la racine du repository"],
) -> str:
    """
    Lit le contenu d'un fichier source.
    Utilise ce skill pour inspecter un fichier avant de le refactoriser.
    """
    try:
        full_path = Path(repo_path) / relative_path
        if not full_path.exists():
            return json.dumps({"status": "error", "error": f"Fichier introuvable : {relative_path}"})

        content = full_path.read_text(encoding="utf-8", errors="replace")
        size_kb = full_path.stat().st_size / 1024

        return json.dumps({
            "status": "ok",
            "relative_path": relative_path,
            "size_kb": round(size_kb, 1),
            "content": content[:8000],   # Limité pour ne pas saturer le contexte
            "truncated": len(content) > 8000,
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 5 — get_shared_context
# ──────────────────────────────────────────────

@tool
def get_shared_context(
    file_path: Annotated[str, "Chemin relatif du fichier en cours de traitement"],
) -> str:
    """
    Retourne le contexte accumulé des fichiers déjà refactorisés.
    Inclut les renames détectés, les signatures publiques et les conventions.
    Utilise ce skill AVANT de refactoriser un fichier pour connaître le contexte de ses dépendances.
    """
    ctx = _session.get("shared_context")
    if not ctx:
        return json.dumps({"status": "error", "error": "Session non initialisée."})

    graph = _session.get("graph")
    dependencies = graph.get_dependencies(file_path) if graph else set()
    context_text = ctx.get_relevant_context(file_path, dependencies)

    return json.dumps({
        "status": "ok",
        "files_processed_so_far": list(ctx.file_contexts.keys()),
        "global_renames": {k: v.new_name for k, v in ctx.global_renames.items()},
        "conventions": ctx.global_conventions,
        "context_for_prompt": context_text,
        "has_context": bool(context_text),
    }, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# SKILL 6 — refactor_file
# ──────────────────────────────────────────────

@tool
def refactor_file(
    relative_path: Annotated[str, "Chemin relatif du fichier à refactoriser"],
    repo_path: Annotated[str, "Chemin absolu vers la racine du repository"],
    language: Annotated[str, "Langage de programmation (Python, JavaScript, etc.)"],
    mode: Annotated[str, "Mode de refactorisation"],
    context_for_prompt: Annotated[str, "Contexte des fichiers dépendants (obtenu via get_shared_context)"],
) -> str:
    """
    Refactorise un fichier source en utilisant le LLM.
    Injecte le contexte des dépendances dans le prompt pour garantir la cohérence.
    Utilise ce skill après get_shared_context pour chaque fichier.
    """
    rf = _session.get("refactorer")
    if not rf:
        return json.dumps({"status": "error", "error": "Session non initialisée."})

    try:
        full_path = Path(repo_path) / relative_path
        code = full_path.read_text(encoding="utf-8", errors="replace")

        # Construction du prompt avec contexte injecté
        from prompts import get_mode_instruction, SYSTEM_BASE
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

        context_block = ""
        if context_for_prompt:
            context_block = f"\n{context_for_prompt}\n\nIMPORTANT : Respecte impérativement les noms et interfaces ci-dessus.\n---\n"

        human = f"""Langage : {language}
Mode : {mode}
Instruction : {get_mode_instruction(mode)}
{context_block}
Code à refactoriser :
```{language.lower()}
{code}
```"""

        llm = ChatOpenAI(
            base_url=lm_studio_cfg.base_url,
            api_key=lm_studio_cfg.api_key,
            model=lm_studio_cfg.model,
            temperature=0.2,
            max_tokens=2000,
            streaming=False,
            request_timeout=900,
            max_retries=0,
        )

        system = SYSTEM_BASE.format(language=language.lower())
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system),
            HumanMessagePromptTemplate.from_template("{human}"),
        ])
        chain = prompt | llm | StrOutputParser()
        full_response = chain.invoke({"human": human})

        # Extraire le code propre
        pattern = rf"```{re.escape(language.lower())}\n(.*?)```"
        match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
        if match:
            clean_code = match.group(1).strip()
        else:
            fallback = re.search(r"```\w*\n(.*?)```", full_response, re.DOTALL)
            clean_code = fallback.group(1).strip() if fallback else full_response

        # Mettre à jour le contexte partagé
        extractor = _session["extractor"]
        file_ctx = extractor.extract(
            refactored_code=clean_code,
            full_response=full_response,
            source_file_path=relative_path,
            language=language,
        )
        _session["shared_context"].add_file_context(file_ctx)

        return json.dumps({
            "status": "ok",
            "relative_path": relative_path,
            "refactored_code": clean_code,
            "full_response_preview": full_response[:500],
            "renames_detected": len(file_ctx.symbol_changes),
            "exported_symbols": file_ctx.exported_symbols[:10],
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.exception("Erreur refactor_file %s", relative_path)
        return json.dumps({"status": "error", "relative_path": relative_path, "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 7 — write_file
# ──────────────────────────────────────────────

@tool
def write_file(
    relative_path: Annotated[str, "Chemin relatif du fichier de sortie"],
    content: Annotated[str, "Contenu refactorisé à écrire"],
) -> str:
    """
    Écrit le code refactorisé dans le dossier de sortie.
    Préserve la structure de dossiers du repository original.
    Utilise ce skill APRÈS refactor_file pour sauvegarder le résultat.
    """
    output_dir = _session.get("output_dir")
    if not output_dir:
        return json.dumps({"status": "error", "error": "Session non initialisée."})

    try:
        dest = output_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

        return json.dumps({
            "status": "ok",
            "written_to": str(dest),
            "size_kb": round(dest.stat().st_size / 1024, 1),
        })

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


# ──────────────────────────────────────────────
# SKILL 8 — propagate_renames
# ──────────────────────────────────────────────

@tool
def propagate_renames() -> str:
    """
    Propage les renames détectés dans tous les fichiers de sortie déjà écrits.
    Assure la cohérence globale : si une fonction a été renommée dans un fichier,
    toutes ses références dans les autres fichiers sont mises à jour.
    Utilise ce skill EN DERNIER, après avoir traité tous les fichiers.
    """
    output_dir = _session.get("output_dir")
    ctx = _session.get("shared_context")

    if not output_dir or not ctx:
        return json.dumps({"status": "error", "error": "Session non initialisée."})

    if not ctx.global_renames:
        return json.dumps({"status": "ok", "message": "Aucun rename à propager.", "files_modified": 0})

    modified_count = 0
    renames_applied: dict[str, list[str]] = {}

    for file_path in output_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix in (".json", ".md", ".zip"):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            modified = content
            applied_here = []
            for old_name, change in ctx.global_renames.items():
                pattern = rf"\b{re.escape(old_name)}\b"
                new_content = re.sub(pattern, change.new_name, modified)
                if new_content != modified:
                    applied_here.append(f"{old_name} → {change.new_name}")
                    modified = new_content
            if modified != content:
                file_path.write_text(modified, encoding="utf-8")
                modified_count += 1
                renames_applied[str(file_path.name)] = applied_here
        except Exception as exc:
            logger.warning("Propagation échouée sur %s : %s", file_path, exc)

    return json.dumps({
        "status": "ok",
        "files_modified": modified_count,
        "renames_applied": renames_applied,
        "total_renames": len(ctx.global_renames),
    }, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# SKILL 9 — create_zip
# ──────────────────────────────────────────────

@tool
def create_zip() -> str:
    """
    Crée le ZIP final contenant le repository refactorisé.
    Inclut également le rapport de contexte et les métadonnées.
    Utilise ce skill EN TOUT DERNIER après propagate_renames.
    """
    output_dir = _session.get("output_dir")
    ctx = _session.get("shared_context")

    if not output_dir:
        return json.dumps({"status": "error", "error": "Session non initialisée."})

    try:
        # Générer le rapport markdown
        report_md = _build_report(ctx)
        (output_dir / "refactoring_report.md").write_text(report_md, encoding="utf-8")

        # Créer le ZIP
        zip_path = output_dir.parent / "refactored_output.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in output_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(output_dir.parent))

        return json.dumps({
            "status": "ok",
            "zip_path": str(zip_path),
            "zip_size_kb": round(zip_path.stat().st_size / 1024, 1),
            "files_in_zip": sum(1 for _ in output_dir.rglob("*") if _.is_file()),
        })

    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})


def _build_report(ctx: SharedContext | None) -> str:
    if not ctx:
        return "# Rapport de refactorisation\n\nAucun contexte disponible."
    lines = [
        "# Rapport de refactorisation — Agent ReAct",
        f"- Mode : {ctx.mode}",
        f"- Fichiers traités : {len(ctx.file_contexts)}",
        f"- Renames globaux : {len(ctx.global_renames)}",
        "",
        "## Renames détectés",
    ]
    for old, change in ctx.global_renames.items():
        lines.append(f"- `{old}` → `{change.new_name}` ({change.kind}, dans `{change.source_file}`)")
    lines += ["", "## Conventions adoptées"]
    for conv in ctx.global_conventions:
        lines.append(f"- {conv}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Liste de tous les skills pour l'agent
# ──────────────────────────────────────────────

ALL_SKILLS = [
    scan_repo,
    analyze_deps,
    get_processing_order,
    read_file,
    get_shared_context,
    refactor_file,
    write_file,
    propagate_renames,
    create_zip,
]
