"""
Interface Gradio — Code Refactorer LLM.

Deux onglets :
  1. Fichier unique  — upload / coller un fichier
  2. Repository      — refactorisation simple (batch) OU agent multi-fichiers
"""
from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator
from pathlib import Path

import gradio as gr

from config import app_cfg, refactoring_cfg
from refactorer import CodeRefactorer, RefactorerError
from batch_refactorer import BatchRefactorer
from agent_refactorer import MultiFileAgent
from react_agent import ReActRefactoringAgent

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Singletons
# ──────────────────────────────────────────────
refactorer = CodeRefactorer()
batch_refactorer = BatchRefactorer()
agent = MultiFileAgent()
react_agent = ReActRefactoringAgent()

# ──────────────────────────────────────────────
# Thème
# ──────────────────────────────────────────────
GRADIO_THEME = gr.Theme.from_hub("NeoPy/soft")

# ──────────────────────────────────────────────
# Extension → Langage
# ──────────────────────────────────────────────
EXT_TO_LANG: dict[str, str] = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "JavaScript", ".tsx": "TypeScript",
    ".java": "Java", ".cs": "C#", ".cpp": "C++", ".cc": "C++",
    ".go": "Go", ".rs": "Rust", ".php": "PHP", ".rb": "Ruby",
}

MAX_FILE_BYTES = app_cfg.max_file_size_mb * 1024 * 1024


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def detect_language_from_path(path: str | None) -> str | None:
    if not path:
        return None
    return EXT_TO_LANG.get(Path(path).suffix.lower())


def extract_code_block(text: str, language: str) -> str:
    pattern = rf"```{re.escape(language.lower())}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    fallback = re.search(r"```\w*\n(.*?)```", text, re.DOTALL)
    return fallback.group(1).strip() if fallback else text


def load_file(file_obj) -> tuple[str, str]:
    if file_obj is None:
        return "", ""
    try:
        size = os.path.getsize(file_obj.name)
        if size > MAX_FILE_BYTES:
            raise ValueError(
                f"Fichier trop volumineux ({size / 1024:.0f} Ko). "
                f"Limite : {app_cfg.max_file_size_mb} Mo."
            )
        content = Path(file_obj.name).read_text(encoding="utf-8", errors="replace")
        detected = detect_language_from_path(file_obj.name) or ""
        return content, detected
    except ValueError as exc:
        return f"❌ {exc}", ""
    except Exception as exc:
        logger.exception("Erreur lecture fichier")
        return f"❌ Erreur lecture : {exc}", ""


def check_health() -> str:
    result = refactorer.health_check()
    if result["status"] == "ok":
        return (
            f"✅ **LM Studio connecté**\n\n"
            f"- **URL** : `{result['base_url']}`\n"
            f"- **Modèle chargé** : `{result['model']}`\n"
            f"- **Modèles disponibles** : {result['model_count']}"
        )
    return (
        f"❌ **LM Studio non disponible**\n\n"
        f"- **URL** : `{result['base_url']}`\n"
        f"- **Erreur** : {result['error']}\n\n"
        f"**Solution** : Lancez LM Studio → chargez un modèle → activez le serveur local."
    )


# ──────────────────────────────────────────────
# Callbacks — Onglet 1 : Fichier unique
# ──────────────────────────────────────────────

def on_file_upload(file_obj) -> tuple[str, str]:
    code, detected = load_file(file_obj)
    return code, detected if detected else gr.update()


def refactor_code(
    code: str,
    language: str,
    mode: str,
    show_full: bool,
) -> Generator[tuple[str, str], None, None]:
    if not code or not code.strip():
        yield "⚠️ Veuillez entrer ou uploader du code source.", ""
        return
    full_response = ""
    yield "⏳ Connexion au modèle LM Studio...", ""
    try:
        for chunk in refactorer.refactor_stream(code, language, mode):
            full_response += chunk
            extracted = extract_code_block(full_response, language)
            yield (full_response if show_full else extracted), extracted
    except RefactorerError as exc:
        yield f"❌ **Erreur** : {exc}", ""
    except Exception as exc:
        logger.exception("Erreur inattendue")
        yield f"❌ **Erreur inattendue** : {exc}", ""


def clear_single() -> tuple:
    return "", "", "", ""


# ──────────────────────────────────────────────
# Callbacks — Onglet 2 : Repository
# ──────────────────────────────────────────────

def scan_repo(repo_path: str) -> str:
    if not repo_path or not repo_path.strip():
        return "⚠️ Veuillez entrer un chemin de dossier."
    from repo_scanner import RepoScanner
    try:
        result = RepoScanner().scan(repo_path.strip())
        langs = "\n".join(
            f"  - {l} : {n} fichier(s)"
            for l, n in sorted(result.languages.items())
        )
        skipped = ""
        if result.skipped_paths:
            skipped = "\n\n**Ignorés :**\n" + "\n".join(
                f"  - `{p}` ({r})"
                for p, r in list(zip(result.skipped_paths, result.skipped_reasons))[:8]
            )
        return (
            f"✅ **{result.total_files} fichier(s)** ({result.total_size_kb:.1f} Ko)\n\n"
            f"**Langages :**\n{langs or '  Aucun'}{skipped}"
        )
    except ValueError as exc:
        return f"❌ {exc}"
    except Exception as exc:
        return f"❌ Erreur : {exc}"


def run_repo(
    repo_path: str,
    mode: str,
    use_agent: bool,
) -> Generator[tuple[str, str, int, object], None, None]:
    """
    Générateur principal pour la refactorisation de repository.

    Args:
        use_agent: True = agent multi-fichiers, False = batch simple.
    """
    if not repo_path or not repo_path.strip():
        yield "⚠️ Chemin manquant.", "⚠️ Chemin manquant.", 0, gr.update(visible=False)
        return

    runner = agent if use_agent else batch_refactorer
    label = "🤖 Agent multi-fichiers" if use_agent else "📦 Batch simple"
    log_lines: list[str] = []

    yield f"🚀 Démarrage ({label})...", f"Démarrage...", 0, gr.update(visible=False)

    try:
        for event in runner.run(repo_path.strip(), mode):
            if event.log_line:
                log_lines.append(event.log_line)

            log_text = "\n".join(log_lines[-60:])

            if event.status == "done":
                zip_path = Path(repo_path.strip()).resolve().parent / "refactored_output.zip"
                if zip_path.exists():
                    yield log_text, event.message, 100, gr.update(value=str(zip_path), visible=True)
                else:
                    yield log_text, event.message, 100, gr.update(visible=False)
            elif event.status == "error" and event.current == 0:
                yield log_text, f"❌ {event.message}", 0, gr.update(visible=False)
                return
            else:
                status = f"[{event.current}/{event.total}] {event.message}"
                yield log_text, status, event.percent, gr.update(visible=False)

    except Exception as exc:
        logger.exception("Erreur run_repo")
        yield str(exc), f"❌ {exc}", 0, gr.update(visible=False)


def clear_repo() -> tuple:
    return "", "*Résultat du scan...*", "", 0, gr.update(visible=False)


# ──────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────

def build_interface() -> gr.Blocks:

    with gr.Blocks(
        title=app_cfg.title,
        theme=GRADIO_THEME,
        css="""
            .mono { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
            .log-box textarea {
                font-family: monospace !important;
                font-size: 12px !important;
                line-height: 1.6 !important;
            }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(f"# {app_cfg.title}")
        gr.Markdown(app_cfg.description)

        with gr.Accordion("🔌 Connexion LM Studio", open=False):
            health_output = gr.Markdown("*Cliquez pour vérifier.*")
            health_btn = gr.Button("Vérifier la connexion", size="sm")
            health_btn.click(fn=check_health, outputs=[health_output])

        # ══════════════════════
        # Onglet 1 — Fichier
        # ══════════════════════
        with gr.Tab("📄 Fichier unique"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Code source")
                    file_input = gr.File(
                        label="Uploader un fichier",
                        file_types=list(EXT_TO_LANG.keys()),
                        type="filepath",
                    )
                    code_input = gr.Code(
                        label="Code legacy",
                        language="python",
                        lines=20,
                        elem_classes=["mono"],
                        interactive=True,
                    )
                    with gr.Row():
                        language_select = gr.Dropdown(
                            label="Langage",
                            choices=list(refactoring_cfg.supported_languages),
                            value="Python",
                            scale=1,
                        )
                        mode_select = gr.Dropdown(
                            label="Mode",
                            choices=list(refactoring_cfg.refactoring_modes),
                            value="Refactorisation complète",
                            scale=2,
                        )
                    show_full = gr.Checkbox(label="Réponse complète", value=True)
                    with gr.Row():
                        refactor_btn = gr.Button("🚀 Refactoriser", variant="primary", scale=3)
                        clear_btn = gr.Button("🗑️ Effacer", scale=1)

                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Résultat")
                    output_display = gr.Markdown(value="*La réponse apparaîtra ici...*")
                    code_output = gr.Code(
                        label="Code refactorisé",
                        language="python",
                        lines=20,
                        elem_classes=["mono"],
                        interactive=False,
                    )

            with gr.Accordion("💡 Exemples", open=False):
                gr.Examples(
                    examples=[
                        ["def calc(l):\n    r = 0\n    for i in range(len(l)):\n        if l[i] > 0:\n            r = r + l[i]\n    return r",
                         "Python", "Refactorisation complète"],
                        ["function getUser(id, cb) {\n  var x = new XMLHttpRequest();\n  x.open('GET','/api/'+id,true);\n  x.onreadystatechange=function(){if(x.readyState==4)cb(null,JSON.parse(x.responseText));};\n  x.send();\n}",
                         "JavaScript", "Conversion vers patterns modernes"],
                    ],
                    inputs=[code_input, language_select, mode_select],
                )

            file_input.change(fn=on_file_upload, inputs=[file_input], outputs=[code_input, language_select])
            language_select.change(
                fn=lambda l: gr.update(language=l.lower().replace("c#", "csharp").replace("c++", "cpp")),
                inputs=[language_select], outputs=[code_input],
            )
            refactor_btn.click(
                fn=refactor_code,
                inputs=[code_input, language_select, mode_select, show_full],
                outputs=[output_display, code_output],
            )
            clear_btn.click(fn=clear_single, outputs=[code_input, output_display, code_output, file_input])

        # ══════════════════════════════════
        # Onglet 2 — Repository
        # ══════════════════════════════════
        with gr.Tab("📁 Repository complet"):

            with gr.Row():
                # ── Colonne config ──
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")

                    repo_path_input = gr.Textbox(
                        label="Chemin du repository (absolu)",
                        placeholder="C:\\Users\\Fatoumata\\MonProjet",
                        lines=1,
                    )
                    repo_mode_select = gr.Dropdown(
                        label="Mode de refactorisation",
                        choices=list(refactoring_cfg.refactoring_modes),
                        value="Refactorisation complète",
                    )

                    # ── Sélecteur de moteur ──
                    use_agent_radio = gr.Radio(
                        label="🔧 Moteur de refactorisation",
                        choices=[
                            "🤖 Agent ReAct AI (recommandé)",
                            "🧩 Pipeline multi-fichiers",
                            "📦 Batch simple (fichiers indépendants)",
                        ],
                        value="🤖 Agent ReAct AI (recommandé)",
                    )

                    agent_info = gr.Markdown(
                        value=(
                            "**Agent multi-fichiers** :\n"
                            "- Analyse les imports et dépendances entre fichiers\n"
                            "- Trie les fichiers dans l'ordre optimal\n"
                            "- Partage le contexte (renames, signatures, types)\n"
                            "- Propage les changements dans tout le projet\n\n"
                            "*Plus lent mais résultat cohérent sur tout le repo.*"
                        )
                    )

                    with gr.Row():
                        scan_btn = gr.Button("🔍 Scanner", variant="secondary", scale=1)
                        repo_run_btn = gr.Button("🚀 Lancer", variant="primary", scale=2)
                        repo_clear_btn = gr.Button("🗑️", scale=1)

                    scan_result = gr.Markdown(
                        value="*Scannez d'abord le repository.*"
                    )
                    zip_download = gr.File(
                        label="📦 ZIP refactorisé",
                        visible=False,
                        interactive=False,
                    )

                # ── Colonne progression ──
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Progression")
                    repo_progress = gr.Slider(
                        label="Progression",
                        minimum=0, maximum=100, value=0,
                        interactive=False,
                    )
                    repo_status = gr.Markdown(value="*En attente...*")
                    repo_log = gr.Textbox(
                        label="Journal de traitement",
                        lines=24,
                        interactive=False,
                        elem_classes=["log-box"],
                        placeholder="Les logs apparaîtront ici...",
                    )

            # ── Mise à jour de la description selon le moteur choisi ──
            def update_agent_info(choice: str) -> str:
                if "ReAct" in choice:
                    return (
                        "**🤖 Agent ReAct AI** — vrai agent IA :\n"
                        "- Le LLM raisonne (Thought → Action → Observation)\n"
                        "- Il choisit lui-même l'ordre et les skills à appeler\n"
                        "- Il s'adapte aux erreurs et situations inattendues\n"
                        "- 9 skills : scan, analyze, read, refactor, write, propagate...\n\n"
                        "*Le plus intelligent — recommandé pour les gros projets.*"
                    )
                if "Pipeline" in choice:
                    return (
                        "**🧩 Pipeline multi-fichiers** :\n"
                        "- Analyse les imports et dépendances entre fichiers\n"
                        "- Trie les fichiers dans l'ordre optimal (tri topologique)\n"
                        "- Partage le contexte (renames, signatures, types)\n"
                        "- Déterministe et prévisible\n\n"
                        "*Bon équilibre vitesse / cohérence.*"
                    )
                return (
                    "**📦 Batch simple** :\n"
                    "- Chaque fichier est traité indépendamment\n"
                    "- Rapide mais sans cohérence entre fichiers\n\n"
                    "*Pour tester rapidement ou petits projets.*"
                )

            use_agent_radio.change(
                fn=update_agent_info,
                inputs=[use_agent_radio],
                outputs=[agent_info],
            )

            # ── Événements ──
            scan_btn.click(fn=scan_repo, inputs=[repo_path_input], outputs=[scan_result])

            def run_with_agent_flag(repo_path, mode, agent_choice):
                if "ReAct" in agent_choice:
                    from react_agent import ReActRefactoringAgent
                    from pathlib import Path as _Path
                    ra = ReActRefactoringAgent()
                    log_lines = []
                    for event in ra.run(repo_path.strip(), mode):
                        log_lines.append(event.content)
                        log_text = "\n".join(log_lines[-60:])
                        if event.type == "done":
                            zip_path = _Path(repo_path.strip()).resolve().parent / "refactored_output.zip"
                            if zip_path.exists():
                                yield log_text, event.content, 100, gr.update(value=str(zip_path), visible=True)
                            else:
                                yield log_text, event.content, 100, gr.update(visible=False)
                        elif event.type == "error":
                            yield log_text, f"❌ {event.content}", 0, gr.update(visible=False)
                            return
                        else:
                            yield log_text, event.content, event.progress, gr.update(visible=False)
                else:
                    use_pipeline = "Pipeline" in agent_choice
                    yield from run_repo(repo_path, mode, use_pipeline)

            repo_run_btn.click(
                fn=run_with_agent_flag,
                inputs=[repo_path_input, repo_mode_select, use_agent_radio],
                outputs=[repo_log, repo_status, repo_progress, zip_download],
            )
            repo_clear_btn.click(
                fn=clear_repo,
                outputs=[repo_path_input, scan_result, repo_log, repo_progress, zip_download],
            )

    return demo


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Démarrage — %s", app_cfg.title)
    demo = build_interface()
    demo.launch(
        server_name=app_cfg.host,
        server_port=app_cfg.port,
        share=app_cfg.share,
        show_error=True,
    )
