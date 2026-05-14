"""
Agent multi-fichiers — version Structured Chat (compatible Qwen3).

Utilise create_structured_chat_agent au lieu de create_react_agent
car Qwen3 produit du JSON multi-lignes que le parser ReAct ne gère pas.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import lm_studio_cfg
from agent_skills import ALL_SKILLS, init_session, _session

logger = logging.getLogger(__name__)
OUTPUT_DIR_NAME = "refactored_output"


SYSTEM_PROMPT = """Tu es un agent qui refactorise un repository de code.

Tu DOIS répondre avec un objet JSON dans un bloc de code markdown, comme ceci :

```json
{{
  "action": "nom_du_skill",
  "action_input": {{
    "param1": "valeur1",
    "param2": "valeur2"
  }}
}}
```

Skills disponibles :
{tools}

Noms valides : {tool_names}

Quand la tâche est terminée, réponds avec :

```json
{{
  "action": "Final Answer",
  "action_input": "résumé de ce qui a été fait"
}}
```

Ordre recommandé :
1. scan_repo
2. analyze_deps
3. get_processing_order
4. Pour chaque fichier : get_shared_context → refactor_file → write_file
5. propagate_renames
6. create_zip
7. Final Answer

IMPORTANT :
- UN SEUL bloc JSON par réponse
- N'invente pas de noms de skills
- Suis l'ordre étape par étape
"""

HUMAN_PROMPT = """{input}

{agent_scratchpad}

(Réponds avec un seul bloc ```json ... ``` valide.)"""


@dataclass
class AgentEvent:
    type: str
    content: str
    progress: int = 0


class ReActRefactoringAgent:
    """Agent structured chat — compatible Qwen3."""

    def __init__(self) -> None:
         self._llm = ChatOpenAI(
            base_url=lm_studio_cfg.base_url,
            api_key=lm_studio_cfg.api_key,
            model=lm_studio_cfg.model,
            temperature=0.0,
            max_tokens=800,
            streaming=False,
            request_timeout=600,
            default_headers={"Connection": "close"},
        )

    def run(
        self,
        repo_path: str,
        mode: str,
    ) -> Generator[AgentEvent, None, None]:
        root = Path(repo_path).resolve()
        output_dir = root.parent / OUTPUT_DIR_NAME

        if output_dir.exists():
            shutil.rmtree(output_dir)

        init_session(str(output_dir), mode)

        yield AgentEvent("thought", "🧠 Agent initialisé...", 0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])

        agent = create_structured_chat_agent(
            llm=self._llm,
            tools=ALL_SKILLS,
            prompt=prompt,
        )

        executor = AgentExecutor(
            agent=agent,
            tools=ALL_SKILLS,
            verbose=True,
            max_iterations=30,
            max_execution_time=1800,
            handle_parsing_errors=(
                "Format invalide. Réponds avec un seul bloc ```json ... ``` "
                "contenant 'action' et 'action_input'."
            ),
            return_intermediate_steps=True,
        )

        task = (
            f"Refactorise le repository '{root}' avec le mode '{mode}'. "
            f"Sortie : '{output_dir}'."
        )

        yield AgentEvent("thought", f"📋 Tâche : refactorisation de {root.name}", 2)

        try:
            result = executor.invoke({"input": task})
            steps = result.get("intermediate_steps", [])
            total = max(len(steps), 1)

            for i, (action, observation) in enumerate(steps):
                progress = min(int((i + 1) / total * 90), 90)
                yield AgentEvent(
                    "action",
                    f"🔧 [{i+1}/{total}] {action.tool}",
                    progress,
                )
                obs_short = str(observation)[:200]
                yield AgentEvent("observation", f"👁️ {obs_short}", progress)

            ctx = _session.get("shared_context")
            zip_path = root.parent / "refactored_output.zip"
            summary = (
                f"✅ **Agent terminé**\n\n"
                f"- Étapes : {len(steps)}\n"
                f"- Fichiers traités : {len(ctx.file_contexts) if ctx else '?'}\n"
                f"- Renames : {len(ctx.global_renames) if ctx else '?'}\n"
                f"- ZIP : {zip_path.exists()}"
            )
            yield AgentEvent("done", summary, 100)

        except Exception as exc:
            logger.exception("Erreur agent")
            yield AgentEvent("error", f"❌ {exc}", 0)