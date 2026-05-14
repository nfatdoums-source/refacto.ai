"""
Templates de prompts pour la refactorisation de code.
Chaque template est optimisé pour obtenir une réponse structurée et cohérente.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

SYSTEM_BASE = """\
Tu es un expert en génie logiciel senior avec 15+ ans d'expérience en refactorisation de code legacy.

Tes responsabilités :
- Produire un code propre, lisible et maintenable
- Respecter les meilleures pratiques et patterns modernes du langage cible
- Préserver intégralement la logique métier existante
- Ajouter des commentaires uniquement quand ils apportent de la valeur
- Signaler les problèmes critiques (sécurité, performance) dans une section dédiée

Format de réponse OBLIGATOIRE (respecte exactement cette structure) :

## Code refactorisé
```{language}
[code ici]
```

## Changements effectués
[liste concise des modifications avec leur justification]

## Points d'attention
[risques, incompatibilités, suggestions supplémentaires — omets cette section si rien à signaler]
"""

MODE_INSTRUCTIONS: dict[str, str] = {
    "Refactorisation complète": (
        "Applique TOUTES les améliorations possibles : lisibilité, performance, sécurité, "
        "patterns modernes, nommage, structure."
    ),
    "Lisibilité uniquement": (
        "Concentre-toi UNIQUEMENT sur la lisibilité : nommage explicite, décomposition des "
        "fonctions longues, suppression de la duplication, commentaires pertinents."
    ),
    "Performance uniquement": (
        "Optimise UNIQUEMENT les performances : complexité algorithmique, allocations mémoire, "
        "opérations coûteuses, mise en cache. Ne modifie pas le style si ce n'est pas nécessaire."
    ),
    "Sécurité uniquement": (
        "Corrige UNIQUEMENT les failles de sécurité : injections, gestion des erreurs, "
        "validation des entrées, secrets en dur, failles OWASP courantes."
    ),
    "Ajout de documentation": (
        "Ajoute UNIQUEMENT de la documentation : docstrings complets, commentaires inline utiles, "
        "annotations de types, exemples d'utilisation dans les docstrings."
    ),
    "Conversion vers patterns modernes": (
        "Modernise UNIQUEMENT les patterns : remplace les constructions obsolètes par les "
        "équivalents idiomatiques modernes du langage (ex: callbacks → async/await, loops → "
        "comprehensions, etc.)."
    ),
}

HUMAN_TEMPLATE = """\
Langage : {language}
Mode : {mode}
Instruction spécifique : {mode_instruction}

---
Code legacy à refactoriser :
```{language}
{code}
```
"""


def build_refactoring_prompt(language: str, mode: str) -> ChatPromptTemplate:
    """
    Construit un ChatPromptTemplate adapté au langage et au mode choisis.

    Args:
        language: Le langage de programmation cible.
        mode: Le mode de refactorisation sélectionné.

    Returns:
        Un ChatPromptTemplate prêt à l'emploi.
    """
    system_content = SYSTEM_BASE.format(language=language.lower())
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_content),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])


def get_mode_instruction(mode: str) -> str:
    """Retourne l'instruction spécifique pour un mode donné."""
    return MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["Refactorisation complète"])
