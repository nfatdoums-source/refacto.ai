# 🔧 Refacto.ai

Refactorisez votre code legacy avec un LLM local — **LM Studio + LangChain + Gradio**.
Aucune donnée ne quitte votre machine.

---

## ⚡ Stack technique

| Composant | Rôle |
|---|---|
| **LM Studio** | Serveur LLM local (API OpenAI-compatible) |
| **LangChain** | Orchestration : prompts, chains, streaming |
| **Gradio 5** | Interface utilisateur web |
| **Python 3.11+** | Runtime |

---

## 🚀 Étapes d'installation

### Étape 1 — Installer LM Studio

1. Téléchargez LM Studio depuis [lmstudio.ai](https://lmstudio.ai)
2. Installez et lancez LM Studio
3. Dans l'onglet **Discover**, recherchez et téléchargez un modèle de code :
   - `Qwen2.5-Coder-7B-Instruct` ✅ recommandé (bon équilibre vitesse/qualité)
   - `DeepSeek-Coder-V2-Lite-Instruct` ✅ excellent pour le code
   - `CodeLlama-7B-Instruct` ✅ léger, rapide
4. Chargez le modèle dans l'onglet **Chat**
5. Allez dans **Local Server** (icône `</>`) → cliquez **Start Server**
6. Vérifiez que le serveur tourne sur `http://localhost:1234`

### Étape 2 — Cloner / créer le projet

```bash
# Créez le dossier du projet
mkdir code-refactorer && cd code-refactorer

# Copiez les fichiers du projet ici
```

### Étape 3 — Créer l'environnement virtuel Python

```bash
# Python 3.11 ou 3.12 recommandé
python -m venv .venv

# Activer (Windows)
.venv\Scripts\activate

# Activer (macOS / Linux)
source .venv/bin/activate
```

### Étape 4 — Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Étape 5 — Configurer les variables d'environnement

```bash
# Copier le fichier exemple
cp .env.example .env

# Ouvrir .env dans Cursor et ajuster si nécessaire
# La configuration par défaut fonctionne sans modification
# si LM Studio tourne sur le port 1234.
```

### Étape 6 — Lancer l'application

```bash
python app.py
```

Ouvrez votre navigateur sur **http://localhost:7860**

---

## 🛠️ Configuration dans Cursor

1. Ouvrez le dossier `code-refactorer` dans Cursor (`File > Open Folder`)
2. Cursor détectera automatiquement l'environnement virtuel `.venv`
3. Sélectionnez l'interpréteur : `Ctrl+Shift+P` → `Python: Select Interpreter` → choisissez `.venv`

---

## 🎯 Utilisation

1. **Uploadez** un fichier source OU **collez** du code directement
2. Sélectionnez le **langage** de programmation
3. Choisissez le **mode** de refactorisation :
   - `Refactorisation complète` — tout améliorer
   - `Lisibilité uniquement` — nommage, structure
   - `Performance uniquement` — algorithmes, mémoire
   - `Sécurité uniquement` — failles, validation
   - `Ajout de documentation` — docstrings, types
   - `Conversion vers patterns modernes` — async/await, comprehensions...
4. Cliquez sur **🚀 Refactoriser**
5. La réponse s'affiche en **streaming** en temps réel

---

## 📁 Structure du projet

```
code-refactorer/
├── app.py            # Interface Gradio + orchestration
├── refactorer.py     # Moteur LangChain + LM Studio
├── prompts.py        # Templates de prompts par mode
├── config.py         # Configuration centralisée
├── requirements.txt  # Dépendances Python épinglées
├── .env.example      # Variables d'environnement exemple
├── .env              # Votre configuration locale (ne pas commiter)
└── README.md         # Ce fichier
```

---

## 🔌 Vérifier la connexion LM Studio

Dans l'interface, dépliez la section **🔌 Connexion LM Studio** et cliquez **Vérifier la connexion**.

Ou depuis le terminal :
```bash
curl http://localhost:1234/v1/models
```

---

## 🐛 Dépannage

| Problème | Solution |
|---|---|
| `Connection refused` | LM Studio n'est pas démarré ou le serveur local n'est pas activé |
| `Model not loaded` | Chargez un modèle dans LM Studio avant de lancer le serveur |
| Réponse très lente | Réduisez `LM_STUDIO_MAX_TOKENS` dans `.env` ou utilisez un modèle plus petit |
| Code coupé | Augmentez `LM_STUDIO_MAX_TOKENS` (ex : 16384) |
| Port 7860 occupé | Changez `APP_PORT` dans `.env` |

---

## 🔒 Confidentialité

Tout le traitement est **100% local**. Aucune donnée n'est envoyée sur internet.
LM Studio et le modèle tournent entièrement sur votre machine.
