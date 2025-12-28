################################################################################
# Makefile Python multi-environnement (Windows)
#
# Ce Makefile permet de :
#   1. Créer un environnement virtuel Python spécifique (3.10, 3.11, 3.12)
#   2. Installer automatiquement les dépendances depuis requirements.txt
#   3. Lancer le script principal Python
#   4. Supprimer complètement l'environnement virtuel
#
# Commandes principales :
#   make venv PY=3.11     -> créer un venv pour Python 3.11
#   make install          -> installer les dépendances dans le venv existant
#   make run              -> lancer src/main.py avec le venv
#   make clean            -> supprimer le venv existant
#   make on               -> ouvrir un nouveau PowerShell avec le venv activé
#   make off              -> ouvrir un PowerShell pour désactiver le venv
#
# Notes :
# - Seul un venv est actif à la fois.
# - Le dossier du venv est nommé py<version>, ex : py3.11
# - L'activation depuis Makefile ne modifie pas le shell courant.
################################################################################

# -----------------------------
# Configuration
# -----------------------------

# Version de Python utiliser par défaut (modifiable en ligne de commande)
PY ?= 3.10
# Nom du dossier de l'environnement virtuel qui sera créer (py3.10, py3.11, etc.)
VENV_NAME = py$(PY)
# Lanceur Python Windows (py)
PYTHON_LAUNCHER = py
# Script Python principal à exécuter
SRC = src/main.py

# Détection automatique du venv existant dans le dossier courant
# $(wildcard py3*) cherche tous les dossiers qui commencent par "py3"
VENV = $(firstword $(wildcard py3*))

# -----------------------------
# Création de l'environnement
# -----------------------------
venv:
# Vérifie si un venv existe déjà (commande propre à windows)
#	@if exist py3* ( \
#		echo "Un environnement existe deja : $(VENV)" && exit 1 \
#	)
# Crée le venv avec la version Python choisie
	$(PYTHON_LAUNCHER) -$(PY) -m venv $(VENV_NAME)
# Met à jour pip dans le nouvel environnement
	$(VENV)/Scripts/python -m pip install --upgrade pip
# Affiche un message de confirmation
#	@echo "Environnement cree : $(VENV_NAME)"

# -----------------------------
# Activation / Désactivation du venv
# -----------------------------
# Ces commandes ouvrent un nouveau PowerShell (enfant) avec le venv activé ou désactivé (ce qui l'active dans le parent).
on:
	powershell -NoExit -Command "& '.\$(VENV)\Scripts\Activate.ps1'"
off:
	powershell -NoExit -Command "& '.\$(VENV)\Scripts\deactivate.bat'"

# -----------------------------
# Installation des dépendances
# -----------------------------
install:
# Vérifie qu'un venv existe avant d'installer
	@if not exist py3* ( \
		echo "Aucun environnement virtuel trouve" && exit 1 \
	)
# Installe toutes les librairies listées dans requirements.txt
	$(VENV)/Scripts/python -m pip install -r requirements.txt

# -----------------------------
# Lancer le programme
# -----------------------------
run:
# Vérifie qu'un venv existe avant d'exécuter le script
	@if not exist py3* ( \
		echo "Aucun environnement virtuel trouve" && exit 1 \
	)
# Exécute le script Python avec le Python du venv
# Cela revient au même qu'écrire python src/main.py
	$(VENV)/Scripts/python $(SRC)

# -----------------------------
# Nettoyage / suppression de l'environnement virtuel
# -----------------------------
clean:
# Supprime tous les dossiers qui commencent par py3 (venvs)
# -Recurse : supprime tout le contenu
# -Force : supprime sans demander de confirmation
	powershell -Command "Get-ChildItem -Directory py3* | Remove-Item -Recurse -Force"