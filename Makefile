################################################################################
# Makefile Python - Windows
#
# Objectif : gérer simplement un environnement virtuel Python sous Windows.
#
# Exemples :
#   make help
#   make setup PY=3.11
#   make run PY=3.11
#   make run PY=3.11 ARGS="--debug --port 8080"
#   make shell PY=3.11
#   make clean PY=3.11
#
# La dernière version Python choisie est mémorisée dans .make-python-version.
# Si aucune version n'a encore été choisie : Python 3.11 par défaut.
################################################################################

.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# Configuration / mémorisation de la version Python
# -----------------------------------------------------------------------------
PY_CONFIG := .make-python-version

# Ce petit fichier est généré automatiquement sous la forme : PY := 3.10
# Une variable passée en ligne de commande (ex. PY=3.12) reste prioritaire.
-include $(PY_CONFIG)

PY              ?= 3.11
PYTHON_LAUNCHER ?= py
VENV_DIR        ?= py$(PY)
SRC             ?= src/main.py
REQ             ?= requirements.txt
LOCK_FILE       ?= requirements.lock.txt
ARGS            ?=

# Utilise des / : Python et Windows les acceptent, et GNU Make les gère mieux.
VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe

# Les recettes ci-dessous utilisent volontairement la syntaxe cmd.exe.
ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
endif

.PHONY: help setup check-python venv install reinstall run shell on off \
        upgrade freeze info list-python clean clear rebuild select-python reset-python

# -----------------------------------------------------------------------------
# Aide
# -----------------------------------------------------------------------------
help:
	@echo.
	@echo ================================================================
	@echo   Makefile Python - Windows
	@echo ================================================================
	@echo.
	@echo Commandes principales :
	@echo   make setup PY=3.11       Cree le venv et installe les dependances
	@echo   make run PY=3.11         Lance $(SRC)
	@echo   make shell PY=3.11       Ouvre PowerShell avec le venv active
	@echo   make clean              Supprime le venv de la version memorisee
	@echo   make clear              Alias de make clean
	@echo   make rebuild            Recree completement le venv
	@echo.
	@echo Outils :
	@echo   make install             Installe requirements.txt
	@echo   make reinstall           Reinstalle les dependances
	@echo   make upgrade             Met a jour pip, setuptools et wheel
	@echo   make freeze              Genere requirements.lock.txt
	@echo   make info                Affiche la configuration courante
	@echo   make list-python         Liste les versions Python disponibles
	@echo   make select-python PY=3.10  Memorise Python 3.10 sans creer de venv
	@echo   make reset-python        Oublie la version memorisee (retour a 3.11)
	@echo.
	@echo Parametres :
	@echo   PY=3.11                  Version Python a utiliser et memoriser via venv/setup
	@echo   SRC=src/main.py          Script principal
	@echo   REQ=requirements.txt     Fichier de dependances
	@echo   ARGS="..."               Arguments transmis au programme
	@echo   VENV_DIR=py3.11          Dossier du venv
	@echo.

# -----------------------------------------------------------------------------
# Verification de Python
# -----------------------------------------------------------------------------
check-python:
	@$(PYTHON_LAUNCHER) -$(PY) --version >nul 2>&1 || (echo [ERREUR] Python $(PY) est introuvable via le launcher "$(PYTHON_LAUNCHER)". & echo Utilisez "make list-python" pour voir les versions installees. & exit /b 1)

list-python:
	@echo Versions Python detectees :
	@$(PYTHON_LAUNCHER) -0p 2>nul || $(PYTHON_LAUNCHER) list

# -----------------------------------------------------------------------------
# Creation du venv
# -----------------------------------------------------------------------------
venv: check-python
	@if exist "$(VENV_PYTHON)" (echo [OK] Environnement deja present : $(VENV_DIR)) else (echo [INFO] Creation de $(VENV_DIR) avec Python $(PY)... && $(PYTHON_LAUNCHER) -$(PY) -m venv "$(VENV_DIR)" && "$(VENV_PYTHON)" -m pip install --upgrade pip setuptools wheel && echo [OK] Environnement cree : $(VENV_DIR))
	@echo PY := $(PY)>"$(PY_CONFIG)"
	@echo [OK] Python $(PY) memorise pour les prochaines commandes.

# Change la version memorisee sans creer d'environnement virtuel.
select-python: check-python
	@echo PY := $(PY)>"$(PY_CONFIG)"
	@echo [OK] Python $(PY) est maintenant la version par defaut de ce projet.

# Supprime uniquement le choix local. Le prochain appel utilisera Python 3.11.
reset-python:
	@if exist "$(PY_CONFIG)" (del /Q "$(PY_CONFIG)" >nul && echo [OK] Version Python memorisee oubliee. Python 3.11 sera utilise par defaut.) else (echo [INFO] Aucune version Python n'etait memorisee.)

# Setup complet pour un nouveau clone du projet.
setup: install
	@echo [OK] Projet pret a etre utilise.
	@echo Lancez : make run PY=$(PY)

# -----------------------------------------------------------------------------
# Dependances
# -----------------------------------------------------------------------------
install: venv
	@if exist "$(REQ)" (echo [INFO] Installation des dependances depuis $(REQ)... && "$(VENV_PYTHON)" -m pip install -r "$(REQ)" && echo [OK] Dependances installees.) else (echo [INFO] Aucun fichier $(REQ) trouve. Rien a installer.)

reinstall: venv
	@if exist "$(REQ)" (echo [INFO] Reinstallation des dependances... && "$(VENV_PYTHON)" -m pip install --upgrade --force-reinstall -r "$(REQ)" && echo [OK] Dependances reinstallees.) else (echo [ERREUR] Fichier $(REQ) introuvable. & exit /b 1)

upgrade: venv
	@echo [INFO] Mise a jour des outils Python...
	@"$(VENV_PYTHON)" -m pip install --upgrade pip setuptools wheel
	@echo [OK] Outils Python mis a jour.

freeze: venv
	@echo [INFO] Generation de $(LOCK_FILE)...
	@"$(VENV_PYTHON)" -m pip freeze > "$(LOCK_FILE)"
	@echo [OK] Fichier genere : $(LOCK_FILE)

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
# run depend de install : un premier "make run" suffit donc a preparer le projet.
run: install
	@if not exist "$(SRC)" (echo [ERREUR] Script principal introuvable : $(SRC) & exit /b 1)
	@echo [INFO] Execution : $(SRC) $(ARGS)
	@"$(VENV_PYTHON)" "$(SRC)" $(ARGS)

# Ouvre un nouveau PowerShell avec le venv active.
shell: venv
	@echo [INFO] Ouverture d'un PowerShell avec $(VENV_DIR) active...
	@powershell.exe -NoLogo -NoExit -ExecutionPolicy Bypass -Command "& '.\$(VENV_DIR)\Scripts\Activate.ps1'"

# Alias conserves pour compatibilite avec l'ancien Makefile.
on: shell

off:
	@echo [INFO] Make ne peut pas desactiver l'environnement du terminal parent.
	@echo Si vous avez utilise "make shell", tapez simplement "exit" dans ce PowerShell.
	@echo Si vous l'avez active manuellement, utilisez la commande "deactivate".

# -----------------------------------------------------------------------------
# Diagnostic
# -----------------------------------------------------------------------------
info:
	@echo.
	@echo Configuration courante :
	@echo   Python utilise : $(PY)
	@if exist "$(PY_CONFIG)" (echo   Memoire projet : $(PY_CONFIG)) else (echo   Memoire projet : aucune ^(defaut 3.11^))
	@echo   Launcher       : $(PYTHON_LAUNCHER)
	@echo   Venv           : $(VENV_DIR)
	@echo   Python du venv : $(VENV_PYTHON)
	@echo   Script         : $(SRC)
	@echo   Requirements   : $(REQ)
	@echo   Arguments      : $(ARGS)
	@echo.
	@if exist "$(VENV_PYTHON)" ("$(VENV_PYTHON)" --version) else (echo Etat du venv   : absent)

# -----------------------------------------------------------------------------
# Nettoyage
# -----------------------------------------------------------------------------
# Ne supprime que le venv explicitement selectionne, jamais tous les dossiers py3*.
clean:
	@if exist "$(VENV_DIR)" (echo [INFO] Suppression de $(VENV_DIR)... && powershell.exe -NoProfile -Command "Remove-Item -LiteralPath '$(VENV_DIR)' -Recurse -Force" && echo [OK] Environnement supprime.) else (echo [INFO] Aucun environnement $(VENV_DIR) a supprimer.)

# Alias pratique pour ceux qui tapent naturellement "clear".
clear: clean

rebuild: clean setup
	@echo [OK] Environnement reconstruit avec Python $(PY).
