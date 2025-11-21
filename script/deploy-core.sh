#!/bin/bash

# Pobierz ścieżkę do folderu projektu (root) na podstawie lokalizacji skryptu
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)  # Wznosi się o jeden poziom w górę do roota (~/projects/tipster)

# Konfiguracja (ścieżki względne do roota)
TEMPLATE_FILE="$PROJECT_ROOT/cloudformation/core.yaml"
STACK_NAME="tipster-core"
PROFILE="AdministratorAccess-284415450706"

# Opcjonalnie: Sprawdź zmiany w pliku core.yaml (z perspektywy roota)
cd "$PROJECT_ROOT"
if git diff --quiet cloudformation/core.yaml; then
  echo "Brak zmian w core.yaml – pomijam deploy, jeśli niepotrzebny."
else
  echo "Wykryto zmiany – deployuję."
fi

# Deploy (bez package, zakładając, że nie potrzeba uploadu artefaktów)
aws cloudformation deploy --template-file "$TEMPLATE_FILE" --stack-name "$STACK_NAME" --capabilities CAPABILITY_NAMED_IAM --profile "$PROFILE"

# Sprawdź status
aws cloudformation describe-stacks --stack-name "$STACK_NAME" --profile "$PROFILE"
