#!/bin/bash

# Pobierz ścieżkę do folderu projektu (root) na podstawie lokalizacji skryptu
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)  # Wznosi się o jeden poziom w górę do roota (~/projects/tipster)

# Konfiguracja (ścieżki względne do roota)
TEMPLATE_FILE="$PROJECT_ROOT/cloudformation/lambda.yaml"
OUTPUT_FILE="$PROJECT_ROOT/packaged-lambda.yaml"
S3_BUCKET="tipster-artifacts"
STACK_NAME="tipster-lambda"
PROFILE="AdministratorAccess-284415450706"

# Sprawdź zmiany w katalogu lambda/ (z perspektywy roota)
cd "$PROJECT_ROOT"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  if git diff --quiet lambda/; then
    echo "Brak zmian w lambda/ – ale kontynuuję pakowanie dla pewności."
  else
    echo "Wykryto zmiany – pakuję i uploaduję."
  fi
else
  echo "Nie wykryto repozytorium Git – zakładam zmiany i kontynuuję."
fi

# Pakuj (zawsze z force dla pewności)
aws cloudformation package --template-file "$TEMPLATE_FILE" --output-template-file "$OUTPUT_FILE" --s3-bucket "$S3_BUCKET" --force-upload --profile "$PROFILE"

# Deploy
aws cloudformation deploy --template-file "$OUTPUT_FILE" --stack-name "$STACK_NAME" --capabilities CAPABILITY_NAMED_IAM --profile "$PROFILE"

# Sprawdź status (wyświetl tylko kluczowe info)
STATUS_OUTPUT=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --profile "$PROFILE" 2>&1)
if echo "$STATUS_OUTPUT" | grep -q "Stack with id .* does not exist"; then
  echo "Błąd: Stack '$STACK_NAME' nie istnieje."
elif echo "$STATUS_OUTPUT" | grep -q "ValidationError"; then
  echo "Błąd walidacji: $(echo "$STATUS_OUTPUT" | grep -o 'ValidationError.*')"
else
  STATUS=$(echo "$STATUS_OUTPUT" | grep -o '"StackStatus": "[^"]*' | cut -d'"' -f4)
  if [ -z "$STATUS" ]; then
    echo "Nieznany status stacka – sprawdź ręcznie."
  else
    echo "Status stacka: $STATUS"
  fi
fi
