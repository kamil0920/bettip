# Automatyzacja zbierania danych meczowych

## Przegląd

Automatyczny system pobierania danych meczów piłkarskich przez GitHub Actions, który:
- Uruchamia się **co tydzień w poniedziałek o 3:00 UTC**
- Pobiera dane meczów z ostatnich 26 dni (domyślnie)
- Zapisuje dane w dedykowanym branch `data`
- Zachowuje pełną historię zmian w danych

## Konfiguracja początkowa

### 1. Dodaj klucz API jako GitHub Secret

1. Przejdź do: **Settings** → **Secrets and variables** → **Actions**
2. Kliknij **New repository secret**
3. Nazwa: `API_FOOTBALL_KEY`
4. Wartość: Twój klucz API z api-football.com
5. Kliknij **Add secret**

### 2. Utwórz branch 'data' (opcjonalne - zostanie utworzony automatycznie)

```bash
git checkout -b data
git push origin data
git checkout main  # lub dev_structure
```

## Jak używać

### Automatyczne uruchomienia (co tydzień)

Workflow uruchamia się automatycznie każdy **poniedziałek o 3:00 UTC** i:
- Pobiera dane z ostatnich **26 dni**
- Dla sezonu **2024**

### Ręczne uruchomienie

Możesz uruchomić workflow ręcznie z dowolnymi parametrami:

1. Przejdź do: **Actions** → **Collect Match Data**
2. Kliknij **Run workflow**
3. Dostosuj parametry:
   - **Liczba dni wstecz**: np. `10`, `26`, `30`
   - **Sezon**: np. `2024`, `2025`
4. Kliknij **Run workflow**

#### Przykłady użycia ręcznego:

**Scenariusz 1: Pierwszy raz - zbierz zaległe dane**
```
Dni wstecz: 26
Sezon: 2024
```

**Scenariusz 2: Regularne aktualizacje**
```
Dni wstecz: 10
Sezon: 2024
```

**Scenariusz 3: Nowy sezon**
```
Dni wstecz: 30
Sezon: 2025
```

## Dostęp do danych

### Klonowanie branch z danymi

```bash
# Klonuj tylko branch z danymi
git clone -b data --single-branch https://github.com/TWOJA-ORGANIZACJA/tipster.git tipster-data

# Lub dodaj do istniejącego repo
git fetch origin data
git checkout data
```

### Pobierz najnowsze dane

```bash
git checkout data
git pull origin data
```

### Struktura danych w branch 'data'

```
data/
└── 01-raw/
    └── premier_league/  # lub inna liga
        └── 2024/
            ├── fixtures.json
            ├── events/
            │   ├── fixture_12345_events.json
            │   └── ...
            ├── lineups/
            │   ├── fixture_12345_lineups.json
            │   └── ...
            └── players/
                ├── fixture_12345_players.json
                └── ...
```

## Monitorowanie

### Sprawdź status ostatniego uruchomienia

1. Przejdź do zakładki **Actions**
2. Zobacz najnowsze uruchomienie **Collect Match Data**
3. Sprawdź:
   - ✅ **Success** - dane pobrane i zapisane
   - ℹ️ **Success** (brak zmian) - wszystko aktualne
   - ❌ **Failed** - sprawdź logi

### Logi i artefakty

Po każdym uruchomieniu dostępne są:
- **Podsumowanie** w zakładce Summary
- **Logi** - szczegółowe informacje o przebiegu
- **Artefakty** - pliki `fixtures_updater.log` i `state.json` (przechowywane 30 dni)

## Zmiana harmonogramu

Aby zmienić częstotliwość uruchomień, edytuj plik `.github/workflows/collect-match-data.yaml`:

```yaml
schedule:
  # Składnia: minuty godziny dzień_miesiąca miesiąc dzień_tygodnia
  - cron: '0 3 * * 1'  # Każdy poniedziałek o 3:00 UTC
```

### Przykłady harmonogramów:

```yaml
# Co 3 dni o 2:00 UTC
- cron: '0 2 */3 * *'

# Codziennie o 4:00 UTC
- cron: '0 4 * * *'

# Dwa razy w tygodniu (poniedziałek i piątek) o 3:00 UTC
- cron: '0 3 * * 1,5'

# Tylko w weekendy o 6:00 UTC
- cron: '0 6 * * 0,6'
```

**Generator cron:** https://crontab.guru/

## Zmiana parametrów domyślnych

W pliku `.github/workflows/collect-match-data.yaml` zmień wartości `default`:

```yaml
workflow_dispatch:
  inputs:
    days_back:
      default: '10'  # zmień tutaj
    season:
      default: '2024'  # zmień tutaj
```

## Limity i koszty

### GitHub Actions

- **Limity**: 2000 minut/miesiąc (darmowe konto)
- **Zużycie**: ~2-5 minut na uruchomienie
- **Koszty**: Darmowe dla publicznych repozytoriów

### API Football

- **Limity**: Zgodnie z Twoim planem API
- **Domyślnie ustawione**: 100 requestów/dzień
- Skrypt używa smart strategy - tylko niezbędne zapytania

### GitHub Storage

- **Limit**: 1 GB na repozytorium (soft limit)
- **Szacunek**: Dane z jednego sezonu to ~10-50 MB
- **Wystarczy na**: 20-100 sezonów

## Rozwiązywanie problemów

### Workflow nie uruchamia się automatycznie

- Sprawdź czy repo jest aktywne (miało aktywność w ostatnich 60 dniach)
- Upewnij się że workflow jest w branch `main` lub domyślnym

### Błąd: API key not found

1. Sprawdź czy secret `API_FOOTBALL_KEY` jest dodany
2. Nazwa musi być dokładnie: `API_FOOTBALL_KEY`

### Błąd: Permission denied

1. Przejdź do: **Settings** → **Actions** → **General**
2. W sekcji **Workflow permissions** wybierz:
   - ✅ **Read and write permissions**
3. Zapisz zmiany

### Dane nie są commitowane

- Sprawdź logi - może nie było nowych danych
- Zobacz w Summary czy `has_changes=true`

## Migracja do AWS S3 (przyszłość)

Jeśli dane przekroczą 500 MB lub będziesz zbierać wiele sezonów/lig, możesz przejść na AWS S3:

**Zalety:**
- Nieograniczona skalowalność
- Bardzo niskie koszty (~$0.023/GB/miesiąc)
- Szybki dostęp

**Wady:**
- Wymaga konta AWS
- Bardziej skomplikowana konfiguracja

Przygotowałem również wersję workflow z AWS S3 - daj znać jeśli będziesz potrzebować.

## Wsparcie

Jeśli masz pytania lub problemy:
1. Sprawdź logi w zakładce Actions
2. Zobacz artefakty z uruchomienia
3. Sprawdź czy secret API_FOOTBALL_KEY jest poprawnie skonfigurowany

## Następne kroki po wdrożeniu

Po pierwszym udanym uruchomieniu:

1. ✅ Sprawdź branch `data` - czy dane są tam zapisane
2. ✅ Zmień `days_back` na `10` w domyślnych parametrach (po zebraniu zaległych danych)
3. ✅ Sklonuj branch `data` lokalnie do analizy
4. ✅ Zintegruj z pipeline'em analizy danych

## Automatyczne powiadomienia (opcjonalne)

Możesz dodać powiadomienia (np. email, Slack) w przypadku błędów - zapytaj jeśli potrzebujesz.