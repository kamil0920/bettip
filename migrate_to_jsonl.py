# scripts/migrate_to_jsonl.py
import json
import shutil
from pathlib import Path

# Ustawienia
LEAGUE = "premier_league"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
BASE_DIR = Path("data/01-raw")


def migrate_folder(season_path: Path, type_name: str):
    source_dir = season_path / type_name
    target_file = season_path / f"{type_name}.jsonl"

    if not source_dir.exists():
        print(f"Skipping {source_dir} (not found)")
        return

    print(f"Migrating {source_dir} -> {target_file}")

    # Zbieramy wszystkie pliki json
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        return

    with open(target_file, 'w', encoding='utf-8') as outfile:
        count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as infile:
                    content = json.load(infile)
                    # Jeśli dane są opakowane w {"data": ...}, wyciągamy je
                    data_to_write = content.get('data', content)

                    json.dump(data_to_write, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    count += 1
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    print(f"✅ Migrated {count} files. Deleting old folder...")
    shutil.rmtree(source_dir)


if __name__ == "__main__":
    for year in YEARS:
        season_path = BASE_DIR / LEAGUE / str(year)
        if season_path.exists():
            print(f"--- Processing Season {year} ---")
            migrate_folder(season_path, "events")
            migrate_folder(season_path, "lineups")
            migrate_folder(season_path, "players")
