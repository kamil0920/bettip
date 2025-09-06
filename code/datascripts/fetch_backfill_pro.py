import os
import json
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np  # <-- FIXED: Added missing import
from api_client_pro import ApiFootballClient
from tqdm import tqdm

# constants
LEAGUE_ID = 39  # Premier League (example)
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023]
BATCH_SIZE = 20  # fixtures per /fixtures?ids= call (max 20)
DEFAULT_CONCURRENCY = 10


def _to_json_safe(value):
    """Return a JSON/string-safe representation for nested or complex objects."""
    if value is None:
        return None

    # native Python primitives are OK
    if isinstance(value, (str, int, float, bool)):
        return value

    # numpy scalar -> Python native
    if isinstance(value, np.generic):
        return value.item()

    # numpy arrays, lists, tuples -> JSON-safe list
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(value.tolist() if isinstance(value, np.ndarray) else list(value))

    # dict
    if isinstance(value, dict):
        return json.dumps(value)

    # fallback: convert to string
    return str(value)


def sanitize_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert problematic columns to JSON-safe or consistent primitive types.
    Ensures parquet writes are always successful.
    """
    for col in df.columns:
        try:
            col_vals = df[col]
            if col_vals.dtype == "O":
                sample = col_vals.head(200).dropna()

                # Detect if we have a mix of bool and str in the same column
                unique_types = {type(v) for v in sample}

                # --- Fix boolean-like strings ---
                if str in unique_types and bool in unique_types:
                    df[col] = df[col].apply(
                        lambda x: None if pd.isna(x)
                        else (True if str(x).lower() == "true" else False if str(x).lower() == "false" else x)
                    )
                    continue

                # --- Fix numbers stored as strings (optional safeguard) ---
                if str in unique_types and (int in unique_types or float in unique_types):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    continue

                # --- For nested or complex objects ---
                if any(isinstance(v, (dict, list, tuple, np.ndarray, np.generic)) for v in sample):
                    print(f"[Sanitize] Column '{col}' has nested structures, converting to JSON-safe format...")
                    df[col] = col_vals.apply(_to_json_safe)

        except Exception as e:
            print(f"[Warning] Failed to sanitize column {col}: {e}")
            df[col] = df[col].apply(lambda v: None if v is None else str(v))

    return df


def ensure_season_dir(base_dir: Path, season: int) -> Path:
    sd = base_dir / str(season)
    sd.mkdir(parents=True, exist_ok=True)
    return sd


def fetch_fixtures_list(client: ApiFootballClient, season: int):
    resp = client.get("/fixtures", params={"league": LEAGUE_ID, "season": season})
    return resp.get("response", [])


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def download_batch(client: ApiFootballClient, batch_ids, season_dir: Path, season: int, batch_index: int):
    ids_str = "-".join(str(x) for x in batch_ids)
    data = client.get("/fixtures", params={"ids": ids_str})
    resp = data.get("response", [])

    df = pd.json_normalize(resp)
    df = sanitize_df_for_parquet(df)

    out_path = season_dir / f"fixtures_detailed_batch_{batch_index:04d}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path.name, [int(x) for x in batch_ids]


def merge_batches(season_dir: Path, pattern="fixtures_detailed_batch_*.parquet", out_name="fixtures_detailed_all.parquet"):
    files = sorted(season_dir.glob(pattern))
    if not files:
        return None

    dfs, bad = [], []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            bad.append((f.name, str(e)))

    if bad:
        print("Warning: Some batch files failed to read and will be skipped:")
        for fn, err in bad:
            print(f" - {fn}: {err}")

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = sanitize_df_for_parquet(df_all)

    out_path = season_dir / out_name
    df_all.to_parquet(out_path, index=False)
    return out_path


def save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill_season(client: ApiFootballClient, season: int, base_dir: Path, concurrency: int,
                    include_players=False, include_injuries=False):
    season_dir = ensure_season_dir(base_dir, season)
    batches_state_path = season_dir / "downloaded_batches.json"
    state = load_json(batches_state_path) or {"done_batches": []}

    save_json(batches_state_path, state)

    # Step 1: Fixtures list
    list_path = season_dir / f"fixtures_list_{season}.parquet"
    if not list_path.exists():
        print(f"Fetching fixtures list for season {season} ...")
        fixtures = fetch_fixtures_list(client, season)
        df_flist = sanitize_df_for_parquet(pd.json_normalize(fixtures))
        df_flist.to_parquet(list_path, index=False)
    else:
        print(f"Fixtures list for season {season} already exists, loading ...")

    df_list = pd.read_parquet(list_path)

    if "fixture.id" not in df_list.columns:
        raise RuntimeError(f"fixtures list file {list_path} missing expected column 'fixture.id'")

    fixture_ids = df_list["fixture.id"].tolist()
    total_batches = math.ceil(len(fixture_ids) / BATCH_SIZE)
    print(f"Season {season}: {len(fixture_ids)} fixtures -> {total_batches} batches (batch size {BATCH_SIZE})")

    # Prepare batch tasks
    batches = [(i, chunk) for i, chunk in enumerate(chunked(fixture_ids, BATCH_SIZE))]
    downloaded = set(state.get("done_batches", []))

    def worker(batch_tuple):
        idx, chunk_ids = batch_tuple
        if idx in downloaded:
            return ("skipped", idx)
        try:
            fname, ids = download_batch(client, chunk_ids, season_dir, season, idx)
            return ("done", idx, fname, ids)
        except Exception as e:
            return ("error", idx, str(e))

    # Run batches with threading
    pending = [b for b in batches if b[0] not in downloaded]
    if pending:
        print(f"Downloading {len(pending)} batches with concurrency={concurrency} ...")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(worker, b): b[0] for b in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Season {season} batches"):
                res = fut.result()
                if res[0] == "done":
                    _, idx, fname, ids = res
                    downloaded.add(idx)
                    state["done_batches"] = sorted(list(downloaded))
                    save_json(batches_state_path, state)
                elif res[0] == "error":
                    _, idx, err = res
                    print(f"Batch {idx} failed: {err}")
    else:
        print("No pending batches — skipping download.")

    # Merge
    merged = merge_batches(season_dir, out_name=f"fixtures_detailed_all_{season}.parquet")
    if merged:
        print(f"Merged detailed fixtures for season {season} -> {merged}")
    else:
        print("No detailed fixture batches found.")

    # Fetch teams
    teams_path = season_dir / f"teams_{season}.parquet"
    if not teams_path.exists():
        print("Fetching teams ...")
        try:
            teams_resp = client.get("/teams", params={"league": LEAGUE_ID, "season": season})
            teams_list = teams_resp.get("response", [])
            if teams_list:
                teams_df = sanitize_df_for_parquet(pd.json_normalize(teams_list))
                teams_df.to_parquet(teams_path, index=False)
                print(f"Saved teams -> {teams_path}")
            else:
                print("No teams returned for this season.")
        except Exception as e:
            print(f"Failed to fetch teams: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--base-dir", type=str, default="data/seasons")
    ap.add_argument("--include-players", action="store_true")
    ap.add_argument("--include-injuries", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    client = ApiFootballClient()

    for s in args.seasons:
        print("\n" + "=" * 80)
        print(f"Backfilling season {s}")
        backfill_season(client, s, base_dir, concurrency=args.concurrency,
                        include_players=args.include_players,
                        include_injuries=args.include_injuries)


if __name__ == "__main__":
    main()
