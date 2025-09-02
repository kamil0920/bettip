import os
import json
import math
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from api_client_pro import ApiFootballClient
from tqdm import tqdm

# constants
LEAGUE_ID = 39
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023]
BATCH_SIZE = 20  # fixtures per /fixtures?ids= call (max 20)
DEFAULT_CONCURRENCY = 10

def ensure_season_dir(base_dir: Path, season: int) -> Path:
    sd = base_dir / str(season)
    sd.mkdir(parents=True, exist_ok=True)
    return sd

def fetch_fixtures_list(client: ApiFootballClient, season: int):
    resp = client.get("/fixtures", params={"league": LEAGUE_ID, "season": season})
    return resp.get("response", [])

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def download_batch(client: ApiFootballClient, batch_ids, season_dir: Path, season: int, batch_index: int):
    ids_str = "-".join(str(x) for x in batch_ids)
    # call API
    data = client.get("/fixtures", params={"ids": ids_str})
    resp = data.get("response", [])
    # save batch to parquet
    df = pd.json_normalize(resp)
    out_path = season_dir / f"fixtures_detailed_batch_{batch_index:04d}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path.name, [int(x) for x in batch_ids]

def merge_batches(season_dir: Path, pattern="fixtures_detailed_batch_*.parquet", out_name="fixtures_detailed_all.parquet"):
    files = sorted(season_dir.glob("fixtures_detailed_batch_*.parquet"))
    if not files:
        return None
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df_all = pd.concat(dfs, ignore_index=True)
    out_path = season_dir / out_name
    df_all.to_parquet(out_path, index=False)
    return out_path

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def backfill_season(client: ApiFootballClient, season: int, base_dir: Path, concurrency: int, include_players=False, include_injuries=False):
    season_dir = ensure_season_dir(base_dir, season)
    batches_state_path = season_dir / "downloaded_batches.json"
    state = load_json(batches_state_path)

    # 1) fixtures list
    list_path = season_dir / f"fixtures_list_{season}.parquet"
    if not list_path.exists():
        print(f"Fetching fixtures list for season {season} ...")
        fixtures = fetch_fixtures_list(client, season)
        pd.json_normalize(fixtures).to_parquet(list_path, index=False)
    else:
        print(f"Fixtures list for season {season} already exists, loading ...")

    df_list = pd.read_parquet(list_path)
    fixture_ids = df_list["fixture.id"].tolist()
    total_batches = math.ceil(len(fixture_ids) / BATCH_SIZE)
    print(f"Season {season}: {len(fixture_ids)} fixtures -> {total_batches} batches (batch size {BATCH_SIZE})")

    # create batch map
    batches = []
    for i, chunk in enumerate(chunked(fixture_ids, BATCH_SIZE)):
        batches.append((i, chunk))

    # determine which batches already downloaded
    downloaded = set(state.get("done_batches", []))

    # worker function closure
    def worker(batch_tuple):
        idx, chunk_ids = batch_tuple
        if idx in downloaded:
            return ("skipped", idx)
        try:
            fname, ids = download_batch(client, chunk_ids, season_dir, season, idx)
            return ("done", idx, fname, ids)
        except Exception as e:
            return ("error", idx, str(e))

    # run with ThreadPoolExecutor
    pending = [b for b in batches if b[0] not in downloaded]
    if pending:
        print(f"Downloading {len(pending)} batches for season {season} with concurrency={concurrency} ...")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(worker, b): b[0] for b in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Season {season} batches"):
                res = fut.result()
                if res[0] == "done":
                    _, idx, fname, ids = res
                    # record done
                    downloaded.add(idx)
                    state["done_batches"] = sorted(list(downloaded))
                    save_json(batches_state_path, state)
                elif res[0] == "skipped":
                    # already done
                    pass
                else:
                    # error
                    _, idx, err = res
                    print(f"Batch {idx} failed: {err}")

    else:
        print(f"No pending batches for season {season}")

    # merge all batches into a single file
    merged = merge_batches(season_dir, out_name=f"fixtures_detailed_all_{season}.parquet")
    if merged:
        print(f"Merged detailed fixtures for season {season} -> {merged}")
    else:
        print(f"No detailed fixture batches found for season {season}")

    # optional: fetch players per team
    if include_players:
        print("Fetching players per team for season", season)
        teams = pd.read_parquet(season_dir / f"teams_{season}.parquet") if (season_dir / f"teams_{season}.parquet").exists() else None
        if teams is not None:
            players_list = []
            for tid in teams["team.id"].unique():
                try:
                    data = client.get("/players", params={"team": int(tid), "season": season})
                    players_list.extend(data.get("response", []))
                except Exception as e:
                    print(f"Failed to fetch players for team {tid}: {e}")
            if players_list:
                pd.json_normalize(players_list).to_parquet(season_dir / f"players_{season}.parquet", index=False)
                print("Saved players file")
        else:
            print("teams file not present in season dir; skip players fetch")

    if include_injuries:
        print("Fetching injuries per team for season", season)
        teams = pd.read_parquet(season_dir / f"teams_{season}.parquet") if (season_dir / f"teams_{season}.parquet").exists() else None
        inj_list = []
        if teams is not None:
            for tid in teams["team.id"].unique():
                try:
                    data = client.get("/injuries", params={"team": int(tid), "season": season})
                    inj_list.extend(data.get("response", []))
                except Exception as e:
                    print(f"Failed to fetch injuries for team {tid}: {e}")
            if inj_list:
                pd.json_normalize(inj_list).to_parquet(season_dir / f"injuries_{season}.parquet", index=False)
                print("Saved injuries file")
        else:
            print("teams file not present in season dir; skip injuries fetch")

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
        print("\n" + "="*80)
        print(f"Backfilling season {s}")
        backfill_season(client, s, base_dir, concurrency=args.concurrency, include_players=args.include_players, include_injuries=args.include_injuries)

if __name__ == "__main__":
    main()
