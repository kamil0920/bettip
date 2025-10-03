"""Extraction functions for fixtures, events, and player stats."""

import json
import pandas as pd
from helpers import maybe_load_json, first_notna, to_int_safe, get_nested_from_obj, ensure_list


def extract_fixture_core(series):
    """Extract core match info from a row (pandas Series)."""

    def try_keys(*keys):
        dotted = ".".join(keys)
        if dotted in series.index and pd.notna(series.get(dotted)):
            v = series.get(dotted)
            if isinstance(v, str):
                return maybe_load_json(v)
            return v
        if keys[0] in series.index:
            val = series.get(keys[0])
            if isinstance(val, str):
                val = maybe_load_json(val)
            if isinstance(val, dict):
                return get_nested_from_obj(val, *keys[1:]) if len(keys) > 1 else val
        # scan other dict-like columns
        for c in series.index:
            v = series.get(c)
            if isinstance(v, str):
                v = maybe_load_json(v)
            if isinstance(v, dict):
                cur = v
                found = True
                for k in keys:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        found = False
                        break
                if found:
                    return cur
        return None

    fixture_id = first_notna(
        try_keys("fixture", "id"),
        series.get("fixture.id"),
        series.get("id")
    )

    date = first_notna(
        try_keys("fixture", "date"),
        series.get("fixture.date"),
        series.get("date")
    )

    timestamp = first_notna(
        try_keys("fixture", "timestamp"),
        series.get("fixture.timestamp"),
        series.get("timestamp")
    )

    referee = first_notna(
        try_keys("fixture", "referee"),
        series.get("fixture.referee"),
        series.get("referee")
    )

    venue_id = first_notna(
        try_keys("fixture", "venue", "id"),
        series.get("fixture.venue.id"),
        series.get("venue.id")
    )

    venue_name = first_notna(
        try_keys("fixture", "venue", "name"),
        series.get("fixture.venue.name"),
        series.get("venue.name")
    )

    status = first_notna(
        try_keys("fixture", "status", "short"),
        series.get("fixture.status.short"),
        series.get("status")
    )

    league_id = first_notna(
        try_keys("league", "id"),
        series.get("league.id"),
        series.get("league_id")
    )

    round_name = first_notna(
        try_keys("league", "round"),
        series.get("league.round"),
        series.get("round")
    )

    home_id = first_notna(
        try_keys("teams", "home", "id"),
        series.get("teams.home.id")
    )
    home_name = first_notna(
        try_keys("teams", "home", "name"),
        series.get("teams.home.name")
    )
    away_id = first_notna(
        try_keys("teams", "away", "id"),
        series.get("teams.away.id")
    )
    away_name = first_notna(
        try_keys("teams", "away", "name"),
        series.get("teams.away.name")
    )

    ft_home = first_notna(
        try_keys("score", "fulltime", "home"),
        try_keys("goals", "home"),
        series.get("goals.home"),
        series.get("goals_home")
    )
    ft_away = first_notna(
        try_keys("score", "fulltime", "away"),
        try_keys("goals", "away"),
        series.get("goals.away"),
        series.get("goals_away")
    )
    ht_home = first_notna(
        try_keys("score", "halftime", "home"),
        series.get("score.halftime.home"),
        series.get("ht_home")
    )
    ht_away = first_notna(
        try_keys("score", "halftime", "away"),
        series.get("score.halftime.away"),
        series.get("ht_away")
    )

    return {
        "fixture_id": to_int_safe(fixture_id),
        "date": date,
        "timestamp": to_int_safe(timestamp),
        "referee": referee,
        "venue_id": to_int_safe(venue_id),
        "venue_name": venue_name,
        "status": status,
        "league_id": to_int_safe(league_id),
        "round": round_name,
        "home_team_id": to_int_safe(home_id),
        "home_team_name": home_name,
        "away_team_id": to_int_safe(away_id),
        "away_team_name": away_name,
        "ft_home": to_int_safe(ft_home),
        "ft_away": to_int_safe(ft_away),
        "ht_home": to_int_safe(ht_home),
        "ht_away": to_int_safe(ht_away),
    }


def extract_events_from_row(series):
    events_found = []
    candidates = []

    # explicit 'events' col
    if 'events' in series.index:
        v = series.get('events')
        if isinstance(v, str):
            v = maybe_load_json(v)
        if isinstance(v, list):
            candidates.append(v)
        elif isinstance(v, dict):
            # dict of lists?
            for val in v.values():
                if isinstance(val, list):
                    candidates.append(val)

    # scan other columns for list-of-dicts
    for c in series.index:
        v = series.get(c)
        if isinstance(v, str):
            v = maybe_load_json(v)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append(v)

    fixture_id = series.get('fixture_id') or series.get('fixture.id')
    for cand in candidates:
        for ev in ensure_list(cand):
            if not isinstance(ev, dict):
                continue
            minute = None
            extra = None
            time_obj = ev.get('time') or ev.get('minute') or ev.get('elapsed')
            if isinstance(time_obj, dict):
                minute = time_obj.get('elapsed') or time_obj.get('minute')
                extra = time_obj.get('extra')
            else:
                minute = time_obj
            etype = ev.get('type') or ev.get('event') or ev.get('result') or ev.get('detail')
            edetail = ev.get('detail') or ev.get('description') or ev.get('comment')
            team = ev.get('team') or {}
            team_id = team.get('id') if isinstance(team, dict) else None
            player = ev.get('player') or {}
            player_id = None
            if isinstance(player, dict):
                player_id = player.get('id') or player.get('player_id')
            else:
                player_id = ev.get('player_id') or ev.get('playerId')
            assist = ev.get('assist') or {}
            assist_id = assist.get('id') if isinstance(assist, dict) else None

            events_found.append({
                "fixture_id": to_int_safe(fixture_id),
                "minute": to_int_safe(minute),
                "extra": to_int_safe(extra),
                "type": etype,
                "detail": edetail,
                "team_id": to_int_safe(team_id),
                "player_id": to_int_safe(player_id),
                "assist_id": to_int_safe(assist_id),
                "raw": json.dumps(ev, default=str)
            })
    return events_found


def extract_player_stats_from_row(series):
    out = []
    # preferred 'players' column
    if 'players' in series.index:
        val = series.get('players')
        if isinstance(val, str):
            val = maybe_load_json(val)
        if isinstance(val, list):
            for block in val:
                if not isinstance(block, dict):
                    continue
                team_obj = block.get('team') or {}
                team_id = team_obj.get('id') if isinstance(team_obj, dict) else None
                players_block = block.get('players') or block.get('statistics') or block.get('players')
                if isinstance(players_block, str):
                    players_block = maybe_load_json(players_block)
                if isinstance(players_block, list):
                    for p in players_block:
                        player_meta = p.get('player') if isinstance(p, dict) else None
                        stats = p.get('statistics') or p.get('stats') or p.get('statistics')
                        if isinstance(stats, dict):
                            stats = [stats]
                        if isinstance(stats, list) and stats:
                            st = stats[0]
                        else:
                            st = p
                        player_id = None
                        player_name = None
                        if isinstance(player_meta, dict):
                            player_id = player_meta.get('id') or player_meta.get('player_id')
                            player_name = player_meta.get('name')
                        else:
                            player_id = p.get('player_id') or p.get('id')
                            player_name = p.get('player_name') or p.get('name')
                        minutes = None
                        goals = None
                        assists = None
                        yellow = None
                        red = None
                        if isinstance(st, dict):
                            minutes = get_nested_from_obj(st, 'games', 'minutes') or st.get('minutes')
                            goals = get_nested_from_obj(st, 'goals', 'total') if isinstance(st.get('goals'),
                                                                                            dict) else st.get('goals')
                            assists = get_nested_from_obj(st, 'goals', 'assists') if isinstance(st.get('goals'),
                                                                                                dict) else st.get(
                                'assists')
                            yellow = get_nested_from_obj(st, 'cards', 'yellow') if isinstance(st.get('cards'),
                                                                                              dict) else st.get(
                                'yellow')
                            red = get_nested_from_obj(st, 'cards', 'red') if isinstance(st.get('cards'),
                                                                                        dict) else st.get('red')
                        out.append({
                            "fixture_id": to_int_safe(series.get('fixture_id') or series.get('fixture.id')),
                            "player_id": to_int_safe(player_id),
                            "player_name": player_name,
                            "team_id": to_int_safe(team_id),
                            "minutes": to_int_safe(minutes),
                            "goals": to_int_safe(goals) if goals is not None else 0,
                            "assists": to_int_safe(assists),
                            "yellow_cards": to_int_safe(yellow),
                            "red_cards": to_int_safe(red),
                            "raw": json.dumps(p, default=str)
                        })
    else:
        # fallback scan
        for c in series.index:
            v = series.get(c)
            if isinstance(v, str):
                v = maybe_load_json(v)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                sample = v[0]
                if 'player' in sample or 'player_id' in sample or 'statistics' in sample:
                    for p in v:
                        player_id = None
                        player_name = None
                        if isinstance(p.get('player'), dict):
                            player_id = p['player'].get('id')
                            player_name = p['player'].get('name')
                        else:
                            player_id = p.get('player_id') or p.get('id')
                            player_name = p.get('player_name') or p.get('name')
                        goals = p.get('goals') or (
                            p.get('statistics', [{}])[0].get('goals') if p.get('statistics') else None)
                        yellow = p.get('yellow') or (p.get('cards', {}).get('yellow') if p.get('cards') else None)
                        out.append({
                            "fixture_id": to_int_safe(series.get('fixture_id') or series.get('fixture.id')),
                            "player_id": to_int_safe(player_id),
                            "player_name": player_name,
                            "team_id": None,
                            "minutes": None,
                            "goals": to_int_safe(goals),
                            "assists": None,
                            "yellow_cards": to_int_safe(yellow),
                            "red_cards": None,
                            "raw": json.dumps(p, default=str)
                        })
    return out
