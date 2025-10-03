import json
import pandas as pd


def first_notna(*vals):
    """Return first value from vals that is not None and not NaN/NA. Keeps 0."""
    for v in vals:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return None


def maybe_load_json(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        if (s[0] in ('{', '[')) and (s[-1] in ('}', ']')):
            try:
                return json.loads(x)
            except Exception:
                # attempt unescape
                try:
                    un = x.encode('utf-8').decode('unicode_escape')
                    return json.loads(un)
                except Exception:
                    return x
    return x


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if s and s[0] == '[' and s[-1] == ']':
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
    return []


def to_int_safe(v):
    """Convert v to int or return None. Handles NaN/pd.NA/empty strings."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    # if already int-like
    try:
        return int(v)
    except Exception:
        pass
    # try parsing numeric string like "123.0"
    try:
        s = str(v).strip()
        if s == "":
            return None
        # remove percent if present (not expected here but safe)
        s2 = s.replace("%", "")
        return int(float(s2))
    except Exception:
        return None


def get_nested_from_obj(obj, *keys):
    cur = obj
    for k in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def sanitize_for_write(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    def safe_val(v):
        if v is None: return None
        try:
            if pd.isna(v): return None
        except Exception:
            pass
        if isinstance(v, (str, int, float, bool)): return v
        if isinstance(v, (list, dict, tuple, np.ndarray)):
            try:
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                return json.dumps(v, default=str)
            except Exception:
                return str(v)
        return str(v)

    for col in df.columns:
        if df[col].dtype == 'O':
            sample = df[col].head(50)
            need = False
            for val in sample:
                if val is None:
                    continue
                if not isinstance(val, (str, int, float, bool)):
                    need = True
                    break
            if need:
                df[col] = df[col].apply(safe_val)
    return df
