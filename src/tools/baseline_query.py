"""Bounded query pagination and single-entry compaction for generated baselines."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_QUERY_BYTES = 10_000
_default_max_results = 25
_min_query_max_bytes = 512


@dataclass(frozen=True)
class QueryOptions:
    """Pagination, byte-budget, and full-data options for baseline queries."""

    offset: int = 0
    limit: int | None = None
    max_bytes: int = DEFAULT_MAX_QUERY_BYTES
    full_data: bool = False

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset must be >= 0")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")
        if not self.full_data and self.max_bytes < _min_query_max_bytes:
            raise ValueError(f"max_bytes must be >= {_min_query_max_bytes}")


def _serialized_bytes(payload: dict[str, Any]) -> int:
    return len((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _make_envelope(
    meta: tuple[str, str, int],
    opts: QueryOptions,
    matches: Any,
    state: tuple[int | None, bool],
) -> dict[str, Any]:
    b_id, query, total = meta
    next_off, item_trunc = state
    lim = opts.limit or (None if opts.full_data else _default_max_results)
    more = next_off is not None
    return {
        "id": b_id,
        "query": query,
        "match_count": total,
        "total_matches": total,
        "returned_matches": len(matches),
        "offset": opts.offset,
        "limit": lim,
        "max_bytes": None if opts.full_data else opts.max_bytes,
        "full_data": opts.full_data,
        "truncated": more or item_trunc,
        "has_more": more,
        "item_truncated": item_trunc,
        "next_offset": next_off,
        "matches": matches,
    }


def _compact_api_entry(
    item: tuple[str, Any], fits: Callable[[Any], bool]
) -> tuple[str, Any]:
    mod, content = item
    syms = content.get("symbols", {}) if isinstance(content, dict) else {}
    raw_all = content.get("__all__") if isinstance(content, dict) else None
    base: dict[str, Any] = {"symbol_count": len(syms), "truncated": True}
    if isinstance(content, dict) and "explicit_all" in content:
        base["explicit_all"] = content["explicit_all"]
    if isinstance(raw_all, list):
        base.update({"all_count": len(raw_all), "__all__": raw_all[:8]})
    elif raw_all is not None:
        base["__all__"] = raw_all
    packed: dict[str, Any] = {}
    for s_name, s_def in sorted(syms.items()):
        if fits({mod: {**base, "symbols": {**packed, s_name: s_def}}}):
            packed[s_name] = s_def
            continue
        if not packed:
            s_type = s_def.get("type") if isinstance(s_def, dict) else None
            stub = {s_name[:64]: {"type": s_type, "truncated": True}}
            if fits({mod: {**base, "symbols": stub}}):
                packed = stub
        break
    if fits({mod: {**base, "symbols": packed}}):
        return (mod, {**base, "symbols": packed})
    keep = ("explicit_all", "symbol_count", "truncated")
    mini = {**{k: base[k] for k in keep if k in base}, "symbols": {}}
    return (mod if fits({mod: mini}) else f"{mod[:48]}...", mini)


def _compact_seq_entry(item: Any, fits: Callable[[Any], bool]) -> Any:
    if isinstance(item, dict):
        raw_s = item.get("stats", {})
        core = ("min", "max", "mean", "stddev", "median", "rounds")
        stats = {k: raw_s[k] for k in core if k in raw_s}
        nm, fn = str(item.get("name", "")), str(item.get("fullname", ""))
        for cut in (120, 48, 16):
            cand: dict[str, Any] = {"name": nm[:cut], "fullname": fn[:cut]}
            cand.update({"stats": stats, "truncated": True})
            if fits([cand]):
                return cand
        return {"name": nm[:16], "truncated": True}
    lno, txt = item[0], str(item[1])
    lo, hi, best = 0, len(txt), "..."
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_s = f"{txt[:mid]}..."
        ok = fits([[lno, cand_s]])
        lo, hi, best = (mid + 1, hi, cand_s) if ok else (lo, mid - 1, best)
    return [lno, best]


def paginate_matches(
    meta: tuple[str, str], items: list[Any], is_map: bool, opts: QueryOptions
) -> dict[str, Any]:
    """Pack matched entries within ``opts`` limits and enforce byte postconditions."""
    total, f_meta = len(items), (meta[0], meta[1], len(items))
    pack = dict if is_map else list

    def _env(seq: list[Any], st: tuple[int | None, bool]) -> dict[str, Any]:
        return _make_envelope(f_meta, opts, pack(seq), st)

    if opts.offset >= total:
        return _env([], (None, False))
    if opts.full_data:
        end = total if opts.limit is None else min(total, opts.offset + opts.limit)
        return _env(items[opts.offset : end], (end if end < total else None, False))
    cap = min(total, opts.offset + (opts.limit or _default_max_results))
    chosen: list[Any] = []
    for idx in range(opts.offset, cap):
        trial, nxt = [*chosen, items[idx]], (idx + 1 if idx + 1 < total else None)
        if _serialized_bytes(_env(trial, (nxt, False))) <= opts.max_bytes:
            chosen = trial
            continue
        break
    if not chosen:
        nxt_one = opts.offset + 1 if opts.offset + 1 < total else None

        def _fits(cand_m: Any) -> bool:
            doc = _make_envelope(f_meta, opts, cand_m, (nxt_one, True))
            return _serialized_bytes(doc) <= opts.max_bytes

        comp_fn = _compact_api_entry if is_map else _compact_seq_entry
        res = _env([comp_fn(items[opts.offset], _fits)], (nxt_one, True))
    else:
        end_i = opts.offset + len(chosen)
        res = _env(chosen, (end_i if end_i < total else None, False))
    if _serialized_bytes(res) > opts.max_bytes:
        raise RuntimeError(f"Query output exceeded max_bytes={opts.max_bytes}")
    return res
