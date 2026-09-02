"""
Phase 1 verification (CLAUDE.md §4.3 checklist item 3):
train / test_spatial / test_temporal must be non-random physical holdouts with
ZERO overlapping rows.

Skips (does not fail) if the pipeline has not been run yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
TRAIN = PROCESSED / "train.npz"
SPATIAL = PROCESSED / "test_spatial.npz"
TEMPORAL = PROCESSED / "test_temporal.npz"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built yet — run `python -m src.data_pipeline`")
    return np.load(p, allow_pickle=True)


def _ids(npz):
    return set(npz["profile_id"].tolist()) if "profile_id" in npz.files else set()


def test_splits_are_disjoint():
    tr, sp, te = _load(TRAIN), _load(SPATIAL), _load(TEMPORAL)
    a, b, c = _ids(tr), _ids(sp), _ids(te)
    assert a and b, "train and spatial-holdout must both be non-empty"
    assert not (a & b), f"train ∩ spatial = {len(a & b)} profiles"
    assert not (a & c), f"train ∩ temporal = {len(a & c)} profiles"
    assert not (b & c), f"spatial ∩ temporal = {len(b & c)} profiles"


def test_spatial_holdout_is_bay_of_bengal():
    sp = _load(SPATIAL)
    lat, lon = sp["lat"], sp["lon"]
    assert np.all((lat >= 5.0) & (lon >= 85.0)), \
        "every spatial-holdout row must be in the BoB box (lat>=5, lon>=85)"


def test_train_excludes_bay_of_bengal():
    tr = _load(TRAIN)
    lat, lon = tr["lat"], tr["lon"]
    assert not np.any((lat >= 5.0) & (lon >= 85.0)), \
        "train must contain no BoB-box rows"


def test_temporal_holdout_is_one_monsoon():
    te = _load(TEMPORAL)
    if len(te["month"]) == 0:
        pytest.skip("temporal holdout empty")
    assert set(np.unique(te["month"]).tolist()) <= {6, 7, 8, 9}
    assert len(np.unique(te["year"])) == 1, "temporal holdout must be a single year"


def test_train_has_a_full_season_of_data():
    tr = _load(TRAIN)
    assert len(tr["profile_id"]) >= 500, "expected at least a season of matched rows"
