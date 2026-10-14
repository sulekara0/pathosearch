import pandas as pd
import pytest

from pathosearch.data.prepare_quilt import stratified_sample


def _candidates() -> pd.DataFrame:
    """Fake candidate pool: 3 categories x 10 rows, RangeIndex like collect_candidates()."""
    rows = [
        {"image_path": f"{cat}_{i}.jpg", "pathology": f"['{cat}']"}
        for cat in ("Bone", "Renal", "Breast")
        for i in range(10)
    ]
    return pd.DataFrame(rows)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "stratified_sample top-up uses df.index.isin(result.index) after result was "
        "reindexed (ignore_index=True), so it compares unrelated labels: already-selected "
        "rows can be drawn again and unselected rows with low index can never be drawn."
    ),
)
def test_stratified_sample_topup_returns_no_duplicates() -> None:
    # n=10 over 3 equal categories -> quota round(3.33)=3 each = 9 rows, so the
    # top-up branch (len(result) < n) must add exactly 1 *new* row.
    df = _candidates()
    for seed in range(30):
        out = stratified_sample(df, n=10, seed=seed)
        assert len(out) == 10
        assert out["image_path"].is_unique, f"duplicate row drawn with seed={seed}"
