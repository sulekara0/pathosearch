# adjudicate_app.py — resolve overlap disagreements (UI only; logic in pathosearch.eval.adjudication)
#
#   uv run streamlit run adjudicate_app.py -- --golden eval_data/golden_v0
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from pathosearch.config import load_settings
from pathosearch.errors import DataNotFoundError
from pathosearch.eval.adjudication import (
    DisagreementItem,
    adjudication_store,
    decide,
    load_disagreements,
)
from pathosearch.eval.golden import GoldenPaths
from pathosearch.eval.labeling import LabelStore, build_image_index, resolve_image

DECISIONS = ((0, "0 · İlgisiz"), (1, "1 · Kısmen"), (2, "2 · İlgili"), (None, "Emin değilim"))
CSS = """
<style>
.block-container { padding-top: 3rem; padding-bottom: 1rem; }
.st-key-image_box img { max-height: 55vh; object-fit: contain; }
</style>
"""


def parse_args() -> argparse.Namespace:
    s = load_settings()
    p = argparse.ArgumentParser(prog="adjudicate_app.py")
    p.add_argument(
        "--golden", type=Path, required=True, help="golden dir, e.g. eval_data/golden_v0"
    )
    p.add_argument(
        "--report", type=Path, default=None, help="default: reports/agreement_<golden>.json"
    )
    p.add_argument("--images-dir", type=Path, default=s.images_dir)
    args = p.parse_args(sys.argv[1:])
    if args.report is None:
        args.report = Path("reports") / f"agreement_{args.golden.name}.json"
    return args


@st.cache_resource(show_spinner="Görsel dizini taranıyor...")
def image_index(images_dir: str) -> dict[str, Path]:
    return build_image_index(Path(images_dir))


def shown(grade: int | None) -> str:
    return "Emin değilim" if grade is None else str(grade)


def on_decide(item: DisagreementItem, grade: int | None, n_items: int) -> None:
    decide(st.session_state.store, item, grade)
    st.session_state.cursor = min(st.session_state.cursor + 1, n_items - 1)


def on_move(step: int, n_items: int) -> None:
    st.session_state.cursor = max(0, min(st.session_state.cursor + step, n_items - 1))


def overview(items: list[DisagreementItem], store: LabelStore) -> None:
    rows = []
    for i, it in enumerate(items, 1):
        j = store.get(it.key)
        rows.append(
            {
                "#": i,
                "qid": it.qid,
                "sorgu": it.query,
                "|fark|": it.abs_diff,
                "A": it.a,
                "claude": it.b,
                "karar": "" if j is None else shown(j.grade),
                "görsel": it.image,
                "not": it.note or "",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Uyuşmazlık kararı", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        args = parse_args()
    except SystemExit:
        st.error(
            "Kullanım: `streamlit run adjudicate_app.py -- --golden <dizin> [--report <json>]`"
        )
        st.stop()
    golden = GoldenPaths(args.golden)
    try:
        items = load_disagreements(args.report, golden)
    except (DataNotFoundError, ValueError, KeyError) as exc:
        st.error(f"Uyuşmazlıklar okunamadı:\n\n{exc}")
        st.stop()
    if not items:
        st.success("Raporda uyuşmazlık yok.")
        st.stop()

    if st.session_state.get("store_key") != str(golden.adjudication):
        st.session_state.store = adjudication_store(golden)
        st.session_state.store_key = str(golden.adjudication)
        st.session_state.cursor = 0
    store: LabelStore = st.session_state.store
    n = len(items)
    item = items[st.session_state.cursor]
    n_done = sum(1 for it in items if store.is_done(it.key))

    q_col, p_col = st.columns([6, 3], vertical_alignment="center")
    q_col.markdown(
        f"**{st.session_state.cursor + 1}/{n}** · `{item.qid}` · **Sorgu:** {item.query} "
        f"· _{item.category}_"
    )
    p_col.progress(n_done / n, text=f"Karar verilen: {n_done} / {n}")

    try:
        path = resolve_image(image_index(str(args.images_dir)), item.image)
    except (DataNotFoundError, ValueError) as exc:
        st.error(f"{exc}\n\nBaşka görsele geçilmedi.")
        st.stop()

    img_col, info_col = st.columns([3, 2])
    with img_col, st.container(key="image_box"):
        st.image(str(path), width="stretch")
        st.caption(item.image)
    with info_col:
        a_col, b_col, d_col = st.columns(3)
        a_col.metric("A", item.a)
        b_col.metric("claude", item.b)
        d_col.metric("|fark|", item.abs_diff)
        st.markdown("**claude notu**")
        st.info(item.note or "(bu çift için not bulunamadı)")
        previous = store.get(item.key)
        if previous is not None:
            st.success(f"Mevcut karar: **{shown(previous.grade)}** (yeni karar bunun yerine geçer)")

        cols = st.columns(4)
        for col, (grade, label) in zip(cols, DECISIONS, strict=True):
            col.button(
                label,
                key=f"decide_{'unsure' if grade is None else grade}",
                on_click=on_decide,
                args=(item, grade, n),
                width="stretch",
                type="primary" if grade is not None else "secondary",
            )
        prev_col, next_col = st.columns(2)
        prev_col.button(
            "← Önceki",
            key="prev",
            on_click=on_move,
            args=(-1, n),
            disabled=st.session_state.cursor == 0,
            width="stretch",
        )
        next_col.button(
            "Sonraki →",
            key="next",
            on_click=on_move,
            args=(1, n),
            disabled=st.session_state.cursor == n - 1,
            width="stretch",
        )

    with st.expander(f"Tüm uyuşmazlıklar ({n}, |fark|'a göre sıralı)"):
        overview(items, store)
    st.caption(
        f"Kararlar `{golden.adjudication.as_posix()}` dosyasına eklenir; A ve claude etiket "
        "dosyalarına yazılmaz. Eğitim/araştırma aracıdır; klinik teşhis aracı değildir."
    )


main()
