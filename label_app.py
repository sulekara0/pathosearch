# label_app.py — golden set labeling tool (UI only; logic in pathosearch.eval.labeling)
#
#   uv run streamlit run label_app.py -- --golden eval_data/golden_v0 --annotator A
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from pathosearch.config import load_settings
from pathosearch.errors import DataNotFoundError
from pathosearch.eval.golden import GoldenPaths
from pathosearch.eval.labeling import (
    PRIMARY_ANNOTATOR,
    LabelingSession,
    LabelStore,
    Task,
    build_image_index,
    format_elapsed,
    load_first_captions,
    load_queue,
    resolve_image,
)

GRADE_BUTTONS = ((2, "2 · İlgili"), (1, "1 · Kısmen"), (0, "0 · İlgisiz"))
RULES = """
| Etiket | Kural |
|---|---|
| **2** | Sorgunun doku/organını **ve** sorgulanan bulguyu gösteriyor |
| **1** | Doğru organ/uzmanlık veya yakın bulgu, ama sorgulanan antite değil / kısmen |
| **0** | İlgisiz, histoloji değil veya sorguyla çelişiyor |
| **Emin değilim** | Karar verilemedi; qrels'e girmez, sayısı raporlanır |
"""
CSS = """
<style>
.st-key-query_header {
    position: sticky; top: 2.9rem; z-index: 100; padding: 0.2rem 0 0.4rem;
    background: var(--background-color, white); border-bottom: 1px solid rgba(128,128,128,.35);
}
.block-container { padding-top: 3rem; padding-bottom: 1rem; }
.st-key-image_box img { max-height: 46vh; object-fit: contain; }
.st-key-image_box_full img { object-fit: contain; }
</style>
"""


def parse_args() -> argparse.Namespace:
    s = load_settings()
    p = argparse.ArgumentParser(prog="label_app.py")
    p.add_argument(
        "--golden", type=Path, required=True, help="golden dir, e.g. eval_data/golden_v0"
    )
    p.add_argument(
        "--annotator", required=True, help=f"annotator id ({PRIMARY_ANNOTATOR} = full pool)"
    )
    p.add_argument("--images-dir", type=Path, default=s.images_dir)
    p.add_argument("--csv", type=Path, default=s.csv_path, help="Quilt lookup CSV (captions)")
    return p.parse_args(sys.argv[1:])


@st.cache_resource(show_spinner="Görsel dizini taranıyor...")
def image_index(images_dir: str) -> dict[str, Path]:
    return build_image_index(Path(images_dir))


@st.cache_resource(show_spinner="Caption'lar CSV'den okunuyor...")
def captions(csv_path: str, images: tuple[str, ...]) -> dict[str, str]:
    return load_first_captions(Path(csv_path), images)


def get_session(args: argparse.Namespace) -> LabelingSession:
    key = f"session:{args.golden}:{args.annotator}"
    if st.session_state.get("session_key") != key:
        golden = GoldenPaths(args.golden)
        queue = load_queue(golden, args.annotator)
        store = LabelStore(golden.judgments(args.annotator), args.annotator)
        st.session_state.session = LabelingSession(queue, store)
        st.session_state.session_key = key
        st.session_state.caption_key = None
    session: LabelingSession = st.session_state.session
    return session


# --- callbacks (run before the script reruns) ------------------------------------


def on_grade(grade: int | None) -> None:
    session: LabelingSession = st.session_state.session
    task = session.current
    if task is not None:
        session.submit(grade, caption_seen=st.session_state.caption_key == task.key)
    st.session_state.caption_key = None


def on_undo() -> None:
    session: LabelingSession = st.session_state.session
    session.undo()
    st.session_state.caption_key = None


def on_show_caption(key: tuple[str, str]) -> None:
    st.session_state.caption_key = key


# --- rendering ---------------------------------------------------------------------


@st.fragment(run_every="1s")
def clock(session: LabelingSession) -> None:
    spl = session.seconds_per_label()
    per_label = "–" if spl is None else f"{spl:.1f}"
    st.markdown(
        f"⏱ **{format_elapsed(session.elapsed())}** · oturum **{session.n_submitted}** · "
        f"**{per_label}** sn/etiket"
    )


def header(session: LabelingSession, task: Task | None) -> None:
    """Query, progress and counters on a single sticky row."""
    with st.container(key="query_header"):
        q_col, p_col, c_col = st.columns([5, 3, 3], vertical_alignment="center")
        if task is not None:
            q_col.markdown(f"**Sorgu:** {task.query.text}")
        p_col.progress(
            session.done / max(session.total, 1), text=f"{session.done} / {session.total}"
        )
        with c_col:
            clock(session)


def grading_buttons() -> None:
    cols = st.columns(4)
    for col, (grade, label) in zip(cols, GRADE_BUTTONS, strict=False):
        col.button(
            label,
            key=f"grade_{grade}",
            shortcut=str(grade),
            on_click=on_grade,
            args=(grade,),
            width="stretch",
            type="primary",
        )
    cols[3].button(
        "Emin değilim", key="grade_unsure", on_click=on_grade, args=(None,), width="stretch"
    )


def main() -> None:
    st.set_page_config(page_title="Golden set etiketleme", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        args = parse_args()
    except SystemExit:
        st.error("Kullanım: `streamlit run label_app.py -- --golden <dizin> --annotator <id>`")
        st.stop()
    try:
        session = get_session(args)
    except (DataNotFoundError, ValueError) as exc:
        st.error(f"Golden dizini okunamadı:\n\n{exc}")
        st.stop()

    task = session.current
    header(session, task)
    if task is None:
        st.success(f"Tüm çiftler etiketlendi ({session.total}). Annotator: {args.annotator}")
        st.button(
            "Geri al", key="undo", shortcut="u", on_click=on_undo, disabled=not session.can_undo
        )
        st.caption("Eğitim/araştırma aracıdır; klinik teşhis aracı değildir.")
        return

    key = task.key
    try:
        path = resolve_image(image_index(str(args.images_dir)), task.item.image)
    except (DataNotFoundError, ValueError) as exc:
        st.error(f"{exc}\n\nBaşka görsele geçilmedi; etiketleme durduruldu.")
        st.stop()

    previous = session.store.get(key)
    if session.revisiting and previous is not None:
        shown = "Emin değilim" if previous.grade is None else previous.grade
        st.info(f"Geri alınan çift. Önceki etiket: **{shown}** (yeni etiket bunun yerine geçer)")

    full_size = bool(st.session_state.get("full_size", False))
    with st.container(key="image_box_full" if full_size else "image_box"):
        st.image(str(path), width="stretch")

    grading_buttons()

    # secondary controls; the caption button stays separate from the grading row
    undo_col, cap_col, full_col = st.columns(3, vertical_alignment="center")
    undo_col.button(
        "↶ Geri al (u)", key="undo", shortcut="u", on_click=on_undo, disabled=not session.can_undo
    )
    cap_col.button(
        "Caption göster",
        key="show_caption",
        on_click=on_show_caption,
        args=(key,),
        help="Caption'ı açmak etikete 'caption_seen=true' olarak kaydedilir.",
    )
    full_col.toggle("Tam boy", key="full_size")
    if st.session_state.caption_key == key:
        try:
            cap = captions(str(args.csv), tuple(sorted({t.item.image for t in session.queue}))).get(
                task.item.image
            )
        except DataNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        st.info(cap if cap else "(CSV'de bu görsel için caption bulunamadı)")

    with st.expander("Etiket kuralı"):
        st.markdown(RULES)
    st.caption("Eğitim/araştırma aracıdır; klinik teşhis aracı değildir.")


main()
