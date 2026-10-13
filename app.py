# app.py — Histopatoloji Multimodal Arama Demo (Streamlit)
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from PIL import Image

from pathosearch import pipeline
from pathosearch.config import load_settings, require_dir, require_file
from pathosearch.domain_gate import DomainGate  # CLIP text-anchor domain gate
from pathosearch.errors import DataNotFoundError
from pathosearch.pipeline import SearchIndex

# ---------------------------------------------------------------------------
# Paths & constants (from env/.env via pathosearch.config; no silent fallbacks)
# ---------------------------------------------------------------------------
SETTINGS = load_settings()
ART_ROOT = SETTINGS.artifacts_dir
IMG_DIR = SETTINGS.images_dir
CAPTIONS_PATH = SETTINGS.captions_path

ENCODERS = ["clip", "quilt"]
ENCODER_LABELS = {
    "clip":  "Generic CLIP (ViT-B/32)",
    "quilt": "QuiltNet (ViT-B/32)",
}
MODES = ["Tek Model", "Karşılaştırma", "Görsel→Görsel"]

# ---------------------------------------------------------------------------
# Artifact validation
# ---------------------------------------------------------------------------
def check_artifacts(encoder_name: str) -> Tuple[bool, str]:
    art_dir = ART_ROOT / encoder_name
    required = ["embeddings.npy", "ids.json", "faiss_index.bin"]
    missing = [f for f in required if not (art_dir / f).exists()]
    if not missing:
        return True, ""
    cmd1 = f"python scripts/build_embeddings.py --encoder {encoder_name} --out_dir artifacts/{encoder_name}"
    cmd2 = f"python scripts/faiss_search.py build --art_dir artifacts/{encoder_name}"
    msg = (
        f"**{encoder_name}** için eksik dosyalar: `{', '.join(missing)}`\n\n"
        f"Çalıştır:\n```\n{cmd1}\n{cmd2}\n```"
    )
    return False, msg

# ---------------------------------------------------------------------------
# Cached resources (one entry per (name) or (art_dir) key)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Model yükleniyor…")
def load_encoder(name: str):
    from pathosearch.encoders import get_encoder
    return get_encoder(name)

@st.cache_resource(show_spinner="Domain gate hazırlanıyor…")
def load_domain_gate() -> DomainGate:
    # Always CLIP text tower — encoder-independent domain detection.
    return DomainGate(load_encoder("clip"))

@st.cache_resource(show_spinner="Index yükleniyor…")
def load_index_data(art_dir: str) -> Tuple:
    from pathosearch.faiss_search import load_index, load_embeddings, load_captions
    index, metric = load_index(art_dir)
    _, ids = load_embeddings(art_dir)
    cap_map: Dict[str, dict] = load_captions(CAPTIONS_PATH)
    return index, metric, ids, cap_map

# ---------------------------------------------------------------------------
# Search wrappers: bind Streamlit-cached resources to the shared pipeline
# ---------------------------------------------------------------------------
def get_search_index(encoder_name: str) -> SearchIndex:
    index, metric, ids, cap_map = load_index_data(str(ART_ROOT / encoder_name))
    return SearchIndex(index, metric, ids, cap_map)


def run_image_search(encoder_name: str, img: Image.Image, topk: int) -> Tuple[List[dict], float, float]:
    return pipeline.run_image_search(
        load_encoder(encoder_name), get_search_index(encoder_name), img, topk
    )


def search_pipeline(query: str, encoder_name: str, topk: int,
                    enable_rerank: bool, domain_threshold: float) -> dict:
    """Domain gate -> retrieval -> optional rerank (logic lives in pathosearch.pipeline).

    The retrieval encoder/index are loaded lazily, only for in-domain queries.
    """
    return pipeline.search_pipeline(
        query,
        load_domain_gate(),
        lambda: (load_encoder(encoder_name), get_search_index(encoder_name)),
        topk,
        enable_rerank,
        domain_threshold,
    )

# ---------------------------------------------------------------------------
# Card renderer
# ---------------------------------------------------------------------------
def draw_card(col, result: dict, is_overlap: bool = False) -> None:
    img_path = IMG_DIR / Path(result["id"]).name
    with col:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            st.warning(f"Açılamadı: {Path(result['id']).name}")
            return
        overlap_tag = "↔  " if is_overlap else ""
        st.image(img, caption=f"{overlap_tag}{result['score']:.3f}", use_container_width=True)
        if result["pathology"]:
            st.caption(f"🔬 {result['pathology']}")
        if result["caption"]:
            preview = result["caption"][:80] + ("…" if len(result["caption"]) > 80 else "")
            st.caption(preview)
            with st.expander("Tam caption"):
                st.write(result["caption"])

def display_results(results: List[dict], ncols: int = 4) -> None:
    cols = st.columns(ncols)
    for i, r in enumerate(results):
        draw_card(cols[i % ncols], r)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Histopatoloji Multimodal Arama",
    page_icon="🔬",
    layout="wide",
)
st.title("🔬 Histopatoloji Multimodal Arama")
st.caption("Generic CLIP vs QuiltNet — Alan adaptasyonu karşılaştırması | Quilt-1M + FAISS")


def require_data() -> None:
    """Stop the app with an explicit message if the corpus/artifacts are missing."""
    hint = (
        "Yolları `PATHOSEARCH_DATA_DIR`, `PATHOSEARCH_ARTIFACTS_DIR`, "
        "`PATHOSEARCH_IMAGES_DIR`, `PATHOSEARCH_CAPTIONS_PATH` ortam değişkenleriyle "
        "(veya `.env` dosyasıyla) verin; ayrıntı için `.env.example`."
    )
    try:
        require_dir(ART_ROOT, hint)
        require_dir(IMG_DIR, hint)
        require_file(CAPTIONS_PATH, hint)
    except DataNotFoundError as exc:
        st.error(f"Veri bulunamadı.\n\n```\n{exc}\n```")
        st.stop()


require_data()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("search_history", [])
st.session_state.setdefault("query_input", "")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")

    mode = st.radio("Mod", MODES, index=0)

    if mode in ("Tek Model", "Görsel→Görsel"):
        encoder_name = st.selectbox(
            "Model",
            ENCODERS,
            format_func=lambda x: ENCODER_LABELS[x],
            key="encoder_select",
        )
    else:
        encoder_name = None  # karşılaştırma: her iki encoder kullanılır

    topk = st.slider("Top-K", min_value=4, max_value=20, value=8, step=4)

    st.markdown("### Arama Ayarları")

    enable_rerank = st.checkbox(
        "Akıllı Sıralama (V5b — Weighted Jaccard)",
        value=True,
        help="Caption tabanlı ağırlıklı Jaccard re-ranking — Precision@10 0.539 → 0.609. "
             "Sadece Tek Model modunda aktif (ham model karşılaştırmasını bozmamak için).",
        disabled=(mode != "Tek Model"),
    )
    if mode != "Tek Model":
        st.caption("ℹ️ Rerank yalnız Tek Model modunda. Bu modda atıl.")

    domain_threshold = st.slider(
        "Domain Eşiği (anchor benzerliği)",
        min_value=0.65, max_value=0.88, value=0.80, step=0.01,
        help="Sorgunun histopatoloji domain'inde olup olmadığını CLIP text-anchor "
             "benzerliğiyle ölçer (text↔text, modality gap yok). "
             "Düşük = gevşek, yüksek = sıkı. Tüm modlarda aktif.",
    )

    st.divider()

    # Artifact health check — sidebar'da uyarı, ana alan etkilenir
    if mode == "Karşılaştırma":
        checks_ok = True
        for enc in ENCODERS:
            ok, err_msg = check_artifacts(enc)
            if not ok:
                st.error(err_msg)
                checks_ok = False
    else:
        checks_ok, err_msg = check_artifacts(encoder_name)
        if not checks_ok:
            st.error(err_msg)

    st.divider()

    # Son aramalar
    if st.session_state.search_history:
        st.subheader("Son Aramalar")
        for past_q in st.session_state.search_history:
            if st.button(past_q, key=f"hist_{past_q}", use_container_width=True):
                st.session_state["tq_single"] = past_q
                st.session_state["tq_compare"] = past_q
                st.rerun()

# ---------------------------------------------------------------------------
# Main: Tek Model
# ---------------------------------------------------------------------------
if mode == "Tek Model":
    query = st.text_input(
        "Metin sorgusu",
        placeholder="lung tissue with cancer cells",
        disabled=not checks_ok,
        key="tq_single",
    )

    if query.strip() and checks_ok:
        res = search_pipeline(query.strip(), encoder_name, topk, enable_rerank, domain_threshold)

        # history (record the attempt regardless of outcome)
        if query.strip() not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query.strip())
            st.session_state.search_history = st.session_state.search_history[:5]

        if res["status"] == "domain_out":
            st.warning(f"⚠️ {res['message']}")
        elif res["status"] == "empty":
            st.info("Sonuç bulunamadı.")
        else:  # ok
            rr = " · rerank: V5b" if res["reranked"] else ""
            st.caption(
                f"⏱ Arama: **{res['enc_ms'] + res['srch_ms']:.1f}ms** "
                f"(encode: {res['enc_ms']:.1f}ms, search: {res['srch_ms']:.1f}ms){rr} "
                f"| domain: {res['domain_score']:.2f} | {len(res['results'])} sonuç"
            )
            display_results(res["results"])

# ---------------------------------------------------------------------------
# Main: Karşılaştırma
# ---------------------------------------------------------------------------
elif mode == "Karşılaştırma":
    query = st.text_input(
        "Metin sorgusu",
        placeholder="invasive ductal carcinoma",
        disabled=not checks_ok,
        key="tq_compare",
    )

    if query.strip() and checks_ok:
        # Rerank disabled in compare mode — keep raw model behaviour honest.
        # Domain gate is encoder-independent (CLIP text tower) → both share the verdict.
        res_clip  = search_pipeline(query.strip(), "clip",  topk, False, domain_threshold)
        res_quilt = search_pipeline(query.strip(), "quilt", topk, False, domain_threshold)

        # history
        if query.strip() not in st.session_state.search_history:
            st.session_state.search_history.insert(0, query.strip())
            st.session_state.search_history = st.session_state.search_history[:5]

        if res_clip["status"] == "domain_out":
            st.warning(f"⚠️ {res_clip['message']}")
        else:
            clip_results  = res_clip["results"]
            quilt_results = res_quilt["results"]

            # Overlap counter
            clip_ids  = {r["id"] for r in clip_results}
            quilt_ids = {r["id"] for r in quilt_results}
            overlap   = clip_ids & quilt_ids
            overlap_count = len(overlap)
            overlap_pct   = (overlap_count / topk) * 100
            clip_only     = len(clip_ids - quilt_ids)
            quilt_only    = len(quilt_ids - clip_ids)

            st.info(
                f"🟢 Top-{topk}'te **{overlap_count}** ortak görsel (%{overlap_pct:.1f}) "
                f"| CLIP'e özel: **{clip_only}**, QuiltNet'e özel: **{quilt_only}** "
                f"| domain: {res_clip['domain_score']:.2f}"
            )

            # Latency row
            lat_l, lat_r = st.columns(2)
            with lat_l:
                st.caption(f"⏱ CLIP — encode: {res_clip['enc_ms']:.1f}ms, search: {res_clip['srch_ms']:.1f}ms")
            with lat_r:
                st.caption(f"⏱ QuiltNet — encode: {res_quilt['enc_ms']:.1f}ms, search: {res_quilt['srch_ms']:.1f}ms")

            # Column headers
            hdr_l, hdr_r = st.columns(2)
            with hdr_l:
                st.subheader("Generic CLIP")
            with hdr_r:
                st.subheader("QuiltNet")

            # Results: rank-aligned rows
            for i in range(max(len(clip_results), len(quilt_results))):
                row_l, row_r = st.columns(2)
                if i < len(clip_results):
                    r = clip_results[i]
                    draw_card(row_l, r, is_overlap=(r["id"] in overlap))
                if i < len(quilt_results):
                    r = quilt_results[i]
                    draw_card(row_r, r, is_overlap=(r["id"] in overlap))

# ---------------------------------------------------------------------------
# Main: Görsel→Görsel
# ---------------------------------------------------------------------------
elif mode == "Görsel→Görsel":
    uploaded = st.file_uploader(
        "Sorgu görseli yükle (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        disabled=not checks_ok,
    )

    if not checks_ok:
        st.error("Seçili model için artifact eksik. Sidebar'daki komutları çalıştırın.")

    if uploaded and checks_ok:
        query_img = Image.open(uploaded).convert("RGB")

        prev_col, info_col = st.columns([1, 3])
        with prev_col:
            st.image(query_img, caption="Sorgu görseli", use_container_width=True)
        with info_col:
            st.markdown(f"**Boyut:** {query_img.size[0]}×{query_img.size[1]} px")
            st.markdown(f"**Model:** {ENCODER_LABELS[encoder_name]}")

        results, enc_ms, srch_ms = run_image_search(encoder_name, query_img, topk)

        st.caption(
            f"⏱ Arama: **{enc_ms + srch_ms:.1f}ms** "
            f"(encode: {enc_ms:.1f}ms, search: {srch_ms:.1f}ms)"
        )

        st.subheader("Benzer Görseller")
        cols = st.columns(4)
        for i, r in enumerate(results):
            draw_card(cols[i % 4], r)
