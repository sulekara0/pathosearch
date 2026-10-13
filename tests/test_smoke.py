import importlib

import pytest

MODULES = [
    "pathosearch",
    "pathosearch.encoders",
    "pathosearch.faiss_search",
    "pathosearch.reranker",
    "pathosearch.cross_encoder_reranker",
    "pathosearch.domain_gate",
    "pathosearch.thresholds",
    "pathosearch.data.prepare_quilt",
    "pathosearch.data.build_embeddings",
    "pathosearch.eval.compare_models",
    "pathosearch.eval.make_queries",
    "pathosearch.eval.make_queries_v2",
    "pathosearch.eval.make_queries_ensemble",
    "pathosearch.eval.make_comparison_report",
    "pathosearch.eval.visualize_search",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
