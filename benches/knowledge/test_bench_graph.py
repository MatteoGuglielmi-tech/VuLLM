"""Benchmarks for BFS vs DFS graph traversal on the real CWE hierarchy."""

import pytest

from src.core.knowledge.cwe_knowledge_base import CWEKnowledgeBase
from src.core.knowledge.graph import Graph, LinkType


@pytest.fixture(scope="module")
def cwe_graph() -> Graph:
    """Build the full CWE hierarchy graph once for all benchmarks."""
    kb = CWEKnowledgeBase()
    g = Graph()
    for entry in kb.get_all():
        g.add_vertex(entry.cwe_id)
    for entry in kb.get_all():
        for rel in entry.relationships:
            if rel.nature in (LinkType.CHILDOF, LinkType.CANPRECEDE, LinkType.PEEROF):
                g.add_edge(entry.cwe_id, LinkType(rel.nature), rel.cwe_id)
    return g


# ---------------------------------------------------------------------------
# Verify BFS == DFS on real data before benchmarking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cwe_id", [787, 119, 416, 682, 20])
def test_bfs_dfs_equivalence_real(cwe_graph: Graph, cwe_id: int):
    for link in (LinkType.CHILDOF, LinkType.PARENTOF):
        assert cwe_graph.search("bfs", cwe_id, link) == cwe_graph.search(
            "dfs", cwe_id, link
        )


# ---------------------------------------------------------------------------
# Ancestors (narrow/deep): CWE-787 → 119 → … → root
# ---------------------------------------------------------------------------


def test_bench_bfs_ancestors_787(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "bfs", 787, LinkType.CHILDOF)


def test_bench_dfs_ancestors_787(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "dfs", 787, LinkType.CHILDOF)


# ---------------------------------------------------------------------------
# Descendants (wide/broad): CWE-119 has many children
# ---------------------------------------------------------------------------


def test_bench_bfs_descendants_119(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "bfs", 119, LinkType.PARENTOF)


def test_bench_dfs_descendants_119(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "dfs", 119, LinkType.PARENTOF)


# ---------------------------------------------------------------------------
# Descendants from a high-level pillar: CWE-682 (broad tree)
# ---------------------------------------------------------------------------


def test_bench_bfs_descendants_682(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "bfs", 682, LinkType.PARENTOF)


def test_bench_dfs_descendants_682(benchmark, cwe_graph: Graph):
    benchmark(cwe_graph.search, "dfs", 682, LinkType.PARENTOF)
