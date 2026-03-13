"""Graph-based dependency memory system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class DependencyGraphStore:
    """Persistent graph-based dependency store."""

    def __init__(self, workspace_dir: Path) -> None:
        self.graph_path = workspace_dir / "dependency_graph.json"
        self._graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        if not self.graph_path.exists():
            return
        try:
            data = json.loads(self.graph_path.read_text(encoding="utf-8"))
            for node, deps in data.items():
                self._graph.add_node(node)
                for dep in deps:
                    self._graph.add_edge(node, dep)
        except Exception as e:
            logger.warning(f"Failed to load dependency graph: {e}")

    def save(self) -> None:
        data: dict[str, list[str]] = {}
        for node in self._graph.nodes():
            data[node] = list(self._graph.successors(node))
        self.graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_dependency(self, source: str, target: str) -> None:
        self._graph.add_edge(source, target)

    def get_dependencies(self, file_path: str) -> list[str]:
        """Get direct dependencies of a file."""
        if file_path not in self._graph:
            return []
        return list(self._graph.successors(file_path))

    def get_dependents(self, file_path: str) -> list[str]:
        """Get files that depend on this file."""
        if file_path not in self._graph:
            return []
        return list(self._graph.predecessors(file_path))

    def get_transitive_dependencies(self, file_path: str) -> set[str]:
        """Get all transitive dependencies."""
        if file_path not in self._graph:
            return set()
        return nx.descendants(self._graph, file_path)

    def detect_cycles(self) -> list[list[str]]:
        """Detect dependency cycles."""
        try:
            return list(nx.simple_cycles(self._graph))
        except Exception:
            return []

    def get_layers(self) -> dict[str, int]:
        """Compute layer numbers. Leaf nodes (no outgoing deps) are layer 0."""
        layers: dict[str, int] = {}
        # Reverse topological order: process leaves first
        for node in reversed(list(nx.topological_sort(self._graph))):
            successors = list(self._graph.successors(node))
            if not successors:
                layers[node] = 0
            else:
                layers[node] = max(layers.get(s, 0) for s in successors) + 1
        return layers

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    # ── Impact analysis for modification workflows ───────────────────

    def get_impact_analysis(self, file_path: str) -> dict[str, list[str]]:
        """Analyze the impact of modifying a file.

        Returns a dict with:
        - direct_dependents: files that directly import from this file
        - transitive_dependents: all files transitively affected
        - direct_dependencies: files this file directly imports
        - test_files: test files that may need updating (heuristic)
        """
        result: dict[str, list[str]] = {
            "direct_dependents": self.get_dependents(file_path),
            "transitive_dependents": [],
            "direct_dependencies": self.get_dependencies(file_path),
            "test_files": [],
        }

        # Transitive dependents — all files that transitively import from this file
        if file_path in self._graph:
            result["transitive_dependents"] = sorted(
                nx.ancestors(self._graph, file_path) - {file_path}
            )

        # Heuristic: find test files (files with "test" in the name that are
        # direct or transitive dependents, or that share a module name)
        target_stem = Path(file_path).stem
        all_affected = set(result["direct_dependents"]) | set(result["transitive_dependents"])
        for node in self._graph.nodes():
            node_lower = node.lower()
            if "test" in node_lower and (
                node in all_affected
                or target_stem in node_lower
            ):
                result["test_files"].append(node)

        return result

    def get_modification_order(self, files: list[str]) -> list[str]:
        """Given a set of files to modify, return them in safe modification order.

        Files with no dependencies on other files in the set come first.
        This ensures that when a file is modified, all files it depends on
        have already been modified.
        """
        # Build a subgraph of only the files we're modifying
        subgraph = self._graph.subgraph(
            [f for f in files if f in self._graph]
        )
        try:
            ordered = list(nx.topological_sort(subgraph))
            # Add files not in the graph at the end
            remaining = [f for f in files if f not in self._graph]
            return ordered + remaining
        except nx.NetworkXUnfeasible:
            logger.warning("Cycle detected among modification targets — returning original order")
            return files
