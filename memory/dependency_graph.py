"""Graph-based dependency memory system."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
