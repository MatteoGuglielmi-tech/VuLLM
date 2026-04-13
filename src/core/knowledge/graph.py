from __future__ import annotations

from collections import deque
from typing import Literal

from .typedefs import LinkType


class _Vertex:
    def __init__(
        self,
        cwe_id: int,
        neighbours: dict[LinkType, set[_Vertex]] | None = None,
    ):
        self.cwe_id = cwe_id
        self.neighbours = neighbours or {}


class Graph:
    _inverse: dict[LinkType, LinkType] = {
        LinkType.CHILDOF: LinkType.PARENTOF,
        LinkType.PARENTOF: LinkType.CHILDOF,
        LinkType.CANPRECEDE: LinkType.CANFOLLOW,
        LinkType.CANFOLLOW: LinkType.CANPRECEDE,
        LinkType.PEEROF: LinkType.PEEROF,
    }

    def __init__(self):
        self._vertices: dict[int, _Vertex] = {}

    def add_vertex(self, cwe_id: int):
        if cwe_id not in self._vertices:
            self._vertices[cwe_id] = _Vertex(cwe_id=cwe_id, neighbours=None)

    def add_edge(
        self, source_node_id: int, relation_type: LinkType, target_node_id: int
    ):
        if source_node_id in self._vertices and target_node_id in self._vertices:
            self._vertices[source_node_id].neighbours.setdefault(
                relation_type, set()
            ).add(self._vertices[target_node_id])

            self._vertices[target_node_id].neighbours.setdefault(
                self._inverse[relation_type], set()
            ).add(self._vertices[source_node_id])

    def _bfs(self, start_id: int, relation_type: LinkType) -> set[int]:
        visited: set[int] = set()
        _bfs_queue: deque[int] = deque()
        _bfs_queue.append(start_id)

        while len(_bfs_queue) >= 1:
            current_id: int = _bfs_queue.popleft()  # get current node
            visited.add(current_id)  # update visited
            node_neighbours: set[int] = set(  # retrieve neighbours
                [
                    v.cwe_id
                    for v in self._vertices[current_id].neighbours.get(
                        relation_type, set()
                    )
                ]
            )
            _bfs_queue.extend(
                node_neighbours - visited
            )  # append non-visited neighbours to rear

        # delete starting id
        visited.remove(start_id)

        return visited

    def _dfs(self, start_id: int, relation_type: LinkType) -> set[int]:
        visited: set[int] = set()

        def _recurse(node_id: int):
            visited.add(node_id)

            node_neighbours: set[int] = set(  # retrieve neighbours
                [
                    v.cwe_id
                    for v in self._vertices[node_id].neighbours.get(
                        relation_type, set()
                    )
                ]
            )

            for neighbour in node_neighbours:
                if neighbour not in visited:
                    _recurse(node_id=neighbour)

        _recurse(start_id)
        visited.remove(start_id)

        return visited

    def neighbours(self, node_id: int, relation_type: LinkType) -> set[int]:
        """Return direct (single-hop) neighbours of *node_id* along *relation_type*."""
        if node_id not in self._vertices:
            raise ValueError("Graph doesn't contain specified vertex")
        return {
            v.cwe_id
            for v in self._vertices[node_id].neighbours.get(relation_type, set())
        }

    def search(
        self, backend: Literal["dfs", "bfs"], start_id: int, relation_type: LinkType
    ) -> set[int]:
        if len(self._vertices) == 0:
            raise ValueError("Graph has not vertices")
        if start_id not in self._vertices:
            raise ValueError("Graph doesn't contain specified vertex")

        if backend == "dfs":
            return self._dfs(start_id=start_id, relation_type=relation_type)
        else:
            return self._bfs(start_id=start_id, relation_type=relation_type)
