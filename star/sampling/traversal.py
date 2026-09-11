from __future__ import annotations

import os
import random
from collections import defaultdict, deque


def grid_adjacency(height: int, width: int) -> dict[int, list[int]]:
    if height <= 0 or width <= 0:
        raise ValueError("Grid dimensions must be positive.")
    adjacency = {}
    for row in range(height):
        for column in range(width):
            vertex = row * width + column
            neighbors = []
            if row > 0:
                neighbors.append((row - 1) * width + column)
            if row < height - 1:
                neighbors.append((row + 1) * width + column)
            if column > 0:
                neighbors.append(row * width + column - 1)
            if column < width - 1:
                neighbors.append(row * width + column + 1)
            adjacency[vertex] = neighbors
    return adjacency


def corner_indices(height: int, width: int) -> list[int]:
    if height <= 0 or width <= 0:
        raise ValueError("Grid dimensions must be positive.")
    return list(
        dict.fromkeys(
            (0, width - 1, (height - 1) * width, height * width - 1)
        )
    )


def is_connected(adjacency: dict[int, list[int]], nodes: set[int]) -> bool:
    if not nodes:
        return True
    start = min(nodes)
    seen = {start}
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(nodes)


def induced_subgraph(
    adjacency: dict[int, list[int]], nodes: set[int]
) -> dict[int, list[int]]:
    selected = set(nodes)
    return {
        vertex: [neighbor for neighbor in adjacency[vertex] if neighbor in selected]
        for vertex in sorted(selected)
    }


def bfs_shortest_dist_adj(
    adjacency: dict[int, list[int]], root: int
) -> dict[int, int]:
    if root not in adjacency:
        raise ValueError(f"Root {root} is not present in the graph.")
    distances = {root: 0}
    queue = deque([root])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex]:
            if neighbor not in distances:
                distances[neighbor] = distances[vertex] + 1
                queue.append(neighbor)
    return distances


def random_spanning_tree(
    graph: dict[int, list[int]],
    root: int,
    *,
    rng: random.Random | None = None,
) -> dict[int, list[int]]:
    rng = random if rng is None else rng

    if not isinstance(graph, dict):
        raise TypeError(f"graph must be a dict, got {type(graph).__name__}.")
    if root not in graph:
        raise ValueError(f"Root {root} is not present in the graph.")

    reachable = {root}
    queue = deque([root])
    while queue:
        vertex = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor not in graph:
                raise ValueError(
                    f"Neighbor {neighbor} is not present in the graph."
                )
            if vertex not in graph[neighbor]:
                raise ValueError("Wilson sampling requires an undirected graph.")
            if neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    if len(reachable) != len(graph):
        raise ValueError("Cannot sample a spanning tree from a disconnected graph.")

    parent = {root: None}
    visited = {root}
    for start in graph:
        vertex = start
        while vertex not in visited:
            neighbors = graph[vertex]
            if not neighbors:
                raise ValueError(
                    "Cannot sample a spanning tree from a disconnected graph."
                )
            parent[vertex] = rng.choice(neighbors)
            vertex = parent[vertex]

        vertex = start
        while vertex not in visited:
            visited.add(vertex)
            vertex = parent[vertex]

    tree = {vertex: [] for vertex in graph}
    for child, parent_vertex in parent.items():
        if parent_vertex is not None:
            tree[child].append(parent_vertex)
            tree[parent_vertex].append(child)
    return tree


def bfs_tree(
    tree: dict[int, list[int]], root: int
) -> tuple[list[int], dict[int, int]]:
    if root not in tree:
        raise ValueError(f"Root {root} is not present in the tree.")
    order = []
    depths = {}
    parents = {root: None}
    queue = deque([(root, 0)])
    while queue:
        vertex, depth = queue.popleft()
        order.append(vertex)
        depths[vertex] = depth
        for neighbor in sorted(tree[vertex]):
            if neighbor not in parents:
                parents[neighbor] = vertex
                queue.append((neighbor, depth + 1))
    return order, depths


def sample_bfs_spanning_tree_order(
    height: int,
    width: int,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    rng = random if rng is None else rng
    adjacency = grid_adjacency(height, width)
    corners = corner_indices(height, width)
    root = rng.choice(corners)
    tree = random_spanning_tree(adjacency, root, rng=rng)
    order, _ = bfs_tree(tree, root)
    if len(order) != height * width:
        raise RuntimeError("The sampled traversal does not cover the token grid.")
    return order


def _sample_seeded_bfs_spanning_tree_order(
    task: tuple[int, int, int, int, int],
) -> tuple[int, list[int]]:
    slot, sample_id, height, width, seed = task
    try:
        order = sample_bfs_spanning_tree_order(
            height,
            width,
            rng=random.Random(seed),
        )
    except Exception as exc:
        raise RuntimeError(
            "UST sampling failed for "
            f"batch slot {slot} (sample_id={sample_id})."
        ) from exc
    return slot, order


def _order_worker_process_id() -> int:
    return os.getpid()


def connected_components(
    adjacency: dict[int, list[int]], nodes: set[int]
) -> list[set[int]]:
    remaining = set(nodes)
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency.get(vertex, []):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def observed_boundary(
    full_adjacency: dict[int, list[int]],
    mask_component: set[int],
    observed: set[int],
) -> set[int]:
    return {
        neighbor
        for vertex in mask_component
        for neighbor in full_adjacency[vertex]
        if neighbor in observed
    }


def _optimal_roots(
    height: int,
    width: int,
    mask: set[int],
) -> list[int]:
    mask = set(mask)
    full_adjacency = grid_adjacency(height, width)
    vertices = set(full_adjacency)
    if not mask:
        raise ValueError("Mask must contain at least one vertex.")
    if not mask < vertices:
        raise ValueError("Mask must be a proper subset of the token grid.")
    observed = vertices - mask

    candidate_roots = [
        corner for corner in corner_indices(height, width) if corner in observed
    ]
    if not candidate_roots:
        raise ValueError("All grid corners lie inside the mask.")

    mask_adjacency = induced_subgraph(full_adjacency, mask)
    components = connected_components(mask_adjacency, mask)
    boundaries = [
        observed_boundary(full_adjacency, component, observed)
        for component in components
    ]
    if any(not boundary for boundary in boundaries):
        raise ValueError("A mask component has no observed-side boundary.")

    observed_adjacency = induced_subgraph(full_adjacency, observed)
    if not is_connected(observed_adjacency, observed):
        raise ValueError("The observed induced graph must be connected.")

    scores = {}
    for root in candidate_roots:
        distances = bfs_shortest_dist_adj(observed_adjacency, root)
        component_means = [
            sum(distances[vertex] for vertex in boundary) / len(boundary)
            for boundary in boundaries
        ]
        scores[root] = min(component_means)

    best_score = max(scores.values())
    return [root for root in candidate_roots if scores[root] == best_score]


def _merge_trees(
    first: dict[int, list[int]], second: dict[int, list[int]]
) -> dict[int, list[int]]:
    merged = defaultdict(set)
    for tree in (first, second):
        for vertex, neighbors in tree.items():
            merged[vertex].update(neighbors)
            for neighbor in neighbors:
                merged[neighbor].add(vertex)
    return {vertex: sorted(neighbors) for vertex, neighbors in merged.items()}


def _sample_postfix_order_once(
    height: int,
    width: int,
    mask: set[int],
    root: int,
    *,
    rng,
) -> list[int]:
    full_adjacency = grid_adjacency(height, width)
    vertices = set(full_adjacency)
    mask = set(mask)
    if not mask:
        raise ValueError("Mask must contain at least one vertex.")
    if not mask < vertices:
        raise ValueError("Mask must be a proper subset of the token grid.")
    observed = vertices - mask
    if root not in observed or root not in corner_indices(height, width):
        raise ValueError("Completion root must be an observed grid corner.")

    observed_adjacency = induced_subgraph(full_adjacency, observed)
    if not is_connected(observed_adjacency, observed):
        raise ValueError("The observed induced graph must be connected.")

    mask_adjacency = induced_subgraph(full_adjacency, mask)
    components = connected_components(mask_adjacency, mask)
    boundaries = [
        observed_boundary(full_adjacency, component, observed)
        for component in components
    ]
    if any(not boundary for boundary in boundaries):
        raise ValueError("A mask component has no observed-side boundary.")

    observed_tree = random_spanning_tree(
        observed_adjacency,
        root,
        rng=rng,
    )
    _, observed_depths = bfs_tree(observed_tree, root)
    max_observed_depth = max(observed_depths.values())

    # Remark:
    # We observed extremely rare corner mask types where the anchor-based depth increment cannot be applied at small grid sizes. 
    # We added ad hoc logic for checking tie-breaking rules specifically for these instances.
    attachments = []
    used_tie_fallback = False 
    for component, boundary in zip(components, boundaries):
        candidates = sorted(
            vertex
            for vertex in boundary
            if observed_depths[vertex] == max_observed_depth
        )
        if not candidates and len(components) > 1:
            candidates = sorted(
                vertex
                for vertex in boundary
                if observed_depths[vertex] == max_observed_depth - 1
            )
            used_tie_fallback = True
        if not candidates:
            raise RuntimeError(
                "The tree proposal does not reach every mask boundary."
            )
        anchor = rng.choice(candidates)
        entry = rng.choice(
            [
                neighbor
                for neighbor in full_adjacency[anchor]
                if neighbor in component
            ]
        )
        attachments.append((component, anchor, entry))

    tree = observed_tree
    for component, anchor, entry in attachments:
        component_nodes = component | {anchor}
        component_adjacency = induced_subgraph(full_adjacency, component_nodes)
        for vertex in component:
            component_adjacency[vertex] = [
                neighbor
                for neighbor in component_adjacency[vertex]
                if neighbor != anchor
            ]
        component_adjacency[anchor] = [entry]
        component_adjacency[entry] = sorted(
            set(component_adjacency[entry]) | {anchor}
        )
        component_tree = random_spanning_tree(
            component_adjacency,
            anchor,
            rng=rng,
        )
        tree = _merge_trees(tree, component_tree)

    order, depths = bfs_tree(tree, root)
    if used_tie_fallback:
        order.sort(key=lambda vertex: (depths[vertex], vertex in mask))
        split = len(observed)
        if set(order[:split]) != observed or set(order[split:]) != mask:
            raise RuntimeError("The tree proposal does not place the mask last.")
    return order


def sample_postfix_bfs_order(
    height: int,
    width: int,
    mask: set[int],
    *,
    max_attempts: int = 1000,
    rng: random.Random | None = None,
) -> list[int]:
    rng = random if rng is None else rng
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    mask = set(mask)
    best_roots = _optimal_roots(height, width, mask)
    first_root = (
        best_roots[0] if len(best_roots) == 1 else rng.choice(best_roots)
    )

    last_error = None
    for attempt in range(1, max_attempts + 1):
        candidate_root = (
            first_root
            if attempt == 1 or len(best_roots) == 1
            else rng.choice(best_roots)
        )
        try:
            order = _sample_postfix_order_once(
                height,
                width,
                mask,
                candidate_root,
                rng=rng,
            )
        except RuntimeError as exc:
            last_error = exc
            continue

        num_vertices = height * width
        expected_vertices = set(range(num_vertices))
        split = num_vertices - len(mask)
        if len(order) != num_vertices or set(order) != expected_vertices:
            raise RuntimeError("Completion traversal is not a valid permutation.")
        if set(order[split:]) != mask or set(order[:split]) & mask:
            raise RuntimeError(
                "Completion traversal does not place its mask in the postfix."
            )

        return order

    raise ValueError(
        "Could not construct a postfix traversal. The requested layout is "
        "outside the supported single-pass completion domain."
    ) from last_error


def _sample_seeded_postfix_bfs_order(
    task: tuple[int, int, int, int, tuple[int, ...], int, int],
) -> tuple[int, list[int]]:
    slot, sample_id, height, width, mask, max_attempts, seed = task
    try:
        order = sample_postfix_bfs_order(
            height,
            width,
            set(mask),
            max_attempts=max_attempts,
            rng=random.Random(seed),
        )
    except Exception as exc:
        raise RuntimeError(
            "Completion-order sampling failed for "
            f"batch slot {slot} (sample_id={sample_id}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return slot, order
