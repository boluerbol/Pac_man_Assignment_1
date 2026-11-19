"""baseline_pacman.py — the house Pac-Man. Blind: goes to the nearest food by the
shortest route and ignores the ghost completely. Your ghost should catch it."""


def fastest_path(state, start, goal):
    """Shortest path on the maze (breadth-first search: every step costs 1)."""
    from collections import deque
    came_from = {start: None}
    q = deque([start])
    while q:
        cell = q.popleft()
        if cell == goal:
            path = [cell]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return path[::-1]
        for nxt in state.legal_moves(cell).values():
            if nxt not in came_from:
                came_from[nxt] = cell
                q.append(nxt)
    return None


def first_step(state, path):
    (r0, c0), (r1, c1) = path[0], path[1]
    return {(-1, 0): "N", (1, 0): "S", (0, 1): "E", (0, -1): "W"}[(r1 - r0, c1 - c0)]


def pacman_move(state):
    me = state.pacman
    if not state.food:
        return "STOP"
    # nearest food by Manhattan first, then the real path to it
    target = min(state.food, key=lambda f: state.manhattan(me, f))
    path = fastest_path(state, me, target)
    if path is None or len(path) < 2:
        return "STOP"
    return first_step(state, path)


def ghost_move(state):
    # not used in the league; lets this file play either side in play.py
    me = state.ghost
    target = state.pacman or (state.pacman_last_seen[0] if state.pacman_last_seen else None)
    if target is None or target == me:
        return next(iter(state.legal_moves(me)))
    path = fastest_path(state, me, target)
    if path is None or len(path) < 2:
        return next(iter(state.legal_moves(me)))
    return first_step(state, path)
