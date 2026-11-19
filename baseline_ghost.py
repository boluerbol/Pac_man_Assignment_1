"""baseline_ghost.py — the house ghost. A plain chaser: when it sees Pac-Man it takes the
shortest route to it; when it does not, it goes to where it last saw Pac-Man, and if it
has never seen it, to the nearest food (Pac-Man will be near food). Your Pac-Man should
outrun it; your ghost should do better than it."""


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


def first_step(path):
    (r0, c0), (r1, c1) = path[0], path[1]
    return {(-1, 0): "N", (1, 0): "S", (0, 1): "E", (0, -1): "W"}[(r1 - r0, c1 - c0)]


def ghost_move(state):
    me = state.ghost
    legal = state.legal_moves(me)
    if state.pacman is not None:
        target = state.pacman
    elif state.pacman_last_seen is not None and state.pacman_last_seen[0] != me:
        target = state.pacman_last_seen[0]
    elif state.food:
        target = min(state.food, key=lambda f: state.manhattan(me, f))
    else:
        return next(iter(legal))
    path = fastest_path(state, me, target)
    if path is None or len(path) < 2:
        return next(iter(legal))
    return first_step(path)


def pacman_move(state):
    # not used in the league; lets this file play either side in play.py
    me = state.pacman
    if not state.food:
        return "STOP"
    target = min(state.food, key=lambda f: state.manhattan(me, f))
    path = fastest_path(state, me, target)
    return first_step(path) if path and len(path) > 1 else "STOP"
