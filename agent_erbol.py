"""
agent_erbol.py — Pac-Man League agent (AI: Principles and Techniques, Assignment 1).

Design in brief (reword for report_erbol.pdf):
- astar() is a generic A* over caller-supplied neighbors/cost/heuristic; used by both
  Pac-Man and the ghost.
- One all-pairs BFS distance table (cost-1 edges) is cached per board (keyed on
  state.walls), precomputed in new_game(). It is the heuristic inside astar(), the
  distance metric inside evaluate(), and the basis for target selection. Using real
  maze distances instead of Manhattan is what the assignment tips ask for.
- Pac-Man out of sight: commit to a target food (nearest by maze distance), don't
  re-pick every tick unless it is eaten. Step cost = 1 + a small penalty for stepping
  near the ghost's last seen cell (cost shaping, never walls). A* to target, take step.
- Pac-Man in sight: lookahead() — depth-limited minimax/expectimax tree; one round =
  one Pac-Man move + one ghost reply; leaves scored by evaluate(). The ghost's skip
  tick is handled: when it does not move, the only reply is "stay" (pass None).
- Ghost: A* straight to Pac-Man when visible; else A* to pacman_last_seen; once there
  with Pac-Man still hidden, aim at the nearest remaining food (Pac-Man has to eat).
"""
from collections import deque
import heapq
import itertools

OPPONENT_MODEL = "minimax"   # or "expectimax" — justify in the report
LOOKAHEAD_DEPTH = 2          # rounds; 1 round = my move + ghost reply. Timed under 100 ms.

_pacman_target = None
_ghost_target = None
_distance_cache = {}


def new_game(state):
    """Engine hook: called once per game. Reset memory and precompute distance table."""
    global _pacman_target, _ghost_target
    _pacman_target = None
    _ghost_target = None
    _get_distance_map(state)


# ------------------------------------------------------------------ A* ----

def astar(start, goal, neighbors, cost, heuristic):
    """Cheapest path [start, ..., goal], or None if unreachable."""
    if start == goal:
        return [start]

    counter = itertools.count()
    frontier = [(heuristic(start), next(counter), start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for nbr in neighbors(current):
            if nbr in closed:
                continue
            tentative_g = g_score[current] + cost(current, nbr)
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                heapq.heappush(frontier, (tentative_g + heuristic(nbr), next(counter), nbr))
    return None


# ---------------------------------- maze distance table (BFS) ----

def _get_distance_map(state):
    key = state.walls
    table = _distance_cache.get(key)
    if table is None:
        table = _compute_all_pairs_bfs(state)
        _distance_cache[key] = table
    return table


def _compute_all_pairs_bfs(state):
    """dist[a][b] = shortest maze steps from a to b. Board never changes -> cache once."""
    dist = {}
    cells = [(r, c) for r in range(state.rows) for c in range(state.cols)
             if (r, c) not in state.walls]
    for src in cells:
        d = {src: 0}
        q = deque([src])
        while q:
            cur = q.popleft()
            for nbr in state.legal_moves(cur).values():
                if nbr not in d:
                    d[nbr] = d[cur] + 1
                    q.append(nbr)
        dist[src] = d
    return dist


def _maze_dist(state, a, b):
    row = _get_distance_map(state).get(a)
    if row is not None and b in row:
        return row[b]
    return state.manhattan(a, b)


# --------------------------------------------------------- evaluate ----

def evaluate(state, pacman, ghost, food):
    """Larger = better for Pac-Man. Real maze distances, per assignment tips."""
    if ghost is not None and pacman == ghost:
        return -10_000.0

    score = -len(food) * 10.0        # less food left = better

    if food:
        nearest = min(_maze_dist(state, pacman, f) for f in food)
        score -= 2.0 * nearest

    if ghost is not None:
        g = _maze_dist(state, pacman, ghost)
        if g <= 2:
            score -= (3 - g) * 200.0
        elif g <= 5:
            score -= (6 - g) * 20.0
    return score


# -------------------------------------------------------- lookahead ----

def _ghost_reply_options(state, ghost_pos):
    if not state.ghost_can_move:
        return [None]                # ghost sits out -> only reply is 'stay'
    if ghost_pos is None:
        return [None]
    moves = list(state.legal_moves(ghost_pos).keys())
    return moves if moves else [None]


def lookahead(state, depth):
    ghost_pos = state.ghost if state.ghost is not None else (
        state.ghost_last_seen[0] if state.ghost_last_seen else None)

    best_move, best_val = "STOP", float("-inf")
    for move in list(state.legal_moves(state.pacman).keys()) + ["STOP"]:
        val = _expand(state, move, ghost_pos, depth)
        if val > best_val:
            best_val, best_move = val, move
    return best_move


def _expand(state, pac_move, ghost_pos, depth):
    vals = []
    for reply in _ghost_reply_options(state, ghost_pos):
        nxt = state.next(pac_move, reply)
        new_g = nxt.ghost if nxt.ghost is not None else ghost_pos

        if nxt.done:
            if nxt.caught:
                vals.append(-10_000.0)
            elif not nxt.food:
                vals.append(10_000.0)
            else:
                vals.append(evaluate(nxt, nxt.pacman, new_g, nxt.food))
        elif depth <= 1:
            vals.append(evaluate(nxt, nxt.pacman, new_g, nxt.food))
        else:
            vals.append(_best_own_move(nxt, new_g, depth - 1))

    if OPPONENT_MODEL == "minimax":
        return min(vals)
    return sum(vals) / len(vals)


def _best_own_move(state, ghost_pos, depth):
    best = float("-inf")
    for move in list(state.legal_moves(state.pacman).keys()) + ["STOP"]:
        best = max(best, _expand(state, move, ghost_pos, depth))
    return best


# ----------------------------------------------------- pacman_move ----

def _choose_target(state):
    global _pacman_target
    if not state.food:
        return None
    if _pacman_target is not None and _pacman_target in state.food:
        return _pacman_target
    _pacman_target = min(state.food, key=lambda f: (_maze_dist(state, state.pacman, f), f))
    return _pacman_target


def _pacman_step_cost(state, a, b):
    cost = 1
    if state.ghost_last_seen is not None:
        d = state.manhattan(b, state.ghost_last_seen[0])
        if d <= 1:
            cost += 8
        elif d <= 3:
            cost += 3
    return cost


def pacman_move(state):
    global _pacman_target

    if state.ghost is not None:                    # ghost in sight -> look ahead
        return lookahead(state, LOOKAHEAD_DEPTH)

    target = _choose_target(state)
    if target is None:
        return "STOP"

    path = astar(state.pacman, target,
                 neighbors=lambda c: list(state.legal_moves(c).values()),
                 cost=lambda a, b: _pacman_step_cost(state, a, b),
                 heuristic=lambda c: _maze_dist(state, c, target))
    if not path or len(path) < 2:
        _pacman_target = None
        return "STOP"

    nxt = path[1]
    for move, cell in state.legal_moves(state.pacman).items():
        if cell == nxt:
            return move
    return "STOP"


# ------------------------------------------------------ ghost_move ----

def _choose_ghost_target(state):
    global _ghost_target
    if state.pacman_last_seen is None:
        return None
    last_cell = state.pacman_last_seen[0]
    if state.ghost == last_cell and state.food:
        _ghost_target = min(state.food, key=lambda f: (_maze_dist(state, last_cell, f), f))
        return _ghost_target
    return last_cell


def ghost_move(state):
    target = state.pacman if state.pacman is not None else _choose_ghost_target(state)

    if target is not None:
        path = astar(state.ghost, target,
                     neighbors=lambda c: list(state.legal_moves(c).values()),
                     cost=lambda a, b: 1,
                     heuristic=lambda c: _maze_dist(state, c, target))
        if path and len(path) >= 2:
            nxt = path[1]
            for move, cell in state.legal_moves(state.ghost).items():
                if cell == nxt:
                    return move

    moves = list(state.legal_moves(state.ghost).keys())
    return moves[0] if moves else "N"