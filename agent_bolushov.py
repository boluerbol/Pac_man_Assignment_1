"""Pac-Man League agent (AI: Principles and Techniques, Assignment 1).

DESIGN, IN BRIEF -- reword this for report_<surname>.pdf, don't paste it:

- astar() is a plain, generic A* over caller-supplied neighbors/cost/
  heuristic callables. The same function drives the ghost's chase and
  Pac-Man's route-finding; neither knows how the maze is stored.
- One all-pairs BFS distance table (over cost-1 edges) is cached per board
  (keyed by state.walls) and reused as the A* heuristic and inside
  evaluate(). new_game() precomputes it so it never eats into a move's
  100ms budget.
- Pac-Man, ghost out of sight: commit to the nearest unclaimed food (by
  maze distance, not Manhattan) and A* to it. Cells close to the ghost's
  last seen position cost more to step into, but are never walled off.
- Pac-Man, ghost in sight: lookahead() -- a depth-limited tree, one round =
  one Pac-Man move + one ghost reply, leaves scored by evaluate().
  OPPONENT_MODEL picks min (minimax) or mean (expectimax) at each
  ghost-reply node.
- Ghost: A* straight to Pac-Man when visible; otherwise A* to its last
  seen cell, and once that cell is reached with Pac-Man still hidden, a
  guess at the food Pac-Man still needs from there.

FLAGGED ASSUMPTION: on a tick the ghost sits out (state.ghost_can_move is
False), _ghost_reply_options() below calls state.next(pac_move, None). If
your pacman_engine.py expects something else there (e.g. 'STOP', or the
ghost's own cell), change that one sentinel -- it's the only place this
matters. This was drafted from the assignment PDF only, without
pacman_engine.py itself, so check this against the real file first.
"""

from collections import deque
import heapq
import itertools

OPPONENT_MODEL = "minimax"  # or "expectimax" -- justify your pick in the report.
LOOKAHEAD_DEPTH = 2         # rounds of (my move, ghost reply). Re-time before
                            # raising this -- see check_submission.py.

_pacman_target = None
_ghost_target = None
_distance_cache = {}


def new_game(state):
    """Engine calls this once per game. Reset per-game state and
    precompute the maze distance table up front."""
    global _pacman_target, _ghost_target
    _pacman_target = None
    _ghost_target = None
    _get_distance_map(state)


# --------------------------------------------------------------- A* ----

def astar(start, goal, neighbors, cost, heuristic):
    """Cheapest path [start, ..., goal], or None if there is none."""
    if start == goal:
        return [start]

    counter = itertools.count()
    open_heap = [(heuristic(start), next(counter), start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
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
            if tentative_g < g_score.get(nbr, float('inf')):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                heapq.heappush(open_heap, (tentative_g + heuristic(nbr), next(counter), nbr))

    return None


# ---------------------------------------------- maze distance table ----

def _get_distance_map(state):
    key = state.walls
    table = _distance_cache.get(key)
    if table is None:
        table = _compute_all_pairs_bfs(state)
        _distance_cache[key] = table
    return table


def _compute_all_pairs_bfs(state):
    """dist[a][b] = shortest steps from a to b (cost-1 edges). The board
    never changes within a game, so this is computed once and cached."""
    dist = {}
    cells = [(r, c) for r in range(state.rows) for c in range(state.cols)
             if (r, c) not in state.walls]
    for src in cells:
        d = {src: 0}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for nbr in state.legal_moves(cur).values():
                if nbr not in d:
                    d[nbr] = d[cur] + 1
                    queue.append(nbr)
        dist[src] = d
    return dist


def _heuristic_to(state, cell, target):
    d = _get_distance_map(state).get(cell, {}).get(target)
    return d if d is not None else state.manhattan(cell, target)


# ------------------------------------------------------- evaluate ----

def evaluate(state, pacman, ghost, food):
    """Larger = better for Pac-Man. pacman/ghost/food come in as explicit
    arguments (rather than read off state) because lookahead() calls this
    on hypothetical future positions -- including branches where the
    ghost has dropped out of simulated sight."""
    if ghost is not None and pacman == ghost:
        return -10_000.0

    dist_map = _get_distance_map(state)
    score = 0.0

    score -= len(food) * 10
    if food:
        pac_row = dist_map.get(pacman, {})
        nearest = min(pac_row.get(f, 999) for f in food)
        score -= nearest * 2

    if ghost is not None:
        g_dist = dist_map.get(pacman, {}).get(ghost, 999)
        if g_dist <= 2:
            score -= (3 - g_dist) * 200
        elif g_dist <= 5:
            score -= (6 - g_dist) * 20

    return score


# ------------------------------------------------------- lookahead ----

def _ghost_reply_options(state, ghost_pos):
    if not state.ghost_can_move:
        return [None]          # flagged assumption -- see module docstring
    if ghost_pos is None:
        return [None]
    moves = list(state.legal_moves(ghost_pos).keys())
    return moves if moves else [None]


def lookahead(state, depth):
    ghost_pos = state.ghost if state.ghost is not None else (
        state.ghost_last_seen[0] if state.ghost_last_seen else None)

    best_move, best_value = 'STOP', float('-inf')
    for move in list(state.legal_moves(state.pacman).keys()) + ['STOP']:
        value = _expand(state, move, ghost_pos, depth)
        if value > best_value:
            best_value, best_move = value, move
    return best_move


def _expand(state, pac_move, ghost_pos, depth):
    values = []
    for ghost_reply in _ghost_reply_options(state, ghost_pos):
        nxt = state.next(pac_move, ghost_reply)
        new_ghost_pos = nxt.ghost if nxt.ghost is not None else ghost_pos

        if nxt.done:
            if nxt.caught:
                values.append(-10_000.0)
            elif not nxt.food:
                values.append(10_000.0)
            else:
                values.append(evaluate(nxt, nxt.pacman, new_ghost_pos, nxt.food))
        elif depth <= 1:
            values.append(evaluate(nxt, nxt.pacman, new_ghost_pos, nxt.food))
        else:
            values.append(_best_reply(nxt, new_ghost_pos, depth - 1))

    if OPPONENT_MODEL == "minimax":
        return min(values)
    return sum(values) / len(values)


def _best_reply(state, ghost_pos, depth):
    best = float('-inf')
    for move in list(state.legal_moves(state.pacman).keys()) + ['STOP']:
        best = max(best, _expand(state, move, ghost_pos, depth))
    return best


# ------------------------------------------------------ pacman_move ----

def _choose_pacman_target(state):
    global _pacman_target
    if not state.food:
        return None
    if _pacman_target is not None and _pacman_target in state.food:
        return _pacman_target
    dist_map = _get_distance_map(state)
    pac_row = dist_map.get(state.pacman, {})
    _pacman_target = min(state.food, key=lambda f: (pac_row.get(f, float('inf')), f))
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

    if state.ghost is not None:
        return lookahead(state, LOOKAHEAD_DEPTH)

    target = _choose_pacman_target(state)
    if target is None:
        return 'STOP'

    path = astar(state.pacman, target,
                 neighbors=lambda c: list(state.legal_moves(c).values()),
                 cost=lambda a, b: _pacman_step_cost(state, a, b),
                 heuristic=lambda c: _heuristic_to(state, c, target))

    if not path or len(path) < 2:
        _pacman_target = None
        return 'STOP'

    next_cell = path[1]
    for move, cell in state.legal_moves(state.pacman).items():
        if cell == next_cell:
            return move
    return 'STOP'


# ------------------------------------------------------- ghost_move ----

def _choose_ghost_target(state):
    """Only called when Pac-Man is currently out of the ghost's sight."""
    global _ghost_target
    if state.pacman_last_seen is None:
        return None
    last_cell, _tick = state.pacman_last_seen
    if state.ghost == last_cell and state.food:
        dist_map = _get_distance_map(state)
        ref_row = dist_map.get(last_cell, {})
        _ghost_target = min(state.food, key=lambda f: (ref_row.get(f, float('inf')), f))
        return _ghost_target
    return last_cell


def ghost_move(state):
    target = state.pacman if state.pacman is not None else _choose_ghost_target(state)

    if target is not None:
        path = astar(state.ghost, target,
                     neighbors=lambda c: list(state.legal_moves(c).values()),
                     cost=lambda a, b: 1,
                     heuristic=lambda c: _heuristic_to(state, c, target))
        if path and len(path) >= 2:
            next_cell = path[1]
            for move, cell in state.legal_moves(state.ghost).items():
                if cell == next_cell:
                    return move

    moves = list(state.legal_moves(state.ghost).keys())
    return moves[0] if moves else 'N'
