"""
agent_template.py — your starting point for Assignment 1, The Pac-Man League.

1. Copy this file to  agent_<surname>.py  (lowercase, no spaces, e.g. agent_karimova.py).
2. Fill in the five functions below, in this order:
       astar  ->  pacman_move / ghost_move (out of sight)  ->  evaluate  ->  lookahead
3. Set OPPONENT_MODEL to the model you chose, and justify it in your report.
4. Run  python check_submission.py agent_<surname>.py  until everything is [OK].

Rules that the checker enforces:
  * standard library only — allowed imports: math, heapq, collections, itertools,
    functools, dataclasses, typing, time
  * both pacman_move and ghost_move call YOUR astar
  * whenever the ghost is in sight, pacman_move chooses its move with lookahead()
  * every move must return within 100 ms
  * no randomness; do not touch anything that starts with an underscore on the state

The state object you receive is described at the bottom of this file.
"""
import heapq
import time

# "minimax"   : the ghost's reply is scored as the worst case for you
# "expectimax": the ghost's reply is scored as the average over its moves
OPPONENT_MODEL = "minimax"

LOOKAHEAD_DEPTH = 3
ROOM_RADIUS = 8        # count escape room only when the ghost is this close (steps)
ROOM_WEIGHT = 4        # value of one cell of escape room
ROOM_CAP = 60
PAC_FOOD_ROOM = 20      # Pac-Man: value of one food we can reach before the ghost 
SEARCH_MODE = "food"
CHASE_PLIES = 5
FOOD_ROOM_WEIGHT = 100    # ghost: how much one reachable food for Pac-Man counts         # how many half-moves the ghost looks ahead when it sees Pac-Man  # how the ghost searches when it cannot see Pac-Man
         # more room than this does not matter          # rounds to look ahead; 1 round = your move + the ghost's reply

STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


# ----------------------------------------------------------------------------- 1. A*

def astar(start, goal, neighbors, cost, heuristic):
    """Return the cheapest path from start to goal as a list of cells
    [start, ..., goal], or None if there is no path.

        neighbors(cell) -> iterable of cells you can step to
        cost(a, b)      -> number, cost of stepping from a to b
        heuristic(cell) -> number, estimate of the remaining cost to goal

    This is the generic search of Lab 3 with a priority queue ordered by f = g + h.
    Test the goal when a cell comes OUT of the frontier, keep the best g per cell,
    and rebuild the path from a came_from dictionary."""
    counter = 0                                   # tie-breaker so heapq never compares cells
    frontier = [(heuristic(start), counter, start)]
    g = {start: 0}                                # best known cost to reach each cell
    came_from = {start: None}

    while frontier:
        f, _, cell = heapq.heappop(frontier)
        if cell == goal:                          # goal test when the cell comes OUT
            path = [cell]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return path[::-1]
        if f - heuristic(cell) > g[cell]:         # an old, worse entry: skip it
            continue
        for nxt in neighbors(cell):
            new_g = g[cell] + cost(cell, nxt)
            if new_g < g.get(nxt, float("inf")):
                g[nxt] = new_g
                came_from[nxt] = cell
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(nxt), counter, nxt))
    return None


# --------------------------------------------------------------- 2. helpers you will want

def first_step(path):
    """The move ('N', 'S', 'E' or 'W') that takes you from path[0] to path[1]."""
    (r0, c0), (r1, c1) = path[0], path[1]
    return {v: k for k, v in STEP.items()}[(r1 - r0, c1 - c0)]


def maze_neighbors(state):
    """neighbors() for astar on this board: the cells you can step to."""
    return lambda cell: state.legal_moves(cell).values()


_DIST_CACHE = {}

def maze_dist(state, start):
    """Real number of steps from start to every cell (walls matter). Cached per cell."""
    key = (state.walls, start)
    if key not in _DIST_CACHE:
        from collections import deque
        dist = {start: 0}
        q = deque([start])
        while q:
            cell = q.popleft()
            for nxt in state.legal_moves(cell).values():
                if nxt not in dist:
                    dist[nxt] = dist[cell] + 1
                    q.append(nxt)
        _DIST_CACHE[key] = dist
    return _DIST_CACHE[key]


_POCKET_CACHE = {}

def dead_end_cells(state):
    """Cells inside dead-end corridors (you can be trapped there).
    Found by repeatedly removing cells that have only one open neighbour."""
    if state.walls not in _POCKET_CACHE:
        open_cells = {(r, c) for r in range(state.rows) for c in range(state.cols)} - state.walls
        degree = {cell: len(state.legal_moves(cell)) for cell in open_cells}
        pocket = set()
        stack = [cell for cell, deg in degree.items() if deg <= 1]
        while stack:
            cell = stack.pop()
            if cell in pocket:
                continue
            pocket.add(cell)
            for nxt in state.legal_moves(cell).values():
                if nxt not in pocket:
                    degree[nxt] -= 1
                    if degree[nxt] <= 1:
                        stack.append(nxt)
        _POCKET_CACHE[state.walls] = frozenset(pocket)
    return _POCKET_CACHE[state.walls]


_ENTRANCE_CACHE = {}

def pocket_exit(state):
    """For every cell inside a dead-end corridor: the first cell OUTSIDE it (the way out)."""
    if state.walls not in _ENTRANCE_CACHE:
        pocket = dead_end_cells(state)
        exit_of = {}
        from collections import deque
        q = deque()
        for cell in pocket:
            for nxt in state.legal_moves(cell).values():
                if nxt not in pocket:
                    exit_of[cell] = nxt
                    q.append(cell)
        while q:
            cell = q.popleft()
            for nxt in state.legal_moves(cell).values():
                if nxt in pocket and nxt not in exit_of:
                    exit_of[nxt] = exit_of[cell]
                    q.append(nxt)
        _ENTRANCE_CACHE[state.walls] = exit_of
    return _ENTRANCE_CACHE[state.walls]


def ghost_steps_needed(state, cell):
    """The FEWEST steps the ghost could still need to reach `cell` (it may be anywhere it
    could have walked since we last saw it). Small number = danger."""
    if state.ghost is not None:
        return maze_dist(state, state.ghost).get(cell, 99)
    if state.ghost_last_seen is not None:
        pos, seen_tick = state.ghost_last_seen
    else:                                           # never seen: it starts bottom-right
        pos, seen_tick = _guess_ghost_start(state), 1
    elapsed = state.tick - seen_tick
    walked = elapsed - elapsed // 4                 # the ghost sits out every 4th tick
    return max(0, maze_dist(state, pos).get(cell, 99) - walked)


_START = {}                                         # Pac-Man's first position this game

def new_game(state):
    """Called by the engine once at the start of every game."""
    _START.clear()
    _BELIEF.clear()


def _guess_ghost_start(state):
    # the ghost starts in the corner opposite Pac-Man's start
    if "pacman" not in _START and state.pacman is not None:
        _START["pacman"] = state.pacman
    pr, pc = _START.get("pacman", (1, 1))
    corner = (state.rows - 1 - pr, state.cols - 1 - pc)
    open_cells = [(r, c) for r in range(state.rows) for c in range(state.cols) if (r, c) not in state.walls]
    return min(open_cells, key=lambda cell: (state.manhattan(cell, corner), cell))


# ------------------------------------------------------------------ 3. evaluation function

def evaluate(state, pacman, ghost, food):
    """How good this position is for Pac-Man: a number, larger = better.

    `state` is a State (usually one produced by state.next(...)); pacman and ghost are
    positions, food is the set of food still on the board. A first version could count
    food eaten, subtract the distance to the nearest food and punish being close to the
    ghost. Use real maze distances rather than Manhattan: walls matter. Remember that
    pacman == ghost means you were caught."""
    if pacman == ghost or state.caught:            # caught: the worst thing that can happen
        return -100000
    if not food:                                   # board cleared: the best thing
        return 100000

    d = maze_dist(state, pacman)
    nearest_food = min(d.get(f, 99) for f in food)
    ghost_dist = d.get(ghost, 99) if ghost is not None else 99

    score = -100 * len(food)                       # every food eaten = +100
    score -= nearest_food                          # be close to the next food
    if ghost_dist <= 1:
        score -= 5000                              # ghost next to us: very bad
    elif ghost_dist <= 4:
        score -= 300 / ghost_dist                  # ghost close: bad
    if ghost is not None and ghost_dist <= ROOM_RADIUS:
        # escape room: cells we reach before the ghost (it is 3/4 as fast). More room = safer.
        dg = maze_dist(state, ghost)
        room = 0
        food_room = 0
        for cell, steps in d.items():
            if steps * 3 < dg.get(cell, 99) * 4:      # we get there before the ghost
                room += 1
                if cell in food:
                    food_room += 1
        score += ROOM_WEIGHT * min(room, ROOM_CAP) + PAC_FOOD_ROOM * food_room
    if pacman in dead_end_cells(state) and ghost is not None:
        # in a dead end: trapped if the ghost can reach the way out before we get out
        out = pocket_exit(state)[pacman]
        if maze_dist(state, ghost).get(out, 99) <= d.get(out, 99) + 1:
            score -= 3000
    return score


# ------------------------------------------------------------------------ 4. lookahead

def lookahead(state, depth):
    """Pac-Man's best move when the ghost is in sight, by looking `depth` rounds ahead.

    One round = your move, then the ghost's reply. For each of your legal moves (and
    STOP), consider each legal reply of the ghost, build the position with
    state.next(my_move, ghost_move), and score it: with evaluate() at the last round or
    when the game is over, otherwise by looking one round deeper. Combine the ghost's
    replies with min() under "minimax" or with the mean under "expectimax" (see
    OPPONENT_MODEL), and return the move with the best value.

    Details that matter: state.ghost_can_move tells you whether the ghost moves in this
    round (it sits out every 4th tick) — when it cannot move there is exactly one reply,
    'stay' (pass None as its move)."""
    def value(st, d):
        if st.done or d == 0:
            return evaluate(st, st.pacman, st.ghost, st.food)
        return max(after_my_move(st, m, d) for m in list(st.legal_moves(st.pacman)) + ["STOP"])

    def after_my_move(st, my_move, d):
        # all replies of the ghost (or just 'stay' when it sits out this tick)
        if st.ghost_can_move:
            replies = list(st.legal_moves(st.ghost))
        else:
            replies = [None]
        scores = [value(st.next(my_move, g), d - 1) for g in replies]
        if OPPONENT_MODEL == "minimax":
            return min(scores)                     # the ghost plays its best reply
        return sum(scores) / len(scores)           # expectimax: the average reply

    best_move, best_score = "STOP", float("-inf")
    for move in list(state.legal_moves(state.pacman)) + ["STOP"]:
        score = after_my_move(state, move, depth)
        if score > best_score:
            best_move, best_score = move, score
    return best_move


# ---------------------------------------------------------------------- 5. the two movers

def pacman_move(state):
    """Return 'N', 'S', 'E', 'W' or 'STOP'."""
    me = state.pacman

    if state.ghost is not None:                        # the ghost is in sight
        return lookahead(state, LOOKAHEAD_DEPTH)

    # Out of sight: choose a target, a cost function and a heuristic, then A*.
    if not state.food:
        return "STOP"
    d = maze_dist(state, me)
    pocket = dead_end_cells(state)
    exit_of = pocket_exit(state)

    def risky(f):
        """Is it dangerous to go for food f? For food in a dead end: can we eat it and get
        back out before the ghost could reach the way out?"""
        if f in pocket:
            out = exit_of[f]
            ticks_needed = d.get(f, 99) + maze_dist(state, f).get(out, 99)
            return ghost_steps_needed(state, out) <= 0.75 * ticks_needed + 2
        return False

    safe = [f for f in state.food if not risky(f)]
    if safe:
        target = min(safe, key=lambda f: (d.get(f, 999), f))           # nearest safe food by REAL maze distance
    else:
        # only risky food is left: being caught halves the score, so wait in a safe spot
        # (outside dead ends, as far from the ghost as possible) until it becomes safe
        ref = state.ghost_last_seen[0] if state.ghost_last_seen else _guess_ghost_start(state)
        from_ghost = maze_dist(state, ref)
        spots = [c for c in d if c not in pocket and d[c] <= 8]
        target = max(spots, key=lambda c: (ghost_steps_needed(state, c), from_ghost.get(c, 0), -d[c], c))
        if target == me:
            return "STOP"

    def cost(a, b):
        need = ghost_steps_needed(state, b)
        if need <= 2:
            return 1 + 10                                               # the ghost could be here soon
        if need <= 4:
            return 1 + 3
        return 1

    def heuristic(cell):
        return state.manhattan(cell, target)                            # admissible: every step costs >= 1

    path = astar(me, target, maze_neighbors(state), cost, heuristic)
    if path is None or len(path) < 2:
        return "STOP"
    return first_step(path)


# ------------------------------------------------------------------ smart ghost

_BELIEF = {}      # where Pac-Man could be: {"cells": set, "food": frozenset, "tick": int}


def _open_cells(state):
    return [(r, c) for r in range(state.rows) for c in range(state.cols) if (r, c) not in state.walls]


def _grow(state, cells, steps):
    """All cells reachable from `cells` in at most `steps` moves (staying is allowed)."""
    cells = set(cells)
    for _ in range(steps):
        new = set(cells)
        for cell in cells:
            new.update(state.legal_moves(cell).values())
        if new == cells:
            break
        cells = new
    return cells


def update_belief(state):
    """Keep the set of cells where Pac-Man could be right now.
    Clues: we see it (exact), food that just disappeared (it was there), food that is
    still on a cell (it has NOT walked there - it eats everything it steps on), and our own
    vision (it is not within 6 cells, otherwise we would see it)."""
    me = state.ghost
    if "ghost_start" not in _START:
        _START["ghost_start"] = me
    if state.pacman is not None:
        cells = {state.pacman}
    else:
        prev = _BELIEF.get("cells")
        if prev is None:
            if state.pacman_last_seen is not None:
                pos, seen = state.pacman_last_seen
                prev, steps = {pos}, state.tick - seen
            else:                                   # never seen: it started in the opposite corner
                gr, gc = _START["ghost_start"]
                corner = (state.rows - 1 - gr, state.cols - 1 - gc)
                start = min(_open_cells(state), key=lambda c: (state.manhattan(c, corner), c))
                prev, steps = {start}, state.tick - 1
        else:
            steps = state.tick - _BELIEF["tick"]
        cells = _grow(state, prev, max(1, steps))
        eaten = (_BELIEF.get("food") or state.food) - state.food
        if eaten:                                   # it was on an eaten cell very recently
            near = _grow(state, eaten, max(0, steps - 1))
            cells = (cells & near) or near
        cells = {c for c in cells if c not in state.food and state.manhattan(c, me) > state.vision}
        if not cells:                               # lost it completely: anywhere unseen and eaten
            cells = {c for c in _open_cells(state)
                     if c not in state.food and state.manhattan(c, me) > state.vision}
    _BELIEF.update(cells=cells, food=state.food, tick=state.tick)
    return cells


def escape_room(state, ghost, pac):
    """How many cells Pac-Man reaches before the ghost can (the ghost is 3/4 as fast).
    Small = Pac-Man is cornered. The ghost tries to make this as small as possible."""
    if ghost == pac:
        return -10**9
    dg, dp = maze_dist(state, ghost), maze_dist(state, pac)
    room = 0
    food_room = 0
    food = state.food
    for cell, steps in dp.items():
        if steps * 4 < dg.get(cell, 99) * 3:          # Pac-Man gets there first
            room += 1
            if cell in food:
                food_room += 1
    # the ghost wins by catching OR by keeping Pac-Man away from the food
    return food_room * FOOD_ROOM_WEIGHT + room * 10 + dg.get(pac, 99)


def chase_move(state, me, pac):
    """Ghost in sight of Pac-Man: minimax with alpha-beta over CHASE_PLIES half-moves
    (ghost, Pac-Man, ghost, ...). Pac-Man looks for room to escape, the ghost picks the moves
    that leave it the least room. Leaves are scored with escape_room (lower = better for us)."""
    INF = 10**12

    def ghost_options(cell, tick):
        # the ghost must move when it can (no standing still); on its skip tick it stays
        return list(state.legal_moves(cell).values()) if tick % 4 != 0 else [cell]

    def pacman_options(cell):
        return [cell] + list(state.legal_moves(cell).values())   # Pac-Man may stop

    def ghost_turn(g, p, tick, plies, alpha, beta):          # we minimise
        best = INF
        for g2 in sorted(ghost_options(g, tick), key=lambda c: maze_dist(state, c).get(p, 99)):
            if g2 == p:
                return -INF + (CHASE_PLIES - plies)          # caught: sooner is better
            v = escape_room(state, g2, p) if plies <= 1 else pac_turn(g2, p, tick + 1, plies - 1, alpha, beta)
            best = min(best, v)
            beta = min(beta, v)
            if beta <= alpha:
                break
        return best

    def pac_turn(g, p, tick, plies, alpha, beta):            # Pac-Man maximises
        best = -INF
        for p2 in pacman_options(p):
            if p2 == g:
                continue                                     # it will not walk into us
            v = escape_room(state, g, p2) if plies <= 1 else ghost_turn(g, p2, tick, plies - 1, alpha, beta)
            best = max(best, v)
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return best if best > -INF else -INF + 1             # every move gets caught

    def search(plies):
        best_cell, best_key = me, None
        for g1 in sorted(ghost_options(me, state.tick), key=lambda c: (maze_dist(state, c).get(pac, 99), c)):
            if g1 == pac:
                return g1                                    # catch it right now
            v = pac_turn(g1, pac, state.tick + 1, plies - 1, -INF, INF) if plies > 1 else escape_room(state, g1, pac)
            key = (v, maze_dist(state, g1).get(pac, 99), g1)
            if best_key is None or key < best_key:
                best_cell, best_key = g1, key
        return best_cell

    # think 3 half-moves first; go deeper only if that was fast (stay far below 100 ms)
    started = time.perf_counter()
    move = search(3)
    if CHASE_PLIES > 3 and time.perf_counter() - started < 0.006:
        move = search(CHASE_PLIES)
    return move


def search_target(state, me, cells):
    """Pac-Man is out of sight: go to the 'middle' of the cells where it could be,
    so we get closer to all of them at once."""
    pool = sorted(cells)
    sample = pool[:: max(1, len(pool) // 40)]
    def spread(t):
        d = maze_dist(state, t)
        return sum(d.get(c, 99) for c in cells) / len(cells)
    return min(sample, key=lambda t: (spread(t), maze_dist(state, me).get(t, 99), t))


def ghost_move(state):
    """Your ghost's move this tick.
    In sight: look ahead and corner Pac-Man. Out of sight: track where it can be and go there."""
    me = state.ghost
    legal = state.legal_moves(me)
    cells = update_belief(state)

    if state.pacman is not None:                       # Pac-Man is in sight: corner it
        step = chase_move(state, me, state.pacman)
        for move, cell in sorted(legal.items()):
            if cell == step:
                return move
        return sorted(legal)[0]

    if not cells:
        target = me
    elif SEARCH_MODE == "nearest":
        dme = maze_dist(state, me)
        target = min(cells, key=lambda c: (dme.get(c, 99), c))
    elif SEARCH_MODE == "food":
        # Pac-Man is probably next to food: among possible cells prefer those near food
        dme = maze_dist(state, me)
        near_food = [c for c in cells if any(n in state.food for n in state.legal_moves(c).values())]
        pool = near_food or list(cells)
        target = min(pool, key=lambda c: (dme.get(c, 99), c))
    else:
        target = search_target(state, me, cells)
    if target == me:
        return sorted(legal)[0]
    path = astar(me, target, maze_neighbors(state), lambda a, b: 1, lambda c: state.manhattan(c, target))
    if not path or len(path) < 2:
        return sorted(legal)[0]
    for move, cell in sorted(legal.items()):
        if cell == path[1]:
            return move
    return sorted(legal)[0]
