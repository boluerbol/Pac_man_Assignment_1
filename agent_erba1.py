
import heapq
import time
from collections import deque

# "minimax"   : the ghost's reply is scored as the worst case for you
# "expectimax": the ghost's reply is scored as the average over its moves
OPPONENT_MODEL = "minimax"

LOOKAHEAD_DEPTH = 8          # maximum rounds; the search stops earlier when time runs out
TIME_BUDGET = 0.030          # seconds of search per move (the limit is 0.100)

STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
ORDER = ("N", "S", "E", "W")
INF = 10 ** 6
LOST = -100000               # value of a position where Pac-Man is caught or trapped
WIN = 100000                 # value of a cleared board


# ----------------------------------------------------------------------------- 1. A*

def astar(start, goal, neighbors, cost, heuristic):
    """Return the cheapest path from start to goal as a list of cells
    [start, ..., goal], or None if there is no path."""
    counter = 0                                   # tie-breaker: cells are never compared
    frontier = [(heuristic(start), counter, start)]
    best_g = {start: 0}
    came_from = {start: None}
    closed = set()
    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        if cell == goal:                          # goal test when it leaves the frontier
            path = [cell]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return path[::-1]
        closed.add(cell)
        g = best_g[cell]
        for nxt in neighbors(cell):
            ng = g + cost(cell, nxt)
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng
                came_from[nxt] = cell
                counter += 1
                heapq.heappush(frontier, (ng + heuristic(nxt), counter, nxt))
    return None


# --------------------------------------------------------------- 2. helpers

def first_step(path):
    """The move ('N', 'S', 'E' or 'W') that takes you from path[0] to path[1]."""
    (r0, c0), (r1, c1) = path[0], path[1]
    return {v: k for k, v in STEP.items()}[(r1 - r0, c1 - c0)]


def maze_neighbors(state):
    """neighbors() for astar on this board: the cells you can step to."""
    return lambda cell: state.legal_moves(cell).values()


# ---- board analysis, built once per board

board_walls = None
CELLS = []        # index -> cell
INDEX = {}        # cell -> index
DIST = []         # DIST[i][j] maze distance (INF if unreachable)
CORE = []         # CORE[i]: cell i lies on a loop (or links loops)
MOUTH = []        # MOUTH[i]: for a pocket cell, the core cell its pocket opens onto
GUARD = {}        # cache: food set -> cell closest to all food
LURE = -1         # index of the food's centre during a search (-1: not used)
CHASE_MODEL = False   # during Pac-Man's search: expect the ghost to chase (see reply_value)


def prepare(state):
    global board_walls, CELLS, INDEX, DIST, CORE, MOUTH, GUARD
    if board_walls is state.walls:
        return
    cells = [(r, c) for r in range(state.rows) for c in range(state.cols) if (r, c) not in state.walls]
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    nbrs = [[index[x] for x in state.legal_moves(cell).values() if x in index] for cell in cells]

    dist = []
    for s in range(n):
        d = [INF] * n
        d[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = d[u] + 1
            for v in nbrs[u]:
                if d[v] == INF:
                    d[v] = du
                    q.append(v)
        dist.append(d)

    # the core: repeatedly peel off cells with at most one neighbour left
    degree = [len(x) for x in nbrs]
    core = [True] * n
    q = deque(i for i in range(n) if degree[i] <= 1)
    while q:
        u = q.popleft()
        if not core[u]:
            continue
        core[u] = False
        for v in nbrs[u]:
            if core[v]:
                degree[v] -= 1
                if degree[v] <= 1:
                    q.append(v)
    if not any(core):                              # a maze without loops: treat it all as open
        core = [True] * n

    # every pocket cell opens onto one core cell
    mouth = [i if core[i] else -1 for i in range(n)]
    q = deque(i for i in range(n) if core[i])
    while q:
        u = q.popleft()
        for v in nbrs[u]:
            if mouth[v] == -1:
                mouth[v] = mouth[u]
                q.append(v)
    for i in range(n):
        if mouth[i] == -1:                         # unreachable piece: harmless
            mouth[i] = i

    CELLS, INDEX, DIST, CORE, MOUTH, GUARD = cells, index, dist, core, mouth, {}
    board_walls = state.walls


def dist_row(cell):
    return DIST[INDEX[cell]]


ARRIVAL = {}      # tick -> list: ARRIVAL[tick][k] = tick of the ghost's k-th move from `tick` on


def ghost_ticks(tick):
    """Ticks at which the ghost makes its 1st, 2nd, ... move, counting from `tick`
    (it moves on every tick except multiples of 4)."""
    row = ARRIVAL.get(tick)
    if row is None:
        row = [-1]
        t = tick
        while len(row) < 120:
            if t % 4 != 0:
                row.append(t)
            t += 1
        ARRIVAL[tick] = row
    return row


def ghost_time(row, d):
    """Tick of the ghost's d-th move (row from ghost_ticks); d = 0 means it is already there."""
    return row[d] if d < len(row) else INF


def territory(dp, dg):
    """Cells Pac-Man reaches strictly before the ghost: its room to live in."""
    return sum(1 for a, b in zip(dp, dg) if a < b)


# ------------------------------------------------------------------ 3. evaluation function

def evaluate(state, pacman, ghost, food):
    """How good this position is for Pac-Man: a number, larger = better.

    Timing: Pac-Man's j-th move from here happens on tick  tick + j - lag  (lag = 1 when
    it is Pac-Man's turn), the ghost's d-th move on ghost_ticks(tick)[d]. Pac-Man gets
    past a cell safely only if it is there on an earlier tick than the ghost."""
    prepare(state)
    eaten = state.food_eaten
    if ghost is not None and pacman == ghost:
        return LOST - 1000 + 50 * eaten           # caught: the score is halved, food still counts
    if not food:
        return WIN + 50 * eaten
    pi = INDEX[pacman]
    dp = DIST[pi]
    score = 100.0 * eaten

    if ghost is None:
        score -= 2 * min(dp[INDEX[f]] for f in food)
        return score

    gi = INDEX[ghost]
    dg = DIST[gi]
    tick = state.tick
    pac_tick = tick - (1 if state.turn == "pacman" else 0)    # + j = tick of Pac-Man's j-th move
    gt = ghost_ticks(tick)
    nrow = len(gt)

    # trapped in a pocket: the ghost reaches the mouth first -> lost (it only has to walk in)
    if not CORE[pi]:
        m = MOUTH[pi]
        dgm = dg[m]
        if pac_tick + dp[m] >= (gt[dgm] if dgm < nrow else INF):
            return LOST + 50 * eaten - dp[gi]

    # nearest food Pac-Man can fetch safely (and, in a pocket, get back out from)
    best_safe = best_any = INF
    for f in food:
        i = INDEX[f]
        d = dp[i]
        if d < best_any:
            best_any = d
        if d >= best_safe:
            continue
        if CORE[i]:
            dgi = dg[i]
            safe = pac_tick + d < (gt[dgi] if dgi < nrow else INF)
        else:
            m = MOUTH[i]
            dgm = dg[m]
            safe = pac_tick + d + DIST[i][m] < (gt[dgm] if dgm < nrow else INF)
        if safe:
            best_safe = d
    score -= 2 * (best_safe if best_safe < INF else best_any + 15)

    gd = dp[gi]
    if gd <= 1:
        score -= 30
    elif gd == 2:
        score -= 10
    score += 0.3 * territory(dp, dg)

    # the lure: keep the ghost away from the food that is left. When nothing is safe to
    # eat, leading the ghost off (it is slower) is worth more than anything else.
    if LURE >= 0:
        score += (1.5 if best_safe == INF else 0.3) * dg[LURE]
    return score


# ------------------------------------------------------------------------ 4. lookahead

class OutOfTime(Exception):
    pass


def pacman_options(state):
    legal = state.legal_moves(state.pacman)
    return [m for m in ORDER if m in legal] + ["STOP"]


def ghost_options(state):
    if not state.ghost_can_move:
        return [None]                             # skip tick: the only reply is 'stay'
    legal = state.legal_moves(state.ghost)
    return [m for m in ORDER if m in legal]


def leaf(state):
    return evaluate(state, state.pacman, state.ghost, state.food)


# Pac-Man to move: one round = Pac-Man's move, then the ghost's reply.

def max_value(state, depth, alpha, beta, deadline):
    if state.done or depth == 0:
        return leaf(state)
    if time.perf_counter() > deadline:
        raise OutOfTime
    best = -float("inf")
    for pm in pacman_options(state):
        v = reply_value(state, pm, depth, alpha, beta, deadline)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def reply_value(state, pm, depth, alpha, beta, deadline):
    """Value of Pac-Man playing pm, over the ghost's replies."""
    replies = ghost_options(state)
    if CHASE_MODEL and len(replies) > 1:
        # The decoy: safety is still judged by the worst reply (a reply that catches or
        # traps Pac-Man rules the move out), but otherwise the ghost is expected to do
        # what it has been doing: step toward Pac-Man. That is what makes leading it
        # away from the food worth something.
        pac_after = state.legal_moves(state.pacman).get(pm, state.pacman)
        dpa = dist_row(pac_after)
        ghost_legal = state.legal_moves(state.ghost)
        values = {}
        for gm in replies:
            values[gm] = max_value(state.next(pm, gm), depth - 1, -float("inf"), float("inf"), deadline)
        worst = min(values.values())
        if worst <= LOST // 2:
            return worst
        closest = min(dpa[INDEX[ghost_legal[gm]]] for gm in replies)
        return min(v for gm, v in values.items() if dpa[INDEX[ghost_legal[gm]]] == closest)
    if OPPONENT_MODEL == "expectimax":
        total = 0.0
        for gm in replies:
            total += max_value(state.next(pm, gm), depth - 1, -float("inf"), float("inf"), deadline)
        return total / len(replies)
    worst = float("inf")
    for gm in replies:
        v = max_value(state.next(pm, gm), depth - 1, alpha, beta, deadline)
        if v < worst:
            worst = v
        if worst < beta:
            beta = worst
        if alpha >= beta:
            break
    return worst


def lookahead(state, depth):
    """Pac-Man's best move when the ghost is in sight: minimax / expectimax over
    `depth` rounds at most, deepening one round at a time while time is left."""
    global LURE, CHASE_MODEL
    prepare(state)
    deadline = time.perf_counter() + TIME_BUDGET
    LURE = INDEX[guard_cell(state)] if state.food else -1
    CHASE_MODEL = trust_chaser and chase_misses <= 1
    moves = pacman_options(state)
    best_move = moves[0]
    try:
        for d in range(1, depth + 1):
            try:
                alpha, best_here = -float("inf"), None
                for pm in moves:
                    v = reply_value(state, pm, d, alpha, float("inf"), deadline)
                    if best_here is None or v > alpha:
                        alpha, best_here = v, pm
            except OutOfTime:
                break
            best_move = best_here
            moves = [best_move] + [m for m in moves if m != best_move]   # best first: more pruning
            if alpha >= WIN // 2 or alpha <= LOST // 2:   # forced result: deeper will not change it
                break
    finally:
        LURE = -1
        CHASE_MODEL = False
    return best_move


# Ghost to move: one round = the ghost's move, then Pac-Man's reply (next tick).

def min_value(state, depth, alpha, beta, deadline):
    if state.done or depth == 0:
        return leaf(state)
    if time.perf_counter() > deadline:
        raise OutOfTime
    best = float("inf")
    for gm in ghost_options(state):
        v = pac_reply_value(state, gm, depth, alpha, beta, deadline)
        if v < best:
            best = v
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return best


def pac_reply_value(state, gm, depth, alpha, beta, deadline):
    best = -float("inf")
    for pm in pacman_options(state):
        v = min_value(state.next(pm, gm), depth - 1, alpha, beta, deadline)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def ghost_lookahead(state, depth):
    global LURE
    deadline = time.perf_counter() + TIME_BUDGET
    LURE = INDEX[guard_cell(state)] if state.food else -1
    moves = [m for m in ORDER if m in state.legal_moves(state.ghost)]
    best_move = moves[0]
    try:
        for d in range(1, depth + 1):
            try:
                beta, best_here = float("inf"), None
                for gm in moves:
                    v = pac_reply_value(state, gm, d, -float("inf"), beta, deadline)
                    if best_here is None or v < beta:
                        beta, best_here = v, gm
            except OutOfTime:
                break
            best_move = best_here
            moves = [best_move] + [m for m in moves if m != best_move]
            if beta <= LOST // 2 or beta >= WIN // 2:
                break
    finally:
        LURE = -1
    return best_move


# ---------------------------------------------------------------------- 5. the two movers

prev_food = None          # ghost: food set at its previous move
pac_estimate = None       # ghost: (cell, tick) where Pac-Man was last known to be
trust_chaser = True       # Pac-Man: does this ghost fall for the decoy (go where it last saw us)?
predicted = None          # Pac-Man: (cells, tick) where the decoy model put the ghost last move
chase_misses = 0          # Pac-Man: times the ghost, in sight, did not step toward Pac-Man
last_sight = None         # Pac-Man: (ghost cell, tick) at its previous move, if in sight


def new_game(state):
    """Called by the engine once at the start of every game."""
    global prev_food, pac_estimate, trust_chaser, predicted, chase_misses, last_sight
    prepare(state)
    prev_food = state.food
    pac_estimate = None
    trust_chaser = True
    predicted = None
    chase_misses = 0
    last_sight = None


def ghost_moves_since(state, since):
    return sum(1 for t in range(since, state.tick) if t % 4 != 0)


def chaser_ghost_cells(state):
    """The decoy model: most ghosts that lose sight of Pac-Man walk to where they saw it
    last, then search around there. So the ghost is on a shortest path from where it was
    seen to where Pac-Man was then, as far along as its moves allow, or, once it got
    there, near that spot. None when the model has nothing to work with."""
    seen, pseen = state.ghost_last_seen, state.pacman_last_seen
    if seen is None or pseen is None:
        return None
    g0, since = seen
    moves = ghost_moves_since(state, since)
    dg0, dp0 = dist_row(g0), dist_row(pseen[0])
    total = dg0[INDEX[pseen[0]]]
    if moves <= total:
        out = [i for i in range(len(CELLS)) if dg0[i] + dp0[i] <= total + 1 and abs(dg0[i] - moves) <= 2]
    else:
        out = [i for i in range(len(CELLS)) if dp0[i] <= moves - total + 1]
    me = state.pacman
    out = [i for i in out if abs(CELLS[i][0] - me[0]) + abs(CELLS[i][1] - me[1]) > state.vision]
    return out or None


def possible_ghost_cells(state):
    """Every cell the ghost could be on now: reachable from where it was last seen in
    the moves it has had since, and not within sight of Pac-Man."""
    me = state.pacman
    seen = state.ghost_last_seen
    if seen is not None:
        origin, since = seen
    else:
        origin, since = state.next("STOP", None).ghost, 1     # never seen: its start cell
    moves = ghost_moves_since(state, since) + 1
    d0 = DIST[INDEX[origin]]
    out = [i for i, cell in enumerate(CELLS)
           if d0[i] <= moves and abs(cell[0] - me[0]) + abs(cell[1] - me[1]) > state.vision]
    if not out:
        out = [i for i, cell in enumerate(CELLS)
               if abs(cell[0] - me[0]) + abs(cell[1] - me[1]) > state.vision]
    return out


def pacman_move(state):
    """Return 'N', 'S', 'E', 'W' or 'STOP'."""
    global trust_chaser, predicted, chase_misses, last_sight
    prepare(state)
    me = state.pacman

    sight, last_sight = last_sight, ((state.ghost, state.tick) if state.ghost is not None else None)
    if state.ghost is not None and sight is not None and sight[1] == state.tick - 1 \
            and (state.tick - 1) % 4 != 0:
        # last tick the ghost moved after us, seeing us where we stand now
        if dist_row(me)[INDEX[state.ghost]] != dist_row(me)[INDEX[sight[0]]] - 1:
            chase_misses += 1

    if state.ghost is not None:                        # the ghost is in sight
        if predicted is not None and predicted[1] == state.tick - 1:
            gi = INDEX[state.ghost]
            if min(DIST[gi][x] for x in predicted[0]) > 3:
                trust_chaser = False                   # it did not fall for it: play safe from now on
        predicted = None
        return lookahead(state, LOOKAHEAD_DEPTH)

    if not state.food:
        return "STOP"
    dp = dist_row(me)
    gt = ghost_ticks(state.tick)
    pac_tick = state.tick - 1

    chaser = chaser_ghost_cells(state) if trust_chaser else None
    ghosts = chaser if chaser is not None else possible_ghost_cells(state)
    predicted = (chaser, state.tick) if chaser is not None else None
    mouth_time = {}

    def ghost_time_to(m):
        """Earliest tick the ghost could stand on cell m."""
        if m not in mouth_time:
            dm = DIST[m]
            mouth_time[m] = ghost_time(gt, min(dm[x] for x in ghosts))
        return mouth_time[m]

    def target_key(f):
        i = INDEX[f]
        d = dp[i]
        if CORE[i]:
            risk = 0
        else:
            m = MOUTH[i]
            margin = ghost_time_to(m) - (pac_tick + d + DIST[i][m])
            risk = 0 if margin > 0 else 1 - margin      # unsafe pockets last, least bad first
        degree = sum(1 for x in state.legal_moves(f).values() if x in state.food)
        return (risk, d, degree, f)

    target = min(state.food, key=target_key)

    # steer around the ghost: with the decoy model, around where it should be now
    # (this is what brings Pac-Man back the other way round); without it, around
    # where it was seen recently
    danger = None
    if chaser is not None:
        near = [min(DIST[x][j] for x in chaser) for j in range(len(CELLS))]
        danger = (near, 3)
    else:
        seen = state.ghost_last_seen
        if seen is not None and state.tick - seen[1] <= 12:
            danger = (dist_row(seen[0]), state.tick - seen[1] + 2)

    def cost(a, b):
        if danger is not None and danger[0][INDEX[b]] <= danger[1]:
            return 6
        return 1

    def heuristic(cell):
        return state.manhattan(cell, target)          # admissible: every step costs >= 1

    path = astar(me, target, maze_neighbors(state), cost, heuristic)
    if path is None or len(path) < 2:
        return "STOP"
    return first_step(path)


def predict_pacman(state, cell, steps):
    """Where Pac-Man probably is after `steps` moves: walking toward its nearest food."""
    food = state.food
    for _ in range(steps):
        if not food:
            break
        d = dist_row(cell)
        target = min(food, key=lambda f: (d[INDEX[f]], f))
        if target == cell:
            break
        dt = dist_row(target)
        options = [x for x in state.legal_moves(cell).values() if x in INDEX]
        cell = min(options, key=lambda x: (dt[INDEX[x]], x))
        food = food - {cell}
    return cell


def guard_cell(state):
    """The cell with the smallest total maze distance to the remaining food."""
    key = state.food
    if key not in GUARD:
        idx = [INDEX[f] for f in state.food]
        GUARD[key] = CELLS[min(range(len(CELLS)), key=lambda c: (sum(DIST[c][i] for i in idx), c))]
    return GUARD[key]


def ghost_move(state):
    """Return 'N', 'S', 'E' or 'W' (the ghost must move)."""
    global prev_food, pac_estimate
    prepare(state)
    me = state.ghost
    legal = state.legal_moves(me)
    fallback = [m for m in ORDER if m in legal][0]

    # ---- where is Pac-Man?
    if state.pacman is not None:
        pac_estimate = (state.pacman, state.tick)
    else:
        vanished = (prev_food - state.food) if prev_food is not None else frozenset()
        if vanished:
            ref = pac_estimate[0] if pac_estimate else me
            cell = max(vanished, key=lambda f: (dist_row(ref)[INDEX[f]], f))   # the newest dot
            pac_estimate = (cell, state.tick)
        seen = state.pacman_last_seen
        if seen is not None and (pac_estimate is None or seen[1] > pac_estimate[1]):
            pac_estimate = seen
    prev_food = state.food

    if state.pacman is not None:                       # in sight: search
        path = astar(me, state.pacman, maze_neighbors(state), lambda a, b: 1,
                     lambda c: state.manhattan(c, state.pacman))
        if path is not None and len(path) == 2:
            return first_step(path)                    # catch it now
        return ghost_lookahead(state, LOOKAHEAD_DEPTH)

    if pac_estimate is None or state.tick - pac_estimate[1] > 15:
        target = guard_cell(state) if state.food else me   # trail is cold: guard the food
    else:
        cell, when = pac_estimate
        target = predict_pacman(state, cell, state.tick - when)
    if target == me:
        target = guard_cell(state) if state.food else me
        if target == me:
            return fallback

    path = astar(me, target, maze_neighbors(state), lambda a, b: 1,
                 lambda c: state.manhattan(c, target))
    if path is None or len(path) < 2:
        return fallback

    ti = INDEX[target]
    if not CORE[ti]:
        return first_step(path)                        # Pac-Man in a pocket: shortest way runs through the mouth
    if len(path) - 1 > 8:
        return first_step(path)

    # close, Pac-Man on the loops: corner it — fewest cells it reaches first, then distance
    dp = DIST[ti]
    best, best_key = None, None
    for m in ORDER:
        if m not in legal:
            continue
        dg = dist_row(legal[m])
        key = (territory(dp, dg) + 3 * dg[ti], m)
        if best_key is None or key < best_key:
            best, best_key = m, key
    return best
