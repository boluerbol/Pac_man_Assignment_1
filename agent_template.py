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

# "minimax"   : the ghost's reply is scored as the worst case for you
# "expectimax": the ghost's reply is scored as the average over its moves
OPPONENT_MODEL = "minimax"

LOOKAHEAD_DEPTH = 1          # rounds to look ahead; 1 round = your move + the ghost's reply

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
    raise NotImplementedError("write astar first — then run: python tests_astar.py agent_<surname>.py")


# --------------------------------------------------------------- 2. helpers you will want

def first_step(path):
    """The move ('N', 'S', 'E' or 'W') that takes you from path[0] to path[1]."""
    (r0, c0), (r1, c1) = path[0], path[1]
    return {v: k for k, v in STEP.items()}[(r1 - r0, c1 - c0)]


def maze_neighbors(state):
    """neighbors() for astar on this board: the cells you can step to."""
    return lambda cell: state.legal_moves(cell).values()


# ------------------------------------------------------------------ 3. evaluation function

def evaluate(state, pacman, ghost, food):
    """How good this position is for Pac-Man: a number, larger = better.

    `state` is a State (usually one produced by state.next(...)); pacman and ghost are
    positions, food is the set of food still on the board. A first version could count
    food eaten, subtract the distance to the nearest food and punish being close to the
    ghost. Use real maze distances rather than Manhattan: walls matter. Remember that
    pacman == ghost means you were caught."""
    raise NotImplementedError("write evaluate — it is what lookahead scores the leaves with")


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
    raise NotImplementedError("write lookahead — pacman_move must use it whenever state.ghost is not None")


# ---------------------------------------------------------------------- 5. the two movers

def pacman_move(state):
    """Return 'N', 'S', 'E', 'W' or 'STOP'."""
    me = state.pacman

    if state.ghost is not None:                        # the ghost is in sight
        return lookahead(state, LOOKAHEAD_DEPTH)

    # Out of sight: choose a target, a cost function and a heuristic, then A*.
    if not state.food:
        return "STOP"
    target = min(state.food, key=lambda f: state.manhattan(me, f))      # nearest dot — improve this

    def cost(a, b):
        return 1                                                        # every step costs 1 — add danger here

    def heuristic(cell):
        return state.manhattan(cell, target)                            # admissible: every step costs >= 1

    path = astar(me, target, maze_neighbors(state), cost, heuristic)
    if path is None or len(path) < 2:
        return "STOP"
    return first_step(path)


def ghost_move(state):
    """Return 'N', 'S', 'E' or 'W' (the ghost must move)."""
    me = state.ghost
    legal = state.legal_moves(me)

    if state.pacman is not None:                       # Pac-Man is in sight: chase
        target = state.pacman
    elif state.pacman_last_seen is not None:           # go where you last saw it — improve this
        target = state.pacman_last_seen[0]
    else:                                              # never seen: where will it be? near food
        target = min(state.food, key=lambda f: state.manhattan(me, f)) if state.food else me

    if target == me:
        return sorted(legal)[0]

    def cost(a, b):
        return 1

    def heuristic(cell):
        return state.manhattan(cell, target)

    path = astar(me, target, maze_neighbors(state), cost, heuristic)
    if path is None or len(path) < 2:
        return sorted(legal)[0]
    return first_step(path)


# ----------------------------------------------------------------------------- the state
#
#   state.rows, state.cols            board size
#   state.walls, state.food           frozensets of (row, col); food = food still on the board
#   state.pacman, state.ghost         positions (row, col); the OTHER player is None when out of sight
#   state.pacman_last_seen, state.ghost_last_seen   ((row, col), tick), or None if never seen
#   state.tick, state.max_ticks, state.vision       current tick (from 1), 300, 6
#   state.food_total, state.food_eaten
#   state.turn                        'pacman' or 'ghost' — whose move this is
#   state.ghost_can_move              False on ticks that are multiples of 4
#   state.legal_moves(cell)  -> {'N': (r, c), ...}   moves that do not hit a wall
#   state.manhattan(a, b)    -> |dr| + |dc|
#   state.next(pacman_move, ghost_move) -> the State one round ahead, with the rules applied
#       (the ghost's skip tick, eating, capture). If the other player is out of sight
#       the simulation puts it at its last seen cell — a guess. Illegal moves raise ValueError.
#   state.done, state.caught, state.outcome   game status (useful on states from next())
#
#   Module-level variables are allowed (e.g. to remember where the ghost was heading).
#   If you define new_game(state), the engine calls it once at the start of every game.
