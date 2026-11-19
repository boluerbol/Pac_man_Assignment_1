"""
pacman_engine.py — The Pac-Man League game engine.
AI: Principles and Techniques · MDS · Ala-Too International University · Autumn 2026

Do not modify this file: the league runs with the official copy.

Rules (also in the assignment brief):
  * The game runs in ticks, starting at tick 1. In each tick Pac-Man moves first,
    then the ghost. Maximum MAX_TICKS ticks.
  * The ghost is slower: it does not move on ticks that are multiples of 4.
  * Moves: 'N' (row-1), 'S', 'E', 'W'. Pac-Man may also return 'STOP'. The ghost must move.
  * Pac-Man eats the food in every cell it enters.
  * If Pac-Man and the ghost are ever on the same cell, Pac-Man is caught; the game ends.
  * Vision: you always know the board and the food. You see the other player only when
    it is within VISION cells (Manhattan distance); otherwise you get its last seen
    position and the tick when you saw it.
  * Game over: all food eaten, Pac-Man caught, or MAX_TICKS reached.
  * Score: Pac-Man = fraction of food eaten (0..1); halved if caught. Ghost = 1 - Pac-Man.
  * Faults: an illegal move, a Python error, or a move slower than 100 ms is a fault
    (Pac-Man stops; the ghost stays). Three faults in one game = forfeit (score 0).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
import zlib

ENGINE_VERSION = "1.1"
MAX_TICKS = 300
VISION = 6
GHOST_SKIP_EVERY = 4          # ghost does not move when tick % 4 == 0
TIME_LIMIT = 0.100            # seconds per move
MAX_FAULTS = 3

MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
PACMAN_MOVES = ("N", "S", "E", "W", "STOP")
GHOST_MOVES = ("N", "S", "E", "W")


# ----------------------------------------------------------------------------- boards

class Board:
    """Static part of a game: walls, initial food and start cells.
    Symbols: '#' wall · '.' floor with food · ' ' floor without food · 'P' / 'G' starts."""

    def __init__(self, text: str, name: str = "board"):
        lines = [ln.rstrip("\n") for ln in text.strip("\n").splitlines()]
        lines = [ln for ln in lines if ln.strip() != ""]
        width = max(len(ln) for ln in lines)
        lines = [ln.ljust(width) for ln in lines]
        self.name = name
        self.text = "\n".join(lines)
        self.rows = len(lines)
        self.cols = width
        walls, food = set(), set()
        pac = ghost = None
        for r, ln in enumerate(lines):
            for c, ch in enumerate(ln):
                if ch == "#":
                    walls.add((r, c))
                elif ch == ".":
                    food.add((r, c))
                elif ch == "P":
                    pac = (r, c)
                elif ch == "G":
                    ghost = (r, c)
                elif ch == " ":
                    pass
                else:
                    raise ValueError(f"{name}: unknown symbol {ch!r} at row {r}, col {c}")
        # the outside of the board counts as wall
        for r in range(self.rows):
            walls.add((r, -1)); walls.add((r, self.cols))
        for c in range(-1, self.cols + 1):
            walls.add((-1, c)); walls.add((self.rows, c))
        if pac is None or ghost is None:
            raise ValueError(f"{name}: the board needs exactly one P and one G")
        if not food:
            raise ValueError(f"{name}: the board has no food")
        self.walls = frozenset(walls)
        self.food = frozenset(food)
        self.pacman_start = pac
        self.ghost_start = ghost

    @classmethod
    def load(cls, path: str) -> "Board":
        with open(path, encoding="utf-8") as f:
            text = f.read()
        name = os.path.splitext(os.path.basename(path))[0]
        return cls(text, name)

    def seed(self) -> int:
        """A stable number derived from the board text (used by the random ghost)."""
        return zlib.crc32(self.text.encode("utf-8"))


# ------------------------------------------------------------------------------ rules

class Core:
    """The dynamic part of a game. All rules live here; State.next() and the Game
    loop both use it, so a simulated round and a real one follow the same rules."""

    __slots__ = ("board", "pacman", "ghost", "food", "eaten", "tick", "turn", "caught", "outcome", "max_ticks")

    def __init__(self, board: Board, max_ticks: int = MAX_TICKS):
        self.board = board
        self.max_ticks = max_ticks
        self.pacman = board.pacman_start
        self.ghost = board.ghost_start
        self.food = board.food
        self.eaten = 0
        self.tick = 1
        self.turn = "pacman"
        self.caught = False
        self.outcome = None        # None while running; else cleared/caught/timeout/forfeit_*

    def copy(self) -> "Core":
        c = Core.__new__(Core)
        for k in Core.__slots__:
            setattr(c, k, getattr(self, k))
        return c

    @property
    def done(self) -> bool:
        return self.outcome is not None

    def ghost_can_move(self) -> bool:
        return self.tick % GHOST_SKIP_EVERY != 0

    def legal_moves(self, cell):
        r, c = cell
        out = {}
        for m, (dr, dc) in MOVES.items():
            nxt = (r + dr, c + dc)
            if nxt not in self.board.walls:
                out[m] = nxt
        return out

    def _check_end_after_move(self):
        if self.pacman == self.ghost:
            self.caught = True
            self.outcome = "caught"
        elif not self.food:
            self.outcome = "cleared"

    def apply(self, move):
        """Apply one turn for self.turn and advance. `move` None means 'stay'.
        Raises ValueError on an illegal move (the Game loop turns that into a fault)."""
        if self.done:
            return
        if self.turn == "pacman":
            if move not in (None, "STOP"):
                legal = self.legal_moves(self.pacman)
                if move not in legal:
                    raise ValueError(f"illegal Pac-Man move {move!r}")
                self.pacman = legal[move]
                if self.pacman in self.food:
                    self.food = self.food - {self.pacman}
                    self.eaten += 1
            self._check_end_after_move()
            self.turn = "ghost"
        else:
            if self.ghost_can_move() and move is not None:
                legal = self.legal_moves(self.ghost)
                if move not in legal:
                    raise ValueError(f"illegal ghost move {move!r}")
                self.ghost = legal[move]
            self._check_end_after_move()
            self.turn = "pacman"
            self.tick += 1
            if not self.done and self.tick > self.max_ticks:
                self.outcome = "timeout"

    def scores(self):
        if self.outcome == "forfeit_pacman":
            return 0.0, 1.0
        if self.outcome == "forfeit_ghost":
            return 1.0, 0.0
        p = self.eaten / len(self.board.food)
        if self.caught:
            p /= 2
        return p, 1.0 - p


# ------------------------------------------------------------------------------ state

class State:
    """What an agent sees. Read-only: changing attributes has no effect on the game.

    rows, cols                      board size
    walls, food                     frozensets of (row, col); food = food still on the board
    pacman, ghost                   positions (row, col); the OTHER player is None when out of sight
    pacman_last_seen, ghost_last_seen   ((row, col), tick) or None
    tick, max_ticks, vision         current tick (from 1), tick limit, vision radius
    food_total, food_eaten          counts
    turn                            'pacman' or 'ghost': whose move this is
    ghost_can_move                  False on ticks that are multiples of 4
    caught, done, outcome           game status (always running in states handed to agents)

    legal_moves(cell)   -> {'N': (r, c), ...} moves that do not hit a wall
    manhattan(a, b)     -> |dr| + |dc|
    next(pacman_move, ghost_move) -> State one full round ahead, rules applied
    """

    def __init__(self, core: Core, side: str, last_seen: dict, simulated: bool = False):
        b = core.board
        self.rows, self.cols = b.rows, b.cols
        self.walls, self.food = b.walls, core.food
        self.tick, self.max_ticks, self.vision = core.tick, core.max_ticks, VISION
        self.food_total, self.food_eaten = len(b.food), core.eaten
        self.turn = core.turn
        self.ghost_can_move = core.ghost_can_move()
        self.caught, self.done, self.outcome = core.caught, core.done, core.outcome
        self._side = side
        self._last_seen = last_seen
        visible = abs(core.pacman[0] - core.ghost[0]) + abs(core.pacman[1] - core.ghost[1]) <= VISION
        self.visible = visible
        if simulated:
            # positions in a simulated state are the model's positions: both known
            self.pacman, self.ghost = core.pacman, core.ghost
            self._core = core
        else:
            if visible:
                last_seen["pacman"] = (core.pacman, core.tick)
                last_seen["ghost"] = (core.ghost, core.tick)
                self.pacman, self.ghost = core.pacman, core.ghost
                self._core = core.copy()
            else:
                # fog: the state keeps no copy of the other player's true position.
                # The internal copy used by next() places it at its last seen cell
                # (or its start cell if never seen) — a guess, not knowledge.
                fogged = core.copy()
                if side == "pacman":
                    self.pacman, self.ghost = core.pacman, None
                    seen = last_seen.get("ghost")
                    fogged.ghost = seen[0] if seen else b.ghost_start
                else:
                    self.pacman, self.ghost = None, core.ghost
                    seen = last_seen.get("pacman")
                    fogged.pacman = seen[0] if seen else b.pacman_start
                self._core = fogged
        self.pacman_last_seen = last_seen.get("pacman")
        self.ghost_last_seen = last_seen.get("ghost")

    def legal_moves(self, cell):
        return self._core.legal_moves(cell)

    @staticmethod
    def manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def next(self, pacman_move=None, ghost_move=None) -> "State":
        """One full round ahead from this state, with the rules applied.

        At Pac-Man's turn: Pac-Man plays pacman_move this tick, then the ghost plays
        ghost_move (if it moves this tick), and the result is Pac-Man's turn next tick.
        At the ghost's turn: the ghost plays ghost_move, then Pac-Man plays pacman_move
        next tick, and the result is the ghost's turn.
        A move given as None means 'stay'; on the ghost's skip tick its move is ignored.
        If the other player is out of sight, the simulation places it at its last seen
        cell (or its start cell if never seen): a guess, not knowledge. In the returned
        state both positions are the model's positions. Illegal moves raise ValueError."""
        core = self._core.copy()
        if core.done:
            return State(core, self._side, dict(self._last_seen), simulated=True)
        if self.turn == "pacman":
            core.apply(pacman_move)
            if not core.done:
                core.apply(ghost_move)
        else:
            core.apply(ghost_move)
            if not core.done:
                core.apply(pacman_move)
        return State(core, self._side, dict(self._last_seen), simulated=True)

    def __repr__(self):
        return (f"State(tick={self.tick}, turn={self.turn}, pacman={self.pacman}, ghost={self.ghost}, "
                f"food_left={len(self.food)})")


# ------------------------------------------------------------------------------ agents

def load_agent(path: str, name: str | None = None):
    """Import an agent file as a fresh module (module-level variables start clean)."""
    path = os.path.abspath(path)
    modname = name or ("agent_" + str(abs(zlib.crc32(path.encode()))) + "_" + str(time.time_ns()))
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    mod.__agent_path__ = path
    return mod


def agent_label(agent) -> str:
    if agent is None:
        return "human"
    p = getattr(agent, "__agent_path__", None) or getattr(agent, "__file__", "agent")
    return os.path.splitext(os.path.basename(p))[0]


# -------------------------------------------------------------------------------- game

class Game:
    """One game. Agents are modules with pacman_move(state) / ghost_move(state), or None
    for a human player (then turn(move) must be given the move).

    game = Game(board, pacman_agent, ghost_agent)
    game.run()                 # play to the end (both sides must be agents)
    while not game.done: game.turn(move_or_None)   # step by step (arena)
    game.result(), game.replay()
    """

    def __init__(self, board: Board, pacman_agent=None, ghost_agent=None,
                 time_limit: float = TIME_LIMIT, labels=None, meta=None, max_ticks: int | None = None):
        """max_ticks=None means the league limit (MAX_TICKS); the arena passes a huge number
        for games with a human player, which then end only by capture or by eating everything."""
        self.board = board
        self.agents = {"pacman": pacman_agent, "ghost": ghost_agent}
        self.labels = labels or {"pacman": agent_label(pacman_agent), "ghost": agent_label(ghost_agent)}
        self.time_limit = time_limit
        self.core = Core(board, max_ticks or MAX_TICKS)
        self.last_seen = {}
        self.faults = {"pacman": 0, "ghost": 0}
        self.fault_log = {"pacman": [], "ghost": []}
        self.turns = []
        self.meta = dict(meta or {})
        for side, ag in self.agents.items():
            if ag is not None and hasattr(ag, "new_game"):
                try:
                    ag.new_game(self.state_for(side))
                except Exception:
                    pass

    @property
    def done(self):
        return self.core.done

    @property
    def turn_side(self):
        return self.core.turn

    def side_can_move(self, side=None):
        side = side or self.core.turn
        return True if side == "pacman" else self.core.ghost_can_move()

    def needs_human_move(self):
        """True when the side to move is a human who can actually move this turn."""
        return (not self.done and self.agents[self.core.turn] is None and self.side_can_move())

    def state_for(self, side) -> State:
        return State(self.core, side, self.last_seen)

    def view(self) -> dict:
        """Full-information snapshot for viewers (JSON friendly)."""
        c = self.core
        p, g = c.scores() if c.done else (None, None)
        return {
            "tick": c.tick, "turn": c.turn, "pacman": list(c.pacman), "ghost": list(c.ghost),
            "food": sorted(list(x) for x in c.food), "eaten": c.eaten, "total": len(self.board.food),
            "ghost_can_move": c.ghost_can_move(),
            "visible": abs(c.pacman[0] - c.ghost[0]) + abs(c.pacman[1] - c.ghost[1]) <= VISION,
            "done": c.done, "outcome": c.outcome, "caught": c.caught,
            "faults": dict(self.faults), "pacman_score": p, "ghost_score": g,
        }

    def turn(self, move=None) -> dict:
        """Advance one turn. For an agent side the move is asked from the agent; for a human
        side it must be passed in (ignored if that side cannot move this turn)."""
        if self.done:
            return {"done": True}
        side = self.core.turn
        tick = self.core.tick
        can = self.side_can_move(side)
        frm = getattr(self.core, side)
        event = {"tick": tick, "who": side, "from": list(frm), "move": None, "ms": 0.0,
                 "fault": None, "skipped": not can, "ate": False}
        agent = self.agents[side]
        chosen = None
        if can:
            if agent is None:
                chosen = move
                if chosen is None:
                    raise RuntimeError("human move required")
            else:
                state = self.state_for(side)
                fn = agent.pacman_move if side == "pacman" else agent.ghost_move
                t0 = time.perf_counter()
                try:
                    chosen = fn(state)
                    ms = (time.perf_counter() - t0) * 1000.0
                    event["ms"] = round(ms, 2)
                    if ms > self.time_limit * 1000.0:
                        self._fault(side, f"too slow: {ms:.0f} ms", event)
                        chosen = None
                except Exception as e:  # noqa: BLE001 - any error in student code is a fault
                    event["ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                    self._fault(side, "error: " + "".join(traceback.format_exception_only(type(e), e)).strip(), event)
                    chosen = None
            if event["fault"] is None:
                allowed = PACMAN_MOVES if side == "pacman" else GHOST_MOVES
                legal = self.core.legal_moves(frm)
                if chosen not in allowed or (chosen != "STOP" and chosen not in legal):
                    self._fault(side, f"illegal move {chosen!r}", event)
                    chosen = None
            if self.core.done:            # forfeit happened inside _fault
                event["to"] = list(getattr(self.core, side))
                self.turns.append(event)
                return event
            if chosen is None:
                chosen = "STOP" if side == "pacman" else None
            event["move"] = chosen
        food_before = len(self.core.food)
        self.core.apply(chosen if can else None)
        event["to"] = list(getattr(self.core, side))
        event["ate"] = len(self.core.food) < food_before
        event["outcome"] = self.core.outcome
        self.turns.append(event)
        return event

    def _fault(self, side, why, event):
        self.faults[side] += 1
        event["fault"] = why
        self.fault_log[side].append((self.core.tick, why))
        if self.faults[side] >= MAX_FAULTS:
            self.core.outcome = "forfeit_" + side
            event["outcome"] = self.core.outcome

    def run(self, max_turns=None):
        n = 0
        while not self.done:
            self.turn()
            n += 1
            if max_turns and n >= max_turns:
                break
        return self.result()

    def result(self) -> dict:
        p, g = self.core.scores()
        return {
            "outcome": self.core.outcome, "pacman_score": round(p, 4), "ghost_score": round(g, 4),
            "eaten": self.core.eaten, "total": len(self.board.food),
            "ticks": min(self.core.tick, self.core.max_ticks),
            "faults": dict(self.faults), "fault_log": {k: v[:5] for k, v in self.fault_log.items()},
            "pacman": self.labels["pacman"], "ghost": self.labels["ghost"], "board": self.board.name,
        }

    def result_line(self) -> str:
        r = self.result()
        extra = ""
        if r["faults"]["pacman"] or r["faults"]["ghost"]:
            extra = f" faults=P{r['faults']['pacman']}/G{r['faults']['ghost']}"
            for side in ("pacman", "ghost"):
                if r["fault_log"][side]:
                    t, why = r["fault_log"][side][0]
                    extra += f" [{side} tick {t}: {why}]"
        return (f"RESULT: {r['outcome']} Pac-Man {r['pacman_score']:.2f} ghost {r['ghost_score']:.2f} "
                f"({r['eaten']}/{r['total']} eaten in {r['ticks']} ticks){extra}")

    def replay(self) -> dict:
        b = self.board
        return {
            "engine": ENGINE_VERSION,
            "meta": {"board": b.name, "pacman": self.labels["pacman"], "ghost": self.labels["ghost"], **self.meta},
            "board": {"name": b.name, "text": b.text, "rows": b.rows, "cols": b.cols,
                      "walls": sorted(list(x) for x in b.walls if 0 <= x[0] < b.rows and 0 <= x[1] < b.cols),
                      "food": sorted(list(x) for x in b.food),
                      "pacman": list(b.pacman_start), "ghost": list(b.ghost_start)},
            "rules": {"max_ticks": self.core.max_ticks, "vision": VISION, "ghost_skip_every": GHOST_SKIP_EVERY,
                      "time_limit_ms": self.time_limit * 1000},
            "turns": self.turns,
            "result": self.result(),
        }

    def save_replay(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.replay(), f)


# ---------------------------------------------------------------------------- rendering

def render_text(game: Game, fog_for=None) -> str:
    """ASCII picture of the current position. fog_for='pacman'|'ghost' hides the other
    player when out of sight (what that side actually sees)."""
    c = game.core
    b = game.board
    rows = []
    visible = abs(c.pacman[0] - c.ghost[0]) + abs(c.pacman[1] - c.ghost[1]) <= VISION
    for r in range(b.rows):
        line = []
        for col in range(b.cols):
            cell = (r, col)
            if cell == c.pacman and not (fog_for == "ghost" and not visible):
                ch = "P"
            elif cell == c.ghost and not (fog_for == "pacman" and not visible):
                ch = "G"
            elif cell in b.walls:
                ch = "#"
            elif cell in c.food:
                ch = "."
            else:
                ch = " "
            line.append(ch)
        rows.append("".join(line))
    status = (f"tick {min(c.tick, c.max_ticks):3d}  eaten {c.eaten}/{len(b.food)}  "
              f"faults P{game.faults['pacman']}/G{game.faults['ghost']}")
    return "\n".join(rows) + "\n" + status
