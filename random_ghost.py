"""random_ghost.py — the house random ghost. Every move is a uniformly random legal move.
It is seeded from the board, so a given board always produces the same sequence of
choices: games stay reproducible. One of the league opponents of every Pac-Man."""
import random
import zlib

_rng = None


def new_game(state):
    """Called by the engine at the start of a game (optional hook)."""
    global _rng
    key = (state.rows, state.cols, tuple(sorted(state.walls)))
    seed = zlib.crc32(repr(key).encode("utf-8"))
    _rng = random.Random(seed)


def ghost_move(state):
    global _rng
    if _rng is None:
        new_game(state)
    moves = sorted(state.legal_moves(state.ghost))
    return _rng.choice(moves)


def pacman_move(state):
    # not used in the league; lets this file play either side in play.py
    global _rng
    if _rng is None:
        new_game(state)
    moves = sorted(state.legal_moves(state.pacman))
    return _rng.choice(moves)
