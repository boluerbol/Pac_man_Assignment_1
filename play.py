#!/usr/bin/env python3
"""play.py — play one game of the Pac-Man League.

  python play.py --pacman agent_karimova.py --ghost baseline_ghost.py --board boards/practice1.txt --render
  python play.py --pacman baseline_pacman.py --ghost agent_karimova.py --board boards/practice2.txt
  python play.py ... --record game.json        # then open the replay in arena.html
  python play.py ... --fog pacman              # render only what Pac-Man can see

Any agent file with pacman_move / ghost_move can play either side.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pacman_engine import Board, Game, load_agent, render_text  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pacman", required=True, help="agent file that plays Pac-Man")
    ap.add_argument("--ghost", required=True, help="agent file that plays the ghost")
    ap.add_argument("--board", required=True, help="board file")
    ap.add_argument("--render", action="store_true", help="animate the game in the terminal")
    ap.add_argument("--delay", type=float, default=0.08, help="seconds between frames when rendering")
    ap.add_argument("--fog", choices=["pacman", "ghost"], help="render what this side sees")
    ap.add_argument("--record", metavar="FILE", help="save the replay as JSON")
    ap.add_argument("--json", action="store_true", help="print the result as JSON (for scripts)")
    ap.add_argument("--time-limit", type=float, default=0.100, help="seconds per move (league: 0.1)")
    args = ap.parse_args(argv)

    board = Board.load(args.board)
    pac = load_agent(args.pacman)
    gho = load_agent(args.ghost)
    game = Game(board, pac, gho, time_limit=args.time_limit)

    if args.render:
        try:
            while not game.done:
                game.turn()
                if game.turn_side == "pacman":          # draw once per tick
                    sys.stdout.write("\033[2J\033[H" + render_text(game, args.fog) + "\n")
                    sys.stdout.flush()
                    time.sleep(args.delay)
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 1
        sys.stdout.write("\033[2J\033[H" + render_text(game, args.fog) + "\n")
    else:
        game.run()

    if args.record:
        game.save_replay(args.record)
    if args.json:
        print(json.dumps(game.result()))
    else:
        print(game.result_line())
        if args.record:
            print(f"replay saved to {args.record} — open arena.html and drop it there")
    return 0


if __name__ == "__main__":
    sys.exit(main())
