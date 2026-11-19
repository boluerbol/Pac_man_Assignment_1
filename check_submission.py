#!/usr/bin/env python3
"""check_submission.py — run this before you submit. It runs every check the league runs.

  python check_submission.py agent_<surname>.py
  python check_submission.py agent_<surname>.py --boards boards --json report.json

Checks: file name · imports · the five functions and OPPONENT_MODEL · the ten A* tests ·
both movers call your astar · lookahead is used on every tick the ghost is in sight ·
speed (100 ms per move) · no faults. Then it plays your agent against the house agents on
the practice boards and prints the results table for your report.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pacman_engine import Board, Game, load_agent  # noqa: E402
import tests_astar  # noqa: E402

ALLOWED_IMPORTS = {"math", "heapq", "collections", "itertools", "functools", "dataclasses", "typing", "time"}
REQUIRED = ["astar", "evaluate", "lookahead", "pacman_move", "ghost_move"]
MODELS = {"minimax", "expectimax"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "__import__", "compile", "globals", "setattr", "delattr"}
HOUSE = {"baseline_pacman": "baseline_pacman.py", "baseline_ghost": "baseline_ghost.py",
         "random_ghost": "random_ghost.py"}


class Report:
    def __init__(self):
        self.lines = []
        self.ok = True
        self.data = {"checks": [], "games": []}

    def check(self, passed, text, warn=False):
        tag = "OK " if passed else ("WARN" if warn else "FAIL")
        if not passed and not warn:
            self.ok = False
        self.lines.append(f"[{tag}] {text}")
        self.data["checks"].append({"ok": bool(passed), "warn": warn and not passed, "text": text})

    def note(self, text):
        self.lines.append("       " + text)


def static_checks(path, rep):
    name = os.path.basename(path)
    rep.check(re.fullmatch(r"agent_[a-z0-9_]+\.py", name) is not None,
              f"file name {name}" + ("" if re.fullmatch(r"agent_[a-z0-9_]+\.py", name) else
                                     "  (must be agent_<surname>.py, lowercase, no spaces)"))
    src = open(path, encoding="utf-8").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep.check(False, f"syntax error: {e}")
        return None
    imports, bad_attr, bad_names = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_") and node.attr not in ("__name__", "__doc__"):
            bad_attr.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad_names.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_NAMES:
            bad_names.add(node.func.id)
    extra = sorted(imports - ALLOWED_IMPORTS)
    rep.check(not extra, "imports: " + (", ".join(sorted(imports)) or "none") +
              (f"  — not allowed: {', '.join(extra)}" if extra else ""))
    rep.check(not bad_attr, "no access to private attributes" + (f"  — found {', '.join(sorted(bad_attr))}" if bad_attr else ""))
    rep.check(not bad_names, "no file/exec tricks" + (f"  — found {', '.join(sorted(bad_names))}" if bad_names else ""))
    return tree


def instrument(mod, counters):
    """Wrap astar / lookahead / the movers so we can see who calls what."""
    real_astar, real_look = mod.astar, mod.lookahead
    real_pac, real_ghost = mod.pacman_move, mod.ghost_move
    ctx = {"side": None}

    def astar(*a, **k):
        counters["astar_" + str(ctx["side"])] += 1
        return real_astar(*a, **k)

    def lookahead(*a, **k):
        counters["lookahead_calls"] += 1
        counters["_look_this_turn"] = True
        return real_look(*a, **k)

    def pacman_move(state):
        ctx["side"] = "pacman"
        counters["_look_this_turn"] = False
        visible = state.ghost is not None
        try:
            return real_pac(state)
        finally:
            if visible:
                counters["visible_ticks"] += 1
                if counters["_look_this_turn"]:
                    counters["lookahead_on_visible"] += 1

    def ghost_move(state):
        ctx["side"] = "ghost"
        return real_ghost(state)

    mod.astar, mod.lookahead, mod.pacman_move, mod.ghost_move = astar, lookahead, pacman_move, ghost_move


def play(board_path, pac_path, ghost_path, counters, student_side):
    board = Board.load(board_path)
    kit = os.path.dirname(os.path.abspath(__file__))
    if student_side == "pacman":
        pac = load_agent(pac_path)
        instrument(pac, counters)
        gho = load_agent(os.path.join(kit, ghost_path))
    else:
        pac = load_agent(os.path.join(kit, pac_path))
        gho = load_agent(ghost_path)
        instrument(gho, counters)
    game = Game(board, pac, gho)
    game.run()
    r = game.result()
    ms = [e["ms"] for e in game.turns if e["who"] == student_side and not e["skipped"]]
    r["max_ms"] = max(ms) if ms else 0.0
    r["student_side"] = student_side
    r["student_faults"] = r["faults"][student_side]
    r["student_fault_log"] = r["fault_log"][student_side]
    return r


def fmt_result(r):
    if r["outcome"] == "cleared":
        how = "cleared the board"
    elif r["outcome"] == "caught":
        how = f"caught at tick {r['ticks']}"
    elif r["outcome"] == "timeout":
        how = "survived 300 ticks"
    else:
        how = r["outcome"].replace("_", " ")
    return f"{how}, {r['eaten']}/{r['total']} eaten"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent")
    ap.add_argument("--boards", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "boards"),
                    help="folder with the boards to play (default: the practice boards)")
    ap.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rep = Report()
    path = os.path.abspath(args.agent)
    if not os.path.exists(path):
        print(f"[FAIL] no such file: {path}")
        return 1
    tree = static_checks(path, rep)
    if tree is None:
        print("\n".join(rep.lines)); return 1

    try:
        mod = load_agent(path)
    except Exception as e:  # noqa: BLE001
        rep.check(False, f"the file does not import: {type(e).__name__}: {e}")
        print("\n".join(rep.lines)); return 1

    missing = [f for f in REQUIRED if not callable(getattr(mod, f, None))]
    rep.check(not missing, "functions: " + ", ".join(REQUIRED) + (f"  — missing: {', '.join(missing)}" if missing else ""))
    model = getattr(mod, "OPPONENT_MODEL", None)
    rep.check(model in MODELS, f"OPPONENT_MODEL = {model!r}" + ("" if model in MODELS else '  (must be "minimax" or "expectimax")'))
    rep.data["opponent_model"] = model

    # A* tests
    if callable(getattr(mod, "astar", None)):
        passed, total, lines = tests_astar.run_tests(mod.astar)
        rep.check(passed == total, f"A* tests: {passed}/{total}")
        for ln in lines:
            if ln.startswith("[FAIL"):
                rep.note(ln)
        rep.data["astar_tests"] = [passed, total]
    if missing:
        print("\n".join(rep.lines)); return 1

    # games against the house agents
    boards = sorted(glob.glob(os.path.join(args.boards, "*.txt")))
    if not boards:
        rep.check(False, f"no boards found in {args.boards}")
        print("\n".join(rep.lines)); return 1
    counters = {"astar_pacman": 0, "astar_ghost": 0, "lookahead_calls": 0, "visible_ticks": 0,
                "lookahead_on_visible": 0, "_look_this_turn": False}
    table = []
    worst_ms, faults, fault_examples = 0.0, 0, []
    for b in boards:
        row = {"board": os.path.splitext(os.path.basename(b))[0]}
        for key, (pac, gho, side) in {
            "pac_vs_baseline": (path, HOUSE["baseline_ghost"], "pacman"),
            "pac_vs_random": (path, HOUSE["random_ghost"], "pacman"),
            "ghost_vs_baseline": (HOUSE["baseline_pacman"], path, "ghost"),
        }.items():
            r = play(b, pac, gho, counters, side)
            row[key] = r
            worst_ms = max(worst_ms, r["max_ms"])
            faults += r["student_faults"]
            fault_examples += [f"{row['board']} as {side}, tick {t}: {why}" for t, why in r["student_fault_log"]]
            rep.data["games"].append({"board": row["board"], "game": key, "pacman": r["pacman"], "ghost": r["ghost"],
                                      "outcome": r["outcome"], "pacman_score": r["pacman_score"],
                                      "ghost_score": r["ghost_score"], "eaten": r["eaten"], "total": r["total"],
                                      "ticks": r["ticks"], "max_ms": r["max_ms"], "student_faults": r["student_faults"]})
        table.append(row)

    rep.check(counters["astar_pacman"] > 0, f"pacman_move calls your astar ({counters['astar_pacman']} calls)")
    rep.check(counters["astar_ghost"] > 0, f"ghost_move calls your astar ({counters['astar_ghost']} calls)")
    vt, lv = counters["visible_ticks"], counters["lookahead_on_visible"]
    rep.check(vt > 0 and lv == vt, f"lookahead used on {lv}/{vt} ticks with the ghost in sight" +
              ("" if vt else "  (the ghost was never in sight — check your boards)"))
    rep.check(worst_ms <= 100.0, f"slowest move {worst_ms:.0f} ms (limit 100)")
    rep.check(faults == 0, f"faults: {faults}" + ("" if not faults else "  — " + "; ".join(fault_examples[:3])))
    rep.data["counters"] = {k: v for k, v in counters.items() if not k.startswith("_")}
    rep.data["worst_ms"] = worst_ms
    rep.data["faults"] = faults

    # beats the house?
    def beats(key, side):
        wins = sum(1 for row in table if (row[key]["pacman_score"] > 0.5) == (side == "pacman")
                   and row[key]["pacman_score"] != 0.5)
        return wins
    n = len(table)
    w1, w2, w3 = beats("pac_vs_baseline", "pacman"), beats("pac_vs_random", "pacman"), beats("ghost_vs_baseline", "ghost")
    rep.check(w1 == n, f"your Pac-Man beats the baseline ghost on {w1}/{n} boards", warn=True)
    rep.check(w2 == n, f"your Pac-Man beats the random ghost on {w2}/{n} boards", warn=True)
    rep.check(w3 == n, f"your ghost beats the baseline Pac-Man on {w3}/{n} boards", warn=True)
    rep.data["house"] = {"pac_vs_baseline": [w1, n], "pac_vs_random": [w2, n], "ghost_vs_baseline": [w3, n]}

    out = list(rep.lines)
    out.append("")
    out.append("Results against the house agents (copy this table into your report):")
    out.append(f"{'board':<11} {'as Pac-Man vs baseline ghost':<42} {'as Pac-Man vs random ghost':<42} {'as ghost vs baseline Pac-Man':<42}")
    for row in table:
        a, b, c = row["pac_vs_baseline"], row["pac_vs_random"], row["ghost_vs_baseline"]
        out.append(f"{row['board']:<11} {a['pacman_score']:.2f}  {fmt_result(a):<36} {b['pacman_score']:.2f}  {fmt_result(b):<36} {c['ghost_score']:.2f}  {fmt_result(c):<36}")
    out.append("")
    out.append("ALL CHECKS PASSED — your file will work in the league." if rep.ok else
               "SOME CHECKS FAILED — fix the [FAIL] lines above before you submit.")
    rep.data["all_ok"] = rep.ok
    rep.data["table_text"] = "\n".join(out)
    if not args.quiet:
        print("\n".join(out))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rep.data, f, indent=1)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
