#!/usr/bin/env python3
"""arena.py — the Pac-Man League arena: play live (any agent, or YOU, on either side) and
watch replays, in your browser.

  python arena.py                      # opens http://localhost:8000

Play: Pac-Men are listed on the left, ghosts on the right — every agent_*.py in this
folder (shown by surname), the house agents, and "you". Pick one on each side and a board,
press Start. The game runs in real time (default 120 ms per tick): hold an arrow key to
move the side you play, release it to stop; with two humans the arrows take Pac-Man and
W A S D the ghost; P pauses. The rules are the league rules (the ghost sits out every 4th
tick, vision 6, 100 ms per agent move) — except that a game with a human player has no tick
limit: it ends when Pac-Man is caught or the food is gone. Fog shows the board as one side sees
it. Every game can be downloaded as a replay.

Replay: open replay files, or browse a replays folder; pick the Pac-Man owner on the left,
the ghost owner on the right, the board on top, and scrub through the game.

Instructor: run from the kit folder inside the Assignment folder and the arena finds
../Submission (students' files) by itself. The hidden boards and the league replays are
never loaded automatically — on league day pass them explicitly:
  python arena.py --boards ../instructor/boards_hidden --replays ../instructor/league_out/replays
"""
import argparse
import glob
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pacman_engine import Board, Game, load_agent  # noqa: E402

HOUSE = {"baseline_pacman": "pacman", "baseline_ghost": "ghost", "random_ghost": "ghost"}


def label_of(name):
    return name[len("agent_"):] if name.startswith("agent_") else name


class Arena:
    def __init__(self, agent_dirs, board_dirs, replay_dirs):
        self.agent_dirs = agent_dirs
        self.board_dirs = board_dirs
        self.replay_dirs = replay_dirs
        self.games = {}
        self.lock = threading.Lock()
        self.next_id = 1

    # -- catalogue
    def agents(self):
        """name -> {path, label, side, kind}"""
        out = {}
        for d in self.agent_dirs:
            for p in sorted(glob.glob(os.path.join(d, "agent_*.py"))):
                name = os.path.splitext(os.path.basename(p))[0]
                if name == "agent_template" or name in out:
                    continue
                out[name] = {"path": p, "label": label_of(name), "side": "both", "kind": "student"}
        for h, side in HOUSE.items():
            p = os.path.join(HERE, h + ".py")
            if os.path.exists(p):
                out[h] = {"path": p, "label": h, "side": side, "kind": "house"}
        return out

    def boards(self):
        out = {}
        for d in self.board_dirs:
            for p in sorted(glob.glob(os.path.join(d, "*.txt"))):
                out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
        return out

    def replays(self):
        items = []
        for d in self.replay_dirs:
            idx = os.path.join(d, "index.json")
            if os.path.exists(idx):
                try:
                    for it in json.load(open(idx, encoding="utf-8")):
                        it = dict(it)
                        it["path"] = os.path.join(d, it["file"])
                        items.append(it)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            for p in sorted(glob.glob(os.path.join(d, "*.json"))):
                try:
                    with open(p, encoding="utf-8") as f:
                        head = json.load(f)
                    m, r = head.get("meta", {}), head.get("result", {})
                    items.append({"file": os.path.basename(p), "path": p, "pacman": m.get("pacman"),
                                  "ghost": m.get("ghost"), "board": m.get("board"), "outcome": r.get("outcome"),
                                  "pacman_score": r.get("pacman_score"), "ghost_score": r.get("ghost_score")})
                except Exception:  # noqa: BLE001
                    continue
        return items

    # -- games
    def new_game(self, pac_name, ghost_name, board_name, time_limit=0.1):
        agents, boards = self.agents(), self.boards()
        if board_name not in boards:
            raise ValueError(f"unknown board {board_name}")
        board = Board.load(boards[board_name])
        pac = None if pac_name == "human" else load_agent(agents[pac_name]["path"])
        gho = None if ghost_name == "human" else load_agent(agents[ghost_name]["path"])
        labels = {"pacman": "you" if pac_name == "human" else label_of(pac_name),
                  "ghost": "you" if ghost_name == "human" else label_of(ghost_name)}
        human = pac is None or gho is None
        # with a human playing there is no tick limit: the game ends by capture or when the food is gone
        game = Game(board, pac, gho, time_limit=time_limit, labels=labels, meta={"arena": True},
                    max_ticks=10 ** 9 if human else None)
        with self.lock:
            gid = str(self.next_id)
            self.next_id += 1
            self.games[gid] = game
        return gid, game

    def payload(self, game, event=None):
        return {"view": game.view(), "event": event, "done": game.done,
                "needs": game.turn_side if game.needs_human_move() else None,
                "result": game.result() if game.done else None,
                "result_line": game.result_line() if game.done else None}


def board_json(board):
    return {"name": board.name, "rows": board.rows, "cols": board.cols, "text": board.text,
            "walls": sorted(list(x) for x in board.walls if 0 <= x[0] < board.rows and 0 <= x[1] < board.cols),
            "food": sorted(list(x) for x in board.food),
            "pacman": list(board.pacman_start), "ghost": list(board.ghost_start)}


def make_handler(arena: Arena):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/arena.html", "/index.html"):
                    with open(os.path.join(HERE, "arena.html"), "rb") as f:
                        return self._send(200, f.read(), "text/html")
                if u.path == "/api/list":
                    ag = arena.agents()
                    return self._send(200, {"agents": [{"name": n, "label": v["label"], "side": v["side"], "kind": v["kind"]}
                                                       for n, v in sorted(ag.items(), key=lambda kv: (kv[1]["kind"] != "student", kv[1]["label"]))],
                                            "boards": sorted(arena.boards()), "server": True})
                if u.path == "/api/board":
                    name = q.get("name", [""])[0]
                    boards = arena.boards()
                    if name not in boards:
                        return self._send(404, {"error": "no such board"})
                    return self._send(200, board_json(Board.load(boards[name])))
                if u.path == "/api/replays":
                    return self._send(200, [{k: v for k, v in it.items() if k != "path"} for it in arena.replays()])
                if u.path == "/api/replayfile":
                    name = q.get("file", [""])[0]
                    for it in arena.replays():
                        if it["file"] == name:
                            with open(it["path"], "rb") as f:
                                return self._send(200, f.read())
                    return self._send(404, {"error": "no such replay"})
                if u.path == "/api/replay":
                    game = arena.games.get(q.get("id", [""])[0])
                    if game is None:
                        return self._send(404, {"error": "no such game"})
                    return self._send(200, game.replay())
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/new":
                    gid, game = arena.new_game(body.get("pacman", "human"), body.get("ghost", "human"),
                                               body.get("board"), float(body.get("time_limit", 0.1)))
                    out = arena.payload(game)
                    out.update({"id": gid, "board": board_json(game.board), "labels": game.labels,
                                "rules": game.replay()["rules"]})
                    return self._send(200, out)
                if u.path == "/api/turn":
                    game = arena.games.get(str(body.get("id")))
                    if game is None:
                        return self._send(404, {"error": "no such game"})
                    with arena.lock:
                        if game.done:
                            return self._send(200, arena.payload(game))
                        move = body.get("move")
                        if game.needs_human_move():
                            if move is None:
                                return self._send(200, arena.payload(game))
                            side = game.turn_side
                            legal = game.core.legal_moves(getattr(game.core, side))
                            if side == "ghost" and move == "STOP":
                                # a human ghost holding no key stands still (agents must move; humans may rest)
                                event = {"tick": game.core.tick, "who": "ghost", "from": list(game.core.ghost),
                                         "move": "STAY", "ms": 0.0, "fault": None, "skipped": False, "ate": False}
                                game.core.apply(None)
                                event["to"] = list(game.core.ghost)
                                event["outcome"] = game.core.outcome
                                game.turns.append(event)
                                return self._send(200, arena.payload(game, event))
                            if not (move in legal or (move == "STOP" and side == "pacman")):
                                out = arena.payload(game)
                                out["error"] = "illegal move"
                                return self._send(200, out)
                            event = game.turn(move)
                        else:
                            event = game.turn()
                    return self._send(200, arena.payload(game, event))
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._send(500, {"error": str(e)})
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--agents", action="append", default=[], help="extra folder with agent files (repeatable)")
    ap.add_argument("--boards", action="append", default=[], help="extra folder with boards (repeatable)")
    ap.add_argument("--replays", action="append", default=[], help="folder with replay JSON files (repeatable)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    up = os.path.normpath(os.path.join(HERE, ".."))
    auto_agents = [p for p in (os.path.join(up, "Submission"), os.path.join(up, "submission")) if os.path.isdir(p)]
    arena = Arena(agent_dirs=[HERE, os.getcwd()] + auto_agents + args.agents,
                  board_dirs=[os.path.join(HERE, "boards"), os.path.join(os.getcwd(), "boards")] + args.boards,
                  replay_dirs=args.replays + [os.path.join(os.getcwd(), "replays")])
    port = args.port
    for _ in range(20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(arena))
            break
        except OSError:
            port += 1
    else:
        print("no free port"); return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Pac-Man League arena at {url}  (Ctrl+C to stop)")
    ag = arena.agents()
    print("agents:", ", ".join(v["label"] for v in ag.values()) or "none found")
    print("boards:", ", ".join(sorted(arena.boards())) or "none found")
    if auto_agents:
        print("students' folder:", auto_agents[0])
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
