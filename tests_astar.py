#!/usr/bin/env python3
"""tests_astar.py — the ten automated A* tests. This is exactly how the A* part is marked.

  python tests_astar.py agent_<surname>.py

Each test calls your astar(start, goal, neighbors, cost, heuristic) and checks that the
returned path (a) is a list from start to goal, (b) only steps between neighbours, and
(c) has the cheapest possible total cost (one test uses step costs other than 1, as
the road map does — use cost(a, b), do not count steps). Ties are fine.
"""
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MOVES = ((-1, 0), (1, 0), (0, 1), (0, -1))


def grid(text):
    lines = [ln for ln in text.strip("\n").splitlines() if ln.strip()]
    walls = set()
    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            if ch == "#":
                walls.add((r, c))
    rows, cols = len(lines), max(len(ln) for ln in lines)

    def neighbors(cell):
        r, c = cell
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if 0 <= n[0] < rows and 0 <= n[1] < cols and n not in walls:
                yield n

    def cost(a, b):
        return 1

    return neighbors, cost


def manhattan_to(goal):
    return lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def zero(cell):
    return 0


def dijkstra_cost(start, goal, neighbors, cost):
    dist = {start: 0}
    pq = [(0, 0, start)]
    n = 0
    while pq:
        d, _, u = heapq.heappop(pq)
        if u == goal:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v in neighbors(u):
            nd = d + cost(u, v)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                n += 1
                heapq.heappush(pq, (nd, n, v))
    return None


def check_path(path, start, goal, neighbors, cost, optimal):
    if optimal is None:
        return (path is None, "expected None (no path), got %r" % (path,))
    if not isinstance(path, list):
        return (False, "expected a list, got %s" % type(path).__name__)
    if not path or path[0] != start:
        return (False, "path must start at start %r (got %r)" % (start, path[:1]))
    if path[-1] != goal:
        return (False, "path must end at goal %r (got %r)" % (goal, path[-1:]))
    total = 0
    for a, b in zip(path, path[1:]):
        if b not in set(neighbors(a)):
            return (False, "%r -> %r is not a legal step" % (a, b))
        total += cost(a, b)
    if total != optimal:
        return (False, "path cost %s, cheapest possible is %s" % (total, optimal))
    return (True, "cost %s" % total)


def build_tests():
    tests = []

    # 1. straight corridor
    nb, co = grid("""
.......
""")
    tests.append(("straight corridor", (0, 0), (0, 6), nb, co, manhattan_to((0, 6))))

    # 2. around a wall
    nb, co = grid("""
.......
.#####.
.......
""")
    tests.append(("around a wall", (0, 0), (2, 3), nb, co, manhattan_to((2, 3))))

    # 3. no path
    nb, co = grid("""
...#...
...#...
...#...
""")
    tests.append(("no path -> None", (0, 0), (0, 6), nb, co, manhattan_to((0, 6))))

    # 4. start == goal
    nb, co = grid("""
...
...
""")
    tests.append(("start equals goal -> [start]", (1, 1), (1, 1), nb, co, manhattan_to((1, 1))))

    # 5. the gap: a long wall with one opening far from the straight line
    nb, co = grid("""
.........
########.
.........
""")
    tests.append(("find the gap in the wall", (0, 0), (2, 0), nb, co, manhattan_to((2, 0))))

    # 6. weighted steps: the cheap road is longer (cost must be used, not just counted)
    def wcost(a, b):
        # stepping into row 1 costs 5, everything else 1
        return 5 if b[0] == 1 else 1
    nb6, _ = grid("""
.......
.......
.......
""")
    tests.append(("weighted steps: avoid the costly row", (1, 0), (1, 6), nb6, wcost, zero))

    # 7. zero heuristic (plain uniform-cost) must still be optimal
    nb, co = grid("""
..........
.########.
.#......#.
.#.####.#.
.#.#..#.#.
.#.#....#.
.#.#..#.#.
.#.######.
..........
""")
    tests.append(("zero heuristic", (0, 0), (4, 4), nb, co, zero))

    # 8. large open grid with Manhattan heuristic
    big = "\n".join("." * 40 for _ in range(30))
    nb, co = grid(big)
    tests.append(("large open grid", (0, 0), (29, 39), nb, co, manhattan_to((29, 39))))

    # 9. a graph that is not a grid at all: cities and road lengths
    roads = {"A": {"B": 7, "C": 9, "F": 14}, "B": {"A": 7, "C": 10, "D": 15}, "C": {"A": 9, "B": 10, "D": 11, "F": 2},
             "D": {"B": 15, "C": 11, "E": 6}, "E": {"D": 6, "F": 9}, "F": {"A": 14, "C": 2, "E": 9}}
    tests.append(("not a grid: a road map", "A", "E", lambda u: roads[u].keys(), lambda u, v: roads[u][v], zero))

    # 10. a maze with dead ends, Manhattan heuristic
    nb, co = grid("""
.#.......#...
.#.#####.#.#.
...#...#.#.#.
.###.#.#.#.#.
.....#.#...#.
####.#.#####.
.....#.......
.#.###.#####.
.#...........
""")
    tests.append(("maze with dead ends", (0, 0), (8, 12), nb, co, manhattan_to((8, 12))))
    return tests


def run_tests(astar):
    """Returns (passed, total, lines)."""
    lines = []
    passed = 0
    tests = build_tests()
    for i, (name, start, goal, nb, co, h) in enumerate(tests, 1):
        optimal = dijkstra_cost(start, goal, nb, co)
        try:
            path = astar(start, goal, nb, co, h)
            ok, why = check_path(path, start, goal, nb, co, optimal)
        except Exception as e:  # noqa: BLE001
            ok, why = False, "error: %s: %s" % (type(e).__name__, e)
        passed += ok
        lines.append("[%s] test %2d  %-32s %s" % ("OK " if ok else "FAIL", i, name, why))
    return passed, len(tests), lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    from pacman_engine import load_agent
    mod = load_agent(argv[0])
    if not hasattr(mod, "astar"):
        print("[FAIL] no function astar in", argv[0])
        return 1
    passed, total, lines = run_tests(mod.astar)
    print("\n".join(lines))
    print("A* tests: %d/%d" % (passed, total))
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
