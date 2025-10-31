from flask import Flask, request, jsonify
from flask_cors import CORS
import math, heapq, importlib
from math import sqrt
import threading

app = Flask(__name__)
CORS(app)

# ---------------------------
# Load newest matrix on demand (script1.py is rewritten by your counter)
# ---------------------------
def load_matrix():
    import script1
    importlib.reload(script1)  # always pick up latest grid
    return script1.matrix

# ---------------------------
# A* Pathfinding
# ---------------------------
def astar(matrix, start, goal):
    rows, cols = len(matrix), len(matrix[0])

    def in_bounds(rc):
        r, c = rc
        return 0 <= r < rows and 0 <= c < cols

    if not (in_bounds(start) and in_bounds(goal)):
        return None, float("inf")

    def h(a, b):  # Euclidean
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    open_set = [(h(start, goal), 0, start, [start])]
    seen = set()
    while open_set:
        f, g, cur, path = heapq.heappop(open_set)
        if cur in seen:
            continue
        seen.add(cur)

        if cur == goal:
            return path, g

        r, c = cur
        nbrs = [
            (r+1, c), (r-1, c), (r, c+1), (r, c-1),
            (r+1, c+1), (r+1, c-1), (r-1, c+1), (r-1, c-1)
        ]
        for nr, nc in nbrs:
            if 0 <= nr < rows and 0 <= nc < cols:
                step = sqrt(2) if (nr != r and nc != c) else 1
                new_g = g + matrix[nr][nc] * step
                heapq.heappush(open_set, (
                    new_g + h((nr, nc), goal),
                    new_g,
                    (nr, nc),
                    path + [(nr, nc)]
                ))
    return None, float("inf")

# ---------------------------
# Angles from path
# ---------------------------
def angles_from_path(path):
    if not path:
        return [999]
    out = []
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        out.append(180 - ang)
    out.append(999)  # end signal
    return out

# ---------------------------
# Shared state (same behavior as before)
# ---------------------------
lock = threading.Lock()
START_POINT = (9, 0)  # adjust if your start differs
selected_location = None
angles = []
tap_count = 0
current_path = []  # NEW: Store the current path

# 15x15 goals (rows/cols 0..14) — adjust as needed
LOCATION_COORDS = {
    "Entrance/Exit": (9, 0),
    "Foodtruck":     (6, 9),
    "Bar 1":         (5, 0),
    "Bar 2":         (4, 9),
    "Stage":         (0, 4),
}

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "endpoints": ["/grid", "/get_matrix", "/send_location", "/get_angle", "/tap", "/test", "/get_path"]
    })

@app.get("/test")
def test():
    return "Flask server reachable ✅"

@app.get("/grid")
def grid():
    M = load_matrix()
    return jsonify({"rows": len(M), "cols": len(M[0]) if M else 0, "matrix": M})

# Alias for teammate's web app expecting /get_matrix
@app.get("/get_matrix")
def get_matrix_alias():
    M = load_matrix()
    return jsonify({"rows": len(M), "cols": len(M[0]) if M else 0, "matrix": M})

# NEW: Get current path endpoint
@app.get("/get_path")
def get_path():
    global current_path, START_POINT, selected_location
    with lock:
        if selected_location:
            goal = LOCATION_COORDS.get(selected_location["name"])
            return jsonify({
                "path": current_path,
                "start": START_POINT,
                "goal": goal
            })
        return jsonify({"path": [], "start": None, "goal": None})

@app.post("/send_location")
def send_location():
    global selected_location, angles, tap_count, START_POINT, current_path
    data = request.get_json(silent=True) or {}
    sel = data.get("location")
    if not sel or "name" not in sel:
        return jsonify({"status": "error", "message": "No/invalid location"}), 400

    name = sel["name"]
    if name not in LOCATION_COORDS:
        return jsonify({"status": "error", "message": f"Unknown location: {name}"}), 400

    goal = LOCATION_COORDS[name]
    M = load_matrix()
    path, cost = astar(M, START_POINT, goal)

    with lock:
        selected_location = sel
        angles = angles_from_path(path)
        tap_count = 0
        current_path = path if path else []  # NEW: Store the path
        # START_POINT becomes goal after first tap (keeps your flow)

    return jsonify({
        "status": "ok",
        "start": START_POINT,
        "goal": goal,
        "path": path,
        "cost": cost,
        "angles_count": len(angles)
    })

@app.get("/get_angle")
def get_angle():
    global angles, tap_count
    with lock:
        if tap_count < len(angles):
            return jsonify({"angle": angles[tap_count], "index": tap_count, "total": len(angles)})
        return jsonify({"angle": 999, "index": tap_count, "total": len(angles)})

@app.post("/tap")
def tap():
    global tap_count, START_POINT, selected_location, angles
    with lock:
        if tap_count < len(angles):
            tap_count += 1
            finished = tap_count >= len(angles)
            if tap_count == 1 and selected_location:
                START_POINT = LOCATION_COORDS[selected_location["name"]]
            return jsonify({
                "status": "ok",
                "finished": finished,
                "taps_remaining": max(0, len(angles) - tap_count)
            })
        return jsonify({"status": "already_finished", "finished": True, "taps_remaining": 0})

if __name__ == "__main__":
    print("🚀 Flask on http://0.0.0.0:5000  (use ngrok to expose)")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)