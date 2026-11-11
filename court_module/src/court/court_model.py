"""BWF badminton court in metres (top-down), plus named points for calibration.

Coordinate system
-----------------
x runs across the court (0 .. 6.10), y runs along it (0 .. 13.40).
The net is at y = 6.70.

Naming convention (keep it the same for every clip!):
  * Stand at the "B0" end (y = 0) and look towards the net.
  * "D" = doubles side line, "S" = singles side line, "C" = centre line.
  * "L" / "R" = left / right as seen from the B0 end.
  * Pick whichever end of the court you like as B0 (e.g. the end nearest the camera).

A point name is "<x line>-<y line>", e.g. "DL-SS0" is where the left doubles
side line meets the short service line on the B0 half.
"""

COURT_LENGTH = 13.40
COURT_WIDTH = 6.10          # doubles width
SINGLES_WIDTH = 5.18
NET_Y = COURT_LENGTH / 2    # 6.70
SHORT_SERVICE = 1.98        # net -> short service line
LONG_SERVICE_DOUBLES = 0.76  # back line -> doubles long service line

_margin = (COURT_WIDTH - SINGLES_WIDTH) / 2

X_LINES = {
    "DL": 0.0,
    "SL": _margin,
    "C": COURT_WIDTH / 2,
    "SR": COURT_WIDTH - _margin,
    "DR": COURT_WIDTH,
}
Y_LINES = {
    "B0": 0.0,
    "LS0": LONG_SERVICE_DOUBLES,
    "SS0": NET_Y - SHORT_SERVICE,
    "SS1": NET_Y + SHORT_SERVICE,
    "LS1": COURT_LENGTH - LONG_SERVICE_DOUBLES,
    "B1": COURT_LENGTH,
}

_X_DESC = {"DL": "left doubles side line", "SL": "left singles side line",
           "C": "centre line", "SR": "right singles side line",
           "DR": "right doubles side line"}
_Y_DESC = {"B0": "back line (B0 end, y=0)",
           "LS0": "doubles long service line (B0 half)",
           "SS0": "short service line (B0 half)",
           "SS1": "short service line (far half)",
           "LS1": "doubles long service line (far half)",
           "B1": "back line (far end, y=13.40)"}

# Every named point -> (x, y) in metres.
COURT_POINTS = {}
POINT_DESCRIPTIONS = {}
for _xn, _x in X_LINES.items():
    for _yn, _y in Y_LINES.items():
        COURT_POINTS[f"{_xn}-{_yn}"] = (_x, _y)
        POINT_DESCRIPTIONS[f"{_xn}-{_yn}"] = f"{_X_DESC[_xn]} x {_Y_DESC[_yn]}"
# Net post bases sit on the doubles side lines (ground point under each post).
COURT_POINTS["DL-NET"] = (X_LINES["DL"], NET_Y)
COURT_POINTS["DR-NET"] = (X_LINES["DR"], NET_Y)
POINT_DESCRIPTIONS["DL-NET"] = "ground under the left net post (doubles line)"
POINT_DESCRIPTIONS["DR-NET"] = "ground under the right net post (doubles line)"

# Good default clicks: easy to see on most footage. Skip any that are hidden.
DEFAULT_CALIBRATION_POINTS = [
    "DL-B0", "DR-B0", "DL-B1", "DR-B1",
    "DL-SS0", "DR-SS0", "DL-SS1", "DR-SS1",
]

# Painted lines as ((x1, y1), (x2, y2)) segments, for overlays and the minimap.
_W, _L, _N = COURT_WIDTH, COURT_LENGTH, NET_Y
COURT_LINES = [
    ((0, 0), (_W, 0)), ((0, _L), (_W, _L)),                       # back lines
    ((0, 0), (0, _L)), ((_W, 0), (_W, _L)),                       # doubles sides
    ((X_LINES["SL"], 0), (X_LINES["SL"], _L)),                    # singles sides
    ((X_LINES["SR"], 0), (X_LINES["SR"], _L)),
    ((0, Y_LINES["SS0"]), (_W, Y_LINES["SS0"])),                  # short service
    ((0, Y_LINES["SS1"]), (_W, Y_LINES["SS1"])),
    ((0, Y_LINES["LS0"]), (_W, Y_LINES["LS0"])),                  # long service
    ((0, Y_LINES["LS1"]), (_W, Y_LINES["LS1"])),
    ((X_LINES["C"], 0), (X_LINES["C"], Y_LINES["SS0"])),          # centre lines
    ((X_LINES["C"], Y_LINES["SS1"]), (X_LINES["C"], _L)),
]
NET_LINE = ((0, _N), (_W, _N))


def in_court_bounds(xy, margin=0.0):
    """True if (x, y) metres is inside the doubles court grown by `margin` metres."""
    x, y = xy
    return (-margin <= x <= COURT_WIDTH + margin
            and -margin <= y <= COURT_LENGTH + margin)
