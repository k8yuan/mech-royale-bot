"""Static map model: terrain, buildings, line-of-sight, walkability, A* pathfinding.

Built once from hello.map.dir (v2/world.json + v2/terrain.i16). Standard library only.
Fine grid (FINE m) holds ground height and the solid span of prisms for sight/shot checks.
Nav grid (NAV m) holds walkability for a 2.5 m-radius mech and is what A* runs on.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys
import time
from array import array

FINE = 2.0
NAV = 4.0
INF = float("inf")


class Map:
    def __init__(self, map_dir: str, step_up: float = 1.2, head: float = 9.0):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        b = w["bounds"]
        self.minx, self.maxx, self.miny, self.maxy = b["minX"], b["maxX"], b["minY"], b["maxY"]
        t = w["terrain"]
        self._load_terrain(os.path.join(map_dir, "v2", "terrain.i16"), t)

        # ---- fine grid -------------------------------------------------------------
        self.fx0, self.fy0 = self.minx, self.miny
        self.fnx = int(math.ceil((self.maxx - self.minx) / FINE)) + 1
        self.fny = int(math.ceil((self.maxy - self.miny) / FINE)) + 1
        fnx, fny = self.fnx, self.fny
        n = fnx * fny
        ground = array("f", bytes(4 * n))
        for j in range(fny):
            y = self.fy0 + (j + 0.5) * FINE
            row = j * fnx
            for i in range(fnx):
                ground[row + i] = self.ground(self.fx0 + (i + 0.5) * FINE, y)
        self.fground = ground
        top = array("f", [-1e9]) * n     # highest solid top over the cell
        base = array("f", [1e9]) * n     # lowest solid base over the cell
        blocked = bytearray(n)           # 1 = a wall for a walking mech
        for p in w["prisms"]:
            pb, pt = float(p["base"]), float(p["top"])
            deck = p.get("kind") == "bridge"
            rings = [p["ring"]] + [h for h in (p.get("holes") or []) if h]
            for idx in self._scan_fill(rings):
                if pt > top[idx]:
                    top[idx] = pt
                if pb < base[idx]:
                    base[idx] = pb
                g = ground[idx]
                if pt > g + step_up and pb < g + head:
                    # a low bridge deck over a riverbed is a walking surface, not a wall
                    if deck and pt - g < 6.0:
                        continue
                    blocked[idx] = 1
        # steep terrain is a wall too
        lim = math.tan(math.radians(48)) * 2 * FINE
        for j in range(1, fny - 1):
            row = j * fnx
            for i in range(1, fnx - 1):
                k = row + i
                if abs(ground[k + 1] - ground[k - 1]) > lim or abs(ground[k + fnx] - ground[k - fnx]) > lim:
                    blocked[k] = 1
        self.ftop, self.fbase, self.fblocked = top, base, blocked

        # ---- nav grid ----------------------------------------------------------------
        self.nnx = int(math.ceil((self.maxx - self.minx) / NAV))
        self.nny = int(math.ceil((self.maxy - self.miny) / NAV))
        nnx, nny = self.nnx, self.nny
        walk = bytearray(nnx * nny)
        cover = bytearray(nnx * nny)     # how much wall is within ~8 m (0..255)
        offs = (-2.0, 0.0, 2.0)
        margin = 6.0
        for j in range(nny):
            cy = self.miny + (j + 0.5) * NAV
            if cy < self.miny + margin or cy > self.maxy - margin:
                continue
            for i in range(nnx):
                cx = self.minx + (i + 0.5) * NAV
                if cx < self.minx + margin or cx > self.maxx - margin:
                    continue
                ok = True
                for dy in offs:
                    fj = int((cy + dy - self.fy0) / FINE)
                    r = fj * fnx
                    for dx in offs:
                        if blocked[r + int((cx + dx - self.fx0) / FINE)]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    walk[j * nnx + i] = 1
        for j in range(nny):
            for i in range(nnx):
                k = j * nnx + i
                if not walk[k]:
                    continue
                c = 0
                for dj in (-2, -1, 0, 1, 2):
                    jj = j + dj
                    if jj < 0 or jj >= nny:
                        continue
                    for di in (-2, -1, 0, 1, 2):
                        ii = i + di
                        if 0 <= ii < nnx and not walk[jj * nnx + ii]:
                            c += 1
                cover[k] = c
        self.walk, self.cover = walk, cover

        # ---- canopy: nav cells hidden under big tree crowns (vision only; trees never stop shots)
        canopy = bytearray(nnx * nny)
        try:
            tr = w.get("trees") or {}
            fields = tr.get("fields") or []
            data = tr.get("data") or []
            nf = len(fields)
            ix = {k: fields.index(k) for k in ("x", "y", "height", "crownR", "crownBase") if k in fields}
            cxk = fields.index("crownX") if "crownX" in fields else ix.get("x")
            cyk = fields.index("crownY") if "crownY" in fields else ix.get("y")
            if nf and len(ix) == 5:
                for q in range(0, len(data) - nf + 1, nf):
                    r = data[q:q + nf]
                    cr = float(r[ix["crownR"]])
                    if cr < 4.0:
                        continue
                    tx, ty = float(r[cxk]), float(r[cyk])
                    g = self.ground(tx, ty)
                    if float(r[ix["crownBase"]]) > g + 5.5 or float(r[ix["height"]]) < g + 8.0:
                        continue
                    rr = cr - 1.5
                    i0, i1 = int((tx - rr - self.minx) / NAV), int((tx + rr - self.minx) / NAV)
                    j0, j1 = int((ty - rr - self.miny) / NAV), int((ty + rr - self.miny) / NAV)
                    for jj in range(max(0, j0), min(nny - 1, j1) + 1):
                        for ii in range(max(0, i0), min(nnx - 1, i1) + 1):
                            ccx = self.minx + (ii + 0.5) * NAV
                            ccy = self.miny + (jj + 0.5) * NAV
                            if (ccx - tx) ** 2 + (ccy - ty) ** 2 <= rr * rr:
                                k = jj * nnx + ii
                                if canopy[k] < 9:
                                    canopy[k] += 1
        except Exception:
            pass
        self.canopy = canopy

    # ---------------------------------------------------------------------------------
    def _load_terrain(self, path: str, t: dict) -> None:
        self.tnx, self.tny = int(t["nx"]), int(t["ny"])
        self.tx0, self.ty0, self.tstep = float(t["minX"]), float(t["minY"]), float(t["step"])
        scale = float(t.get("scale", 0.01))
        a = array("h")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        self.terr = array("f", (v * scale for v in a))

    def ground(self, x: float, y: float) -> float:
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        i = int(fx)
        j = int(fy)
        if i < 0:
            i, fx = 0, 0.0
        if j < 0:
            j, fy = 0, 0.0
        if i >= self.tnx - 1:
            i, fx = self.tnx - 2, float(self.tnx - 1)
        if j >= self.tny - 1:
            j, fy = self.tny - 2, float(self.tny - 1)
        u, v = fx - i, fy - j
        k = j * self.tnx + i
        T = self.terr
        a, b, c, d = T[k], T[k + 1], T[k + self.tnx], T[k + self.tnx + 1]
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v

    def _scan_fill(self, rings):
        """Fine-grid cell indices whose centers are inside the polygon (even-odd over ring + holes)."""
        edges = []
        ymin, ymax = INF, -INF
        for r in rings:
            m = len(r) // 2
            for k in range(m):
                x1, y1 = r[2 * k], r[2 * k + 1]
                x2, y2 = r[2 * ((k + 1) % m)], r[2 * ((k + 1) % m) + 1]
                if y1 != y2:
                    edges.append((x1, y1, x2, y2))
                ymin = min(ymin, y1)
                ymax = max(ymax, y1)
        out = []
        if not edges:
            return out
        fnx, fny = self.fnx, self.fny
        j0 = max(0, int((ymin - self.fy0) / FINE) - 1)
        j1 = min(fny - 1, int((ymax - self.fy0) / FINE) + 1)
        for j in range(j0, j1 + 1):
            y = self.fy0 + (j + 0.5) * FINE
            xs = []
            for x1, y1, x2, y2 in edges:
                if (y1 <= y < y2) or (y2 <= y < y1):
                    xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            if len(xs) < 2:
                continue
            xs.sort()
            row = j * fnx
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - self.fx0) / FINE - 0.5)))
                i1 = min(fnx - 1, int(math.floor((xs[a + 1] - self.fx0) / FINE - 0.5)))
                for i in range(i0, i1 + 1):
                    out.append(row + i)
        return out

    # ---------------------------------------------------------------------------------
    def fidx(self, x: float, y: float) -> int:
        i = int((x - self.fx0) / FINE)
        j = int((y - self.fy0) / FINE)
        if i < 0 or j < 0 or i >= self.fnx or j >= self.fny:
            return -1
        return j * self.fnx + i

    def los(self, ax, ay, az, bx, by, bz, step: float = 1.5) -> bool:
        """True if the segment a->b passes through no prism and stays above terrain."""
        dx, dy, dz = bx - ax, by - ay, bz - az
        d = math.sqrt(dx * dx + dy * dy)
        n = int(d / step) + 1
        top, base, gr = self.ftop, self.fbase, self.fground
        fx0, fy0, fnx, fny = self.fx0, self.fy0, self.fnx, self.fny
        inv = 1.0 / n
        for s in range(1, n):
            f = s * inv
            x = ax + dx * f
            y = ay + dy * f
            i = int((x - fx0) * 0.5)
            j = int((y - fy0) * 0.5)
            if i < 0 or j < 0 or i >= fnx or j >= fny:
                return False
            k = j * fnx + i
            z = az + dz * f
            if base[k] < z < top[k]:
                return False
            if z < gr[k] - 0.3:
                return False
        return True

    # ---- nav ----------------------------------------------------------------------
    def nidx(self, x: float, y: float) -> int:
        i = int((x - self.minx) / NAV)
        j = int((y - self.miny) / NAV)
        if i < 0 or j < 0 or i >= self.nnx or j >= self.nny:
            return -1
        return j * self.nnx + i

    def ncenter(self, k: int):
        return (self.minx + (k % self.nnx + 0.5) * NAV, self.miny + (k // self.nnx + 0.5) * NAV)

    def mark_blocked(self, x: float, y: float) -> None:
        """Learned at runtime: a gap we could not get through. A* avoids it from now on."""
        k = self.nidx(x, y)
        if k >= 0 and self.walk[k]:
            if getattr(self, "walk0", None) is None:
                self.walk0 = bytes(self.walk)
            self.walk[k] = 0
            self.marks = getattr(self, "marks", 0) + 1

    def reset_marks(self) -> bool:
        """Forget learned blocks (e.g. when they cut off every route). True if anything changed."""
        if getattr(self, "marks", 0) and getattr(self, "walk0", None) is not None and len(self.walk0) == len(self.walk):
            self.walk[:] = self.walk0
            self.marks = 0
            return True
        return False

    def walkable(self, x: float, y: float) -> bool:
        k = self.nidx(x, y)
        return k >= 0 and self.walk[k] == 1

    def cover_at(self, x: float, y: float) -> int:
        k = self.nidx(x, y)
        return self.cover[k] if k >= 0 else 0

    def canopy_at(self, x: float, y: float) -> int:
        k = self.nidx(x, y)
        return self.canopy[k] if k >= 0 else 0

    def nearest_walkable(self, x: float, y: float, rmax: int = 8):
        k = self.nidx(x, y)
        if k >= 0 and self.walk[k]:
            return k
        i0 = int((x - self.minx) / NAV)
        j0 = int((y - self.miny) / NAV)
        best, bd = -1, INF
        for r in range(1, rmax + 1):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nnx and 0 <= j < self.nny and self.walk[j * self.nnx + i]:
                        d = di * di + dj * dj
                        if d < bd:
                            bd, best = d, j * self.nnx + i
            if best >= 0:
                return best
        return -1

    def walk_line(self, ax, ay, bx, by) -> bool:
        """True if a straight walk a->b stays on walkable nav cells (sampled every 2 m)."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        n = int(d / 2.0) + 1
        walk, nnx, nny = self.walk, self.nnx, self.nny
        x0, y0 = self.minx, self.miny
        for s in range(1, n + 1):
            f = s / n
            i = int((ax + dx * f - x0) * 0.25)
            j = int((ay + dy * f - y0) * 0.25)
            if i < 0 or j < 0 or i >= nnx or j >= nny or not walk[j * nnx + i]:
                return False
        return True


class AStar:
    """Time-sliced A* on the nav grid. Call step(budget) each tick until done."""

    SQ2 = math.sqrt(2.0)

    def __init__(self, m: Map, start: int, goal: int, zone=None):
        """zone: optional (cx, cy, r): steps onto cells outside that circle cost extra."""
        self.m = m
        self.zone = zone
        self.start, self.goal = start, goal
        self.done = False
        self.path = None
        n = m.nnx * m.nny
        self.g = {start: 0.0}
        self.came = {}
        self.closed = bytearray(n)
        gx, gy = goal % m.nnx, goal // m.nnx
        self.gx, self.gy = gx, gy
        self.open = [(self._h(start), 0.0, start)]
        if start < 0 or goal < 0:
            self.done = True

    def _h(self, k: int) -> float:
        nnx = self.m.nnx
        dx = abs(k % nnx - self.gx)
        dy = abs(k // nnx - self.gy)
        return (dx + dy + (self.SQ2 - 2) * min(dx, dy)) * 1.05

    def step(self, max_expand: int = 2500, max_s: float = 0.008) -> bool:
        """Expand up to max_expand nodes or max_s seconds, whichever comes first."""
        if self.done:
            return True
        deadline = time.perf_counter() + max_s
        m = self.m
        nnx, nny, walk = m.nnx, m.nny, m.walk
        g, came, closed, op = self.g, self.came, self.closed, self.open
        goal = self.goal
        SQ2 = self.SQ2
        pop, push, h = heapq.heappop, heapq.heappush, self._h
        zone = self.zone
        x0, y0 = m.minx, m.miny
        zr2 = max(0.0, zone[2]) ** 2 if zone is not None else 0.0
        cnt = 0
        while op and cnt < max_expand:
            f, gk, k = pop(op)
            if closed[k]:
                continue
            closed[k] = 1
            cnt += 1
            if cnt & 127 == 0 and time.perf_counter() > deadline:
                push(op, (f, gk, k))
                closed[k] = 0
                break
            if k == goal:
                self._finish(k)
                return True
            i, j = k % nnx, k // nnx
            for di, dj, c in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)):
                ii, jj = i + di, j + dj
                if ii < 0 or jj < 0 or ii >= nnx or jj >= nny:
                    continue
                nk = jj * nnx + ii
                if closed[nk] or not walk[nk]:
                    continue
                if di and dj and (not walk[j * nnx + ii] or not walk[jj * nnx + i]):
                    continue
                ng = gk + c
                if zone is not None:
                    zx = x0 + (ii + 0.5) * NAV - zone[0]
                    zy = y0 + (jj + 0.5) * NAV - zone[1]
                    if zx * zx + zy * zy > zr2:
                        ng += 6.0
                if ng < g.get(nk, INF):
                    g[nk] = ng
                    came[nk] = k
                    push(op, (ng + h(nk), ng, nk))
        if not op:
            self.done = True  # unreachable
        return self.done

    def _finish(self, k: int) -> None:
        pts = []
        while k != self.start:
            pts.append(self.m.ncenter(k))
            k = self.came[k]
        pts.reverse()
        self.path = pts
        self.done = True
