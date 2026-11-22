"""kevin-mech V1: survival first.

Priorities each decision:
  1. never crash, always answer (every stage is guarded; the wrapper re-sends the last move on error)
  2. stay inside the zone (A* path to the safe circle, boost when it is urgent)
  3. break line of sight from enemy mechs using buildings; keep distance; face threats (front armor)
  4. shoot back when engaged or when an enemy is close; farm NPC drones for cubes when alone
  5. dodge charging lasers and incoming mortar shells
"""
from __future__ import annotations

import gc
import math
import random
import sys
import time
import traceback

import mechfight as mf

try:
    import nav
except Exception:  # pragma: no cover - keep running without the map model
    nav = None

PI = math.pi
TAU = 2 * math.pi

UPGRADE_PREF = [
    "plating", "nanites", "armor", "servos", "gun_damage", "gun_rate", "gun_mag",
    "sensors", "capacitor", "reactor", "jets", "gyros", "gun_velocity",
    "laser_damage", "laser_charge", "lightning_efficiency", "lightning_range", "lightning_chain",
    "mortar_reload", "mortar_splash", "mortar_velocity", "laser_pierce",
]


def wrap(a: float) -> float:
    return (a + PI) % TAU - PI


def fin(v: float, d: float = 0.0) -> float:
    return v if isinstance(v, (int, float)) and math.isfinite(v) else d


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Bot:
    name = "kevin-mech"

    def __init__(self, hello):
        self.cfg = cfg = hello.get("config", {})
        self.slot = hello.get("slot")
        self.rng = random.Random(hello.get("seed", 1))
        mech = cfg.get("mech", {})
        self.walk_speed = mech.get("walk_speed", 11.7)
        self.map = None
        if nav is not None:
            try:
                t0 = time.perf_counter()
                self.map = nav.Map(hello["map"]["dir"], step_up=mech.get("step_up", 1.2), head=mech.get("height", 9.0))
                log(f"map built in {time.perf_counter() - t0:.2f}s")
            except Exception:
                log("map load failed:\n" + traceback.format_exc())
                self.map = None
        gc.collect()
        try:
            gc.freeze()
        except AttributeError:
            pass
        w = cfg.get("weapons", {})
        self.gun = w.get("gun", {"speed": 252, "range": 230})
        self.mortar_g = w.get("mortar", {}).get("gravity", 22.0)

        self.memory = {}          # enemy id -> last known dict + 't'
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.attackers = {}       # id -> last time they damaged us
        self.last_hit_t = -99.0
        # navigation state
        self.path = None
        self.path_i = 0
        self.path_goal = None
        self.path_t = -99.0
        self.astar = None
        self.astar_goal = None
        self.goal = None
        self.goal_kind = ""
        self.goal_score_t = -99.0
        self.hist = []            # recent (t, x, y, commanded)
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.scan_yaw = None
        self.dodge_side = 1.0
        self.dodge_switch_t = 0.0
        self.slow_ticks = 0

    # ------------------------------------------------------------------------------------
    def act(self, obs):
        t0 = time.perf_counter()
        act = {"tick": obs["tick"]}
        s = obs["self"]
        t = obs["t"]
        x, y, z = s["pos"]
        self.t, self.x, self.y, self.z = t, x, y, z
        self.s = s
        heavy_ok = self.slow_ticks == 0

        try:
            self._update_memory(obs)
        except Exception:
            self._err("memory")
        try:
            if obs.get("offer"):
                act["pick"] = self._pick_upgrade(obs["offer"])
        except Exception:
            self._err("offer")

        move, boost, jump = (0.0, 0.0), False, False
        try:
            move, boost, jump = self._movement(obs, heavy_ok)
        except Exception:
            self._err("movement")
            move = self._zone_dir(obs)
        aim, fire, reload_ = None, False, False
        try:
            aim, fire, reload_ = self._combat(obs)
        except Exception:
            self._err("combat")
        try:
            tk = self._maybe_take(obs)
            if tk is not None:
                act["take"] = tk
        except Exception:
            self._err("take")

        mx, my = fin(move[0]), fin(move[1])
        n = math.hypot(mx, my)
        if n > 1:
            mx, my = mx / n, my / n
        act["move"] = [round(mx, 3), round(my, 3)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)
        if aim is not None:
            act["aim"] = [round(fin(aim[0]), 4), round(fin(aim[1]), 4)]
        act["fire"] = [bool(fire), False]
        if reload_:
            act["reload"] = True
        self.hist.append((t, x, y, n > 0.4))
        if len(self.hist) > 12:
            self.hist.pop(0)

        el = time.perf_counter() - t0
        self.slow_ticks = 2 if el > 0.025 else max(0, self.slow_ticks - 1)
        try:
            mf.draw.label(f"{self.goal_kind} hp{s['hp']:.0f} {el*1000:.1f}ms")
        except Exception:
            pass
        return act

    def _err(self, where):
        self.errors = getattr(self, "errors", 0) + 1
        if self.errors <= 10 or self.errors % 200 == 0:
            log(f"[{where}] t={getattr(self, 't', '?')}:\n{traceback.format_exc()}")

    # ------------------------------------------------------------------------------------
    def _update_memory(self, obs):
        t = self.t
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech" and m.get("id") != self.slot:
                d = dict(m)
                d["t"] = t
                self.memory[m["id"]] = d
        for k in [k for k, v in self.memory.items() if t - v["t"] > 20]:
            del self.memory[k]
        for e in obs.get("events", []):
            ty = e.get("type")
            if ty == "damaged" and e.get("weapon") != "zone":
                self.last_dmg_t = t
                self.last_dmg_bearing = e.get("bearing")
                if e.get("by") is not None:
                    self.attackers[e["by"]] = t
            elif ty == "hit":
                self.last_hit_t = t
            elif ty == "kill":
                self.memory.pop(e.get("target"), None)

    def _pick_upgrade(self, offer):
        ch = offer.get("choices", [])
        best, bi = 999, 0
        for i, c in enumerate(ch):
            r = UPGRADE_PREF.index(c["id"]) if c.get("id") in UPGRADE_PREF else 500
            if r < best:
                best, bi = r, i
        return bi

    def _maybe_take(self, obs):
        # V1: slot 1 is never fired yet, so do not pick anything up (V2 adds weapon use).
        return None
        s = self.s
        ws = s.get("weapons") or [None, None]
        if len(ws) > 1 and ws[1] is not None:
            return None
        reach = self.cfg.get("loot", {}).get("pickup_radius", 4.0)
        for p in obs.get("pickups", []):
            if p.get("kind") == "laser" and math.hypot(p["pos"][0] - self.x, p["pos"][1] - self.y) < reach - 0.3 \
                    and abs(p["pos"][2] - self.z) < 3.8:
                return p["id"]
        return None

    # ------------------------------------------------------------------------------------
    # zone
    def _zone(self, obs):
        zn = obs["zone"]
        cx, cy = zn["center"]
        r = zn["radius"]
        ncx, ncy = zn.get("next_center") or zn["center"]
        nr = zn.get("next_radius", r)
        if nr is None:
            nr = r
        margin = min(35.0, max(5.0, 0.2 * nr))
        safe_r = max(0.0, nr - margin)
        return cx, cy, r, ncx, ncy, nr, safe_r

    def _zone_dir(self, obs):
        try:
            _, _, _, ncx, ncy, _, _ = self._zone(obs)
            dx, dy = ncx - self.x, ncy - self.y
            d = math.hypot(dx, dy) or 1.0
            return (dx / d, dy / d)
        except Exception:
            return (0.0, 0.0)

    def _zone_urgency(self, obs):
        """Seconds of slack before the zone edge reaches us (negative = already outside)."""
        zn = obs["zone"]
        cx, cy, r, ncx, ncy, nr, safe_r = self._zone(obs)
        t = self.t
        x, y = self.x, self.y
        if not self.s.get("in_zone", True):
            return -1.0
        ss, se = zn.get("shrink_start", t), zn.get("shrink_end", t)
        need = max(0.0, math.hypot(x - ncx, y - ncy) - safe_r) / (self.walk_speed * 0.85)
        if math.hypot(x - ncx, y - ncy) <= nr:
            return 999.0
        # when does the moving edge pass us?
        t_out = se
        if se > ss:
            for k in range(1, 21):
                tau = ss + (se - ss) * k / 20.0
                if tau < t:
                    continue
                f = (tau - ss) / (se - ss)
                ccx, ccy = cx + (ncx - cx) * f, cy + (ncy - cy) * f
                rr = r + (nr - r) * f
                if math.hypot(x - ccx, y - ccy) > rr:
                    t_out = tau
                    break
        return (t_out - t) - need

    # ------------------------------------------------------------------------------------
    # threats
    def _threats(self, obs):
        """Enemy mechs we should care about: currently listed (visible or server-remembered),
        plus our own older memory as weaker threats."""
        out = []
        listed = set()
        for m in obs.get("mechs", []):
            if m.get("kind") != "mech":
                continue
            listed.add(m["id"])
            w = 1.0
            ws = m.get("weapons") or []
            if "laser" in ws:
                w += 0.5
            if "mortar" in ws:
                w += 0.3
            if m["id"] in self.attackers and self.t - self.attackers[m["id"]] < 6:
                w += 0.7
            out.append((m, w, True))
        for k, m in self.memory.items():
            if k in listed:
                continue
            age = self.t - m["t"]
            if age < 12:
                out.append((m, 0.5 * (1 - age / 12), False))
        # phantom threat from unseen damage
        if self.t - self.last_dmg_t < 3 and self.last_dmg_bearing is not None and not out:
            b = self.last_dmg_bearing
            ph = {"id": -1, "pos": [self.x + 90 * math.cos(b), self.y + 90 * math.sin(b), self.z], "vel": [0, 0, 0],
                  "aim": [wrap(b + PI), 0], "weapons": []}
            out.append((ph, 0.8, False))
        out.sort(key=lambda a: math.hypot(a[0]["pos"][0] - self.x, a[0]["pos"][1] - self.y))
        return out[:5]

    def _mortar_impacts(self, obs):
        pts = []
        g = self.mortar_g
        for p in obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            # solve pz + vz t - g/2 t^2 = ground ~ our z
            gz = self.z
            a, b, c = -0.5 * g, vz, pz - gz
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tt = (-b - math.sqrt(disc)) / (2 * a)
            if tt < 0 or tt > 8:
                continue
            pts.append((px + vx * tt, py + vy * tt, tt))
        return pts

    # ------------------------------------------------------------------------------------
    def _movement(self, obs, heavy_ok):
        t, x, y = self.t, self.x, self.y
        s = self.s
        energy = s.get("energy", 0)
        M = self.map
        threats = self._threats(obs)
        self._cached_threats = threats
        impacts = self._mortar_impacts(obs)
        slack = self._zone_urgency(obs)
        cx, cy, r, ncx, ncy, nr, safe_r = self._zone(obs)
        self._zone_now = (cx, cy, r - 8.0)
        hp_frac = s["hp"] / max(1.0, s.get("max_hp", 1400))
        engaged = t - self.last_dmg_t < 2.0
        boost = False
        jump = False

        # --- stuck recovery -------------------------------------------------------------
        if t < self.unstick_until:
            return self.unstick_dir, energy > 5, bool(s.get("grounded", True))
        if len(self.hist) >= 6 and all(h[3] for h in self.hist[-6:]):
            ht, hx, hy, _ = self.hist[-6]
            if math.hypot(x - hx, y - hy) < 1.0 and t - ht > 0.4:
                a = self.rng.uniform(-PI, PI)
                base = self._goal_dir() or (math.cos(a), math.sin(a))
                self.stuck_times = [st for st in getattr(self, "stuck_times", []) if t - st < 6] + [t]
                lm = getattr(self, "last_move", None)
                if M is not None and lm is not None and self.path:
                    # the gap ahead is tighter than the grid thinks: forget it so A* goes around
                    fx, fy = x + lm[0] * 4.5, y + lm[1] * 4.5
                    if M.nidx(fx, fy) != M.nidx(x, y):
                        M.mark_blocked(fx, fy)
                if len(self.stuck_times) >= 2 and energy > 40:
                    # repeated: climb straight over the obstacle toward the goal (jump, then jet)
                    self.unstick_dir = base
                    self.unstick_until = t + 3.0
                else:
                    side = self.rng.choice((-1, 1))
                    ux, uy = base[0] * 0.3 - side * base[1], base[1] * 0.3 + side * base[0]
                    n = math.hypot(ux, uy) or 1
                    self.unstick_dir = (ux / n, uy / n)
                    self.unstick_until = t + 0.7
                self.path = None
                self.hist.clear()
                return self.unstick_dir, energy > 10, True

        # --- laser dodge ------------------------------------------------------------------
        dodge = None
        for m, w, live in threats:
            if not live or not m.get("charging"):
                continue
            bx, by = x - m["pos"][0], y - m["pos"][1]
            bearing = math.atan2(by, bx)
            if abs(wrap(bearing - m["aim"][0])) < 0.15 and math.hypot(bx, by) < 300:
                if t > self.dodge_switch_t:
                    self.dodge_side = self.rng.choice((-1.0, 1.0))
                    self.dodge_switch_t = t + self.rng.uniform(0.4, 0.8)
                ux, uy = -math.sin(bearing) * self.dodge_side, math.cos(bearing) * self.dodge_side
                dodge = (ux, uy)
                break

        # --- mortar dodge ------------------------------------------------------------------
        for ix, iy, tt in impacts:
            fx, fy = x + s["vel"][0] * tt, y + s["vel"][1] * tt
            d = math.hypot(fx - ix, fy - iy)
            if d < 11:
                ax, ay = x - ix, y - iy
                n = math.hypot(ax, ay) or 1.0
                return (ax / n, ay / n), energy > 5, False

        # --- choose goal --------------------------------------------------------------------
        urgent = slack < 6
        kind = ""
        goal = None
        if urgent:
            kind = "zone!"
            goal = self._safe_point(ncx, ncy, safe_r)
            boost = energy > 15 and (slack < 3 or not s.get("in_zone", True))
        elif threats and (any(live for _, _, live in threats) or engaged):
            kind = "evade"
            goal = self._best_spot(obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok)
            boost = engaged and energy > 30
        else:
            cube = self._near_cube(obs)
            if cube is not None:
                kind = "cube"
                goal = cube
            elif math.hypot(x - ncx, y - ncy) > safe_r + 2 and slack < 60:
                kind = "zone"
                goal = self._safe_point(ncx, ncy, safe_r)
            else:
                kind = "hide"
                if t - self.goal_score_t > 1.0 or self.goal is None or self.goal_kind != "hide":
                    goal = self._best_spot(obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok)
                    self.goal_score_t = t
                else:
                    goal = self.goal
        self.goal_kind = kind
        self.goal = goal

        if dodge is not None and not urgent:
            self.goal_kind = "dodge"
            return dodge, energy > 10, False

        if goal is None:
            return (0.0, 0.0), False, False
        mv = self._steer_to(goal, heavy_ok)
        if mv[0] or mv[1]:
            self.last_move = mv
        return mv, boost, jump

    def _goal_dir(self):
        if self.goal is None:
            return None
        dx, dy = self.goal[0] - self.x, self.goal[1] - self.y
        d = math.hypot(dx, dy)
        if d < 0.5:
            return None
        return (dx / d, dy / d)

    def _safe_point(self, ncx, ncy, safe_r):
        x, y = self.x, self.y
        dx, dy = x - ncx, y - ncy
        d = math.hypot(dx, dy)
        if d <= safe_r:
            return (x, y)
        rr = max(0.0, safe_r - 4)
        if d > 1e-6:
            px, py = ncx + dx / d * rr, ncy + dy / d * rr
        else:
            px, py = ncx, ncy
        M = self.map
        if M is not None:
            k = M.nearest_walkable(px, py, 10)
            if k >= 0:
                return M.ncenter(k)
        return (px, py)

    def _near_cube(self, obs):
        best, bd = None, 35.0
        for c in obs.get("cubes", []):
            d = math.hypot(c["pos"][0] - self.x, c["pos"][1] - self.y)
            if d < bd and abs(c["pos"][2] - self.z) < 6:
                bd, best = d, (c["pos"][0], c["pos"][1])
        return best

    def _best_spot(self, obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok):
        """Score nearby walkable spots: out of enemy sight, inside the safe circle, near cover,
        away from enemies and NPCs, short to reach."""
        M = self.map
        x, y = self.x, self.y
        cands = [(x, y)]
        if self.goal is not None and math.hypot(self.goal[0] - x, self.goal[1] - y) < 60:
            cands.append(self.goal)
        nd = 12 if heavy_ok else 8
        rings = (7.0, 16.0, 28.0, 44.0) if heavy_ok else (8.0, 20.0, 36.0)
        off = self.rng.uniform(0, TAU / nd)
        for rad in rings:
            for i in range(nd):
                a = off + TAU * i / nd
                cands.append((x + rad * math.cos(a), y + rad * math.sin(a)))
        # toward the safe center
        dx, dy = ncx - x, ncy - y
        d = math.hypot(dx, dy)
        if d > 1:
            for rad in (15.0, 35.0):
                cands.append((x + dx / d * min(rad, d), y + dy / d * min(rad, d)))

        live = [(m, w) for m, w, lv in threats if lv][:3]
        ghosts = [(m, w) for m, w, lv in threats if not lv][:2]
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]

        best, bs = None, -1e18
        for i, (px, py) in enumerate(cands):
            if M is not None:
                if not M.walkable(px, py):
                    continue
                if i > 0 and not M.walk_line(x, y, px, py):
                    continue
                gz = M.ground(px, py)
            else:
                gz = self.z
            sc = 0.0
            # zone
            dz = math.hypot(px - ncx, py - ncy) - safe_r
            if dz > 0:
                sc -= 2.5 * dz
            dc = math.hypot(px - cx, py - cy) - (r - 5)
            if dc > 0:
                sc -= 20 * dc
            # enemies
            for m, w in live:
                mx, my, mz = m["pos"]
                dd = math.hypot(px - mx, py - my)
                if dd < 260:
                    seen = True
                    if M is not None:
                        seen = M.los(mx, my, mz + 8, px, py, gz + 8.5) or M.los(mx, my, mz + 8, px, py, gz + 4.5)
                    if seen:
                        sc -= 60 * w * (1.0 if dd < 200 else 0.5)
                sc -= w * max(0.0, 150 - dd) * 1.0
            for m, w in ghosts:
                mx, my = m["pos"][0], m["pos"][1]
                dd = math.hypot(px - mx, py - my)
                sc -= w * max(0.0, 130 - dd) * 0.6
            # the squad converges on the zone center: prefer the outer part of the safe circle
            if safe_r > 20:
                sc += 0.15 * min(math.hypot(px - ncx, py - ncy), safe_r)
            for m in npcs:
                dd = math.hypot(px - m["pos"][0], py - m["pos"][1])
                if dd < 55:
                    sc -= (55 - dd) * 0.8
            for ix, iy, _ in impacts:
                if math.hypot(px - ix, py - iy) < 12:
                    sc -= 80
            # cover and travel
            if M is not None:
                sc += 2.0 * min(M.cover_at(px, py), 8)
            sc -= 0.12 * math.hypot(px - x, py - y)
            if i == 1:
                sc += 6  # hysteresis for the current goal
            if sc > bs:
                bs, best = sc, (px, py)
        return best

    # ------------------------------------------------------------------------------------
    def _steer_to(self, goal, heavy_ok):
        M = self.map
        x, y, t = self.x, self.y, self.t
        gx, gy = goal
        d = math.hypot(gx - x, gy - y)
        if d < 1.5:
            self.path = None
            return (0.0, 0.0)
        if M is None or (d < 60 and M.walk_line(x, y, gx, gy)):
            self.path = None
            sp = 1.0 if d > 4 else max(0.3, d / 4)
            return ((gx - x) / d * sp, (gy - y) / d * sp)
        # need a path
        need_new = (self.path is None or self.path_goal is None
                    or math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 12
                    or t - self.path_t > 4.0)
        if need_new:
            if self.astar is None or self.astar_goal is None or math.hypot(self.astar_goal[0] - gx, self.astar_goal[1] - gy) > 12:
                sk = M.nearest_walkable(x, y, 3)
                gk = M.nearest_walkable(gx, gy, 10)
                self.astar = nav.AStar(M, sk, gk, getattr(self, "_zone_now", None))
                self.astar_goal = (gx, gy)
            if self.astar.step(4000, 0.008 if heavy_ok else 0.003):
                if self.astar.path:
                    self.path = self.astar.path + [(gx, gy)]
                    self.path_i = 0
                    self.path_goal = (gx, gy)
                    self.path_t = t
                else:
                    self.path = None
                    self.path_goal = (gx, gy)
                    self.path_t = t
                self.astar = None
        if self.path:
            p = self.path
            while self.path_i < len(p) - 1 and math.hypot(p[self.path_i][0] - x, p[self.path_i][1] - y) < 3.0:
                self.path_i += 1
            tgt = p[self.path_i]
            for k in range(min(len(p) - 1, self.path_i + 8), self.path_i, -1):
                if M.walk_line(x, y, p[k][0], p[k][1]):
                    tgt = p[k]
                    break
            dx, dy = tgt[0] - x, tgt[1] - y
            n = math.hypot(dx, dy) or 1.0
            return (dx / n, dy / n)
        # no path (yet): head straight, the server slides us along walls
        return ((gx - x) / d, (gy - y) / d)

    # ------------------------------------------------------------------------------------
    def _combat(self, obs):
        s = self.s
        t, x, y, z = self.t, self.x, self.y, self.z
        M = self.map
        muzzle = z + s.get("stats", {}).get("muzzle_height", 7.0)
        eye = z + s.get("stats", {}).get("eye_height", 8.0)
        gun = None
        ws = s.get("weapons") or []
        if ws and ws[0] and ws[0].get("kind") == "gun":
            gun = ws[0]
        gstats = (gun or {}).get("stats", self.gun)
        speed = gstats.get("speed", 252.0)
        rng = gstats.get("range", 230.0)
        engaged = t - self.last_dmg_t < 4.0

        # candidate targets
        best, bscore = None, -1e9
        for m in obs.get("mechs", []):
            mx, my, mz = m["pos"]
            d = math.hypot(mx - x, my - y)
            if d > rng * 0.95:
                continue
            is_mech = m.get("kind") == "mech"
            scale = 1 + 0.05 * (m.get("cubes") or 0)
            cz = mz + (4.5 * scale if is_mech else 1.75)
            if M is not None and not M.los(x, y, muzzle, mx, my, cz):
                continue
            sc = -d
            if is_mech:
                sc += 300
                if m["id"] in self.attackers and t - self.attackers[m["id"]] < 5:
                    sc += 150
                hp = m.get("hp")
                if hp is not None:
                    sc += max(0, 600 - hp) * 0.3
            else:
                # NPCs: only when they attack us, or when we are alone and healthy (cube farming)
                attacking = m["id"] in self.attackers and t - self.attackers[m["id"]] < 5
                alone = not any(mm.get("kind") == "mech" for mm in obs.get("mechs", []))
                if not (attacking or (alone and s["hp"] > 0.6 * s.get("max_hp", 1400) and d < 120)):
                    continue
            if sc > bscore:
                bscore, best = sc, (m, cz, d)

        aim = None
        fire = False
        reload_ = False
        if best is not None:
            m, cz, d = best
            mx, my, mz = m["pos"]
            vx, vy, vz = m.get("vel", [0, 0, 0])
            tx, ty, tz = mx, my, cz
            for _ in range(2):
                tt = math.hypot(tx - x, ty - y) / speed
                tx, ty = mx + vx * tt, my + vy * tt
                tz = cz + (vz * tt if m.get("kind") == "mech" else 0)
            yaw = math.atan2(ty - y, tx - x)
            pitch = math.atan2(tz - muzzle, math.hypot(tx - x, ty - y))
            aim = (yaw, pitch)
            is_mech = m.get("kind") == "mech"
            want = True
            if is_mech:
                aiming_at_us = abs(wrap(math.atan2(y - my, x - mx) - m["aim"][0])) < 0.35
                want = engaged or d < 120 or aiming_at_us or (m.get("hp") or 9999) < 400
            if want and gun is not None and gun.get("reload", 0) <= 0 and gun.get("ammo", 1) > 0:
                tol = max(0.012, math.atan2(2.2, max(d, 1.0)))
                cur = s.get("aim", [yaw, pitch])
                if abs(wrap(cur[0] - yaw)) < tol and abs(cur[1] - pitch) < tol * 1.5 + 0.01:
                    fire = True
        else:
            aim = self._idle_aim(obs)
        if gun is not None and gun.get("reload", 0) <= 0:
            ammo = gun.get("ammo", 12)
            if ammo <= 0 or (best is None and ammo < gstats.get("mag", 12) * 0.5 and t - self.last_dmg_t > 2):
                reload_ = True
        return aim, fire, reload_

    def _idle_aim(self, obs):
        t, x, y = self.t, self.x, self.y
        # face the last damage source
        if t - self.last_dmg_t < 3 and self.last_dmg_bearing is not None:
            self.scan_yaw = self.last_dmg_bearing
            return (self.last_dmg_bearing, 0.0)
        # face the nearest known enemy
        thr = getattr(self, "_cached_threats", None) or []
        if thr:
            m = thr[0][0]
            yaw = math.atan2(m["pos"][1] - y, m["pos"][0] - x)
            self.scan_yaw = yaw
            return (yaw, 0.0)
        # face a fresh combat sound
        for snd in obs.get("sounds", []):
            if snd.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and snd.get("dist") in ("near", "mid") and snd.get("age", 9) < 1.0:
                self.scan_yaw = snd["bearing"]
                return (snd["bearing"], 0.0)
        # slow scan
        if self.scan_yaw is None:
            self.scan_yaw = self.s.get("aim", [0, 0])[0]
        self.scan_yaw = wrap(self.scan_yaw + 0.28)
        return (self.scan_yaw, 0.0)


if __name__ == "__main__":
    mf.run(Bot)
