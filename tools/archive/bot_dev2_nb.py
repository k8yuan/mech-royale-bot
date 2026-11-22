"""kevin-mech: survival-first battle royale bot (mechfight/1).

Each decision:
  1. never crash, always answer (each stage is guarded; the wrapper re-sends the last move on error)
  2. stay inside the zone (A* to the safe circle, boost when urgent)
  3. dodge charging lasers aimed at us and incoming mortar shells
  4. fight a lone attacker we can beat; otherwise break line of sight using buildings and back off
  5. when safe: grab a better weapon, shoot crates / NPC drones for cubes, collect cubes, take supply drops
  6. otherwise hide near cover in the outer part of the safe circle, scanning for threats
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
except Exception:  # keep running without the map model
    nav = None

PI = math.pi
TAU = 2 * math.pi

UPGRADE_PREF = [
    "plating", "armor", "nanites", "servos", "laser_damage", "laser_charge", "gun_damage", "gun_rate",
    "lightning_efficiency", "lightning_range", "gun_mag", "capacitor", "reactor", "jets", "sensors",
    "mortar_reload", "mortar_splash", "gyros", "gun_velocity", "lightning_chain", "mortar_velocity", "laser_pierce",
]
# PROFILE "fight": trade shots, farm cubes, take fights we can win (scored 4.00 on the practice board).
# PROFILE "survive": stay unseen, shoot only when engaged or rushed, farm only early, fight only when clearly ahead.
PROFILE = "fight"
_P = {
    "fight":   dict(FIGHT_RATIO=0.9, FIRE_ALWAYS=True, CLOSE_FIRE=130.0, FARM_T_MAX=330.0, QUIET_S=0.0, CANOPY_W=3.0),
    "survive": dict(FIGHT_RATIO=1.4, FIRE_ALWAYS=False, CLOSE_FIRE=75.0, FARM_T_MAX=150.0, QUIET_S=20.0, CANOPY_W=6.0),
}[PROFILE]
FIGHT_RATIO = _P["FIGHT_RATIO"]
FIRE_ALWAYS = _P["FIRE_ALWAYS"]
CLOSE_FIRE = _P["CLOSE_FIRE"]      # always shoot back at mechs closer than this
FARM_T_MAX = _P["FARM_T_MAX"]      # no crate / NPC farming after this match time
QUIET_S = _P["QUIET_S"]            # farm only if no enemy mech was seen for this long
CANOPY_W = _P["CANOPY_W"]
FAR_ESCAPE = True
FARM_HP = 0.75
RELOCATE = True
DODGE_STRAFE = False
STRAFE = True
STRAFE_IDLE = 0.25
JUMP_P_THREAT = 0.3
JUMP_P_IDLE = 0.04
LEAD_TAU = 1.0
STICKY = 250.0
FOCUS_W = 400.0
PURSUE = True
STRAFE_ANG = 0.6
STOP_P = 0.15
STRAFE_BOOST = False
WEAPON_RANK = {"laser": 3, "lightning": 2, "mortar": 1}


def wrap(a: float) -> float:
    return (a + PI) % TAU - PI


def fin(v, d: float = 0.0) -> float:
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else d


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Bot:
    name = "kevin-mech"

    def __init__(self, hello):
        self.cfg = cfg = hello.get("config", {}) or {}
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
        self.wcfg = w
        self.gun = w.get("gun", {"speed": 252, "range": 230})
        self.mortar_g = w.get("mortar", {}).get("gravity", 22.0)
        self.reach = cfg.get("loot", {}).get("pickup_radius", 4.0)

        self.memory = {}          # enemy id -> last known dict + 't'
        self.items = {}           # id -> {"kind", "pos", "t", ...} for cubes / pickups / crates
        self.last_dmg_t = -99.0
        self.last_dmg_bearing = None
        self.attackers = {}       # id -> last time they damaged us
        self.last_hit_t = -99.0
        self.dmg_window = []      # (t, amount) recent damage taken
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
        self.hist = []
        self.unstick_until = -1.0
        self.unstick_dir = (1.0, 0.0)
        self.scan_yaw = None
        self.dodge_side = 1.0
        self.dodge_switch_t = 0.0
        self.slow_ticks = 0
        self.target_id = None
        self.target_seen_t = -99.0
        self.take_fail = {}       # pickup id -> t
        self.last_move = None
        self._zone_now = None
        self._cached_threats = []
        self.strafe_side = 1.0
        self.strafe_flip_t = 0.0
        self.strafe_ang = 0.0
        self.strafe_stop = False
        self.boosting = True

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _vec(v, n):
        if not isinstance(v, (list, tuple)):
            return [0.0] * n
        out = [fin(a) for a in list(v)[:n]]
        return out + [0.0] * (n - len(out))

    def _sanitize(self, obs):
        s = obs.setdefault("self", {})
        s["pos"] = self._vec(s.get("pos"), 3)
        s["vel"] = self._vec(s.get("vel"), 3)
        s["aim"] = self._vec(s.get("aim"), 2)
        s["hp"] = fin(s.get("hp"), 1.0)
        if not isinstance(s.get("stats"), dict):
            s["stats"] = {}
        if not isinstance(s.get("weapons"), list):
            s["weapons"] = [None, None]
        for key in ("mechs", "projectiles", "crates", "cubes", "pickups", "sounds", "beams", "events", "drops"):
            v = obs.get(key)
            obs[key] = [e for e in v if isinstance(e, dict)] if isinstance(v, list) else []
        for m in obs["mechs"]:
            m["pos"] = self._vec(m.get("pos"), 3)
            m["vel"] = self._vec(m.get("vel"), 3)
            m["aim"] = self._vec(m.get("aim"), 2)
        for key in ("projectiles", "crates", "cubes", "pickups", "drops"):
            for e in obs[key]:
                e["pos"] = self._vec(e.get("pos"), 3)
                if key == "projectiles":
                    e["vel"] = self._vec(e.get("vel"), 3)
        zn = obs.get("zone")
        if not isinstance(zn, dict):
            zn = obs["zone"] = {}
        zn["center"] = self._vec(zn.get("center"), 2) if zn.get("center") is not None else [s["pos"][0], s["pos"][1]]
        zn["radius"] = fin(zn.get("radius"), 1e4)
        if zn.get("next_center") is None:
            zn["next_center"] = list(zn["center"])
        else:
            zn["next_center"] = self._vec(zn["next_center"], 2)
        zn["next_radius"] = fin(zn.get("next_radius"), zn["radius"])
        obs["t"] = fin(obs.get("t"), 0.0)

    def act(self, obs):
        t0 = time.perf_counter()
        try:
            self._sanitize(obs)
        except Exception:
            self._err("sanitize")
        act = {"tick": obs["tick"]}
        s = obs["self"]
        t = obs["t"]
        x, y, z = s["pos"]
        self.t, self.x, self.y, self.z = t, x, y, z
        self.s = s
        self.obs = obs
        heavy_ok = self.slow_ticks == 0

        for stage in (self._update_memory, self._update_items):
            try:
                stage(obs)
            except Exception:
                self._err(stage.__name__)
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
        aim, fire0, fire1, reload_ = None, False, False, False
        try:
            aim, fire0, fire1, reload_ = self._combat(obs)
        except Exception:
            self._err("combat")
        try:
            tk = self._maybe_take(obs)
            if tk is not None:
                act["take"] = tk[0]
                if tk[1] is not None:
                    act["drop"] = tk[1]
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
        act["fire"] = [bool(fire0), bool(fire1)]
        if reload_:
            act["reload"] = True
        self.hist.append((t, x, y, n > 0.4))
        if len(self.hist) > 12:
            self.hist.pop(0)

        el = time.perf_counter() - t0
        self.slow_ticks = 3 if el > 0.020 else max(0, self.slow_ticks - 1)
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
    # memory
    def _update_memory(self, obs):
        t = self.t
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech" and m.get("id") != self.slot:
                d = dict(m)
                d["t"] = t
                prev = self.memory.get(m["id"])
                hist = prev.get("vh", []) if prev else []
                hist = [h for h in hist if t - h[0] < 1.2] + [(t, m["vel"][0], m["vel"][1])]
                d["vh"] = hist
                self.memory[m["id"]] = d
        for k in [k for k, v in self.memory.items() if t - v["t"] > 25]:
            del self.memory[k]
        for e in obs.get("events", []):
            ty = e.get("type")
            if ty == "damaged":
                if e.get("weapon") != "zone":
                    self.last_dmg_t = t
                    self.last_dmg_bearing = e.get("bearing")
                    if e.get("by") is not None:
                        self.attackers[e["by"]] = t
                self.dmg_window.append((t, e.get("amount", 0)))
            elif ty == "hit":
                self.last_hit_t = t
            elif ty == "kill":
                self.memory.pop(e.get("target"), None)
            elif ty == "take_failed":
                self.take_fail[e.get("id")] = t
                if e.get("reason") == "gone":
                    self.items.pop(e.get("id"), None)
            elif ty == "took":
                self.items.pop(e.get("id"), None)
        self.dmg_window = [d for d in self.dmg_window if t - d[0] < 3]

    def _update_items(self, obs):
        t, x, y = self.t, self.x, self.y
        seen = set()
        for m in obs.get("mechs", []):
            if m.get("kind") == "npc":
                self.items[m["id"]] = {"cat": "npcs", "pos": m["pos"], "t": t, "hp": m.get("hp", 350)}
                seen.add(m["id"])
        for kind in ("cubes", "pickups", "crates"):
            for it in obs.get(kind, []) or []:
                d = {"cat": kind, "pos": it["pos"], "t": t}
                if kind == "pickups":
                    d["kind"] = it.get("kind")
                if kind == "crates":
                    d["hp"] = it.get("hp", 300)
                self.items[it["id"]] = d
                seen.add(it["id"])
        # forget items that should be visible right here but are not (taken or destroyed);
        # NPCs wander, so their memory is short
        for k in [k for k, v in self.items.items()
                  if k not in seen and (math.hypot(v["pos"][0] - x, v["pos"][1] - y) < 25 or t - v["t"] > 90
                                        or (v["cat"] == "npcs" and t - v["t"] > 20))]:
            del self.items[k]

    def _farm_ok(self, hp_frac):
        """Shooting crates / drones is loud and pauses regen: only when healthy, early, and unseen."""
        if hp_frac <= FARM_HP or self.t > FARM_T_MAX:
            return False
        if QUIET_S > 0 and self.memory:
            if self.t - max(v["t"] for v in self.memory.values()) < QUIET_S:
                return False
        return True

    def _pick_upgrade(self, offer):
        ch = offer.get("choices", [])
        best, bi = 999, 0
        for i, c in enumerate(ch):
            r = UPGRADE_PREF.index(c["id"]) if c.get("id") in UPGRADE_PREF else 500
            if r < best:
                best, bi = r, i
        return bi

    # ------------------------------------------------------------------------------------
    # weapons / loot helpers
    def _w1(self):
        ws = self.s.get("weapons") or []
        return ws[1] if len(ws) > 1 else None

    def _gun(self):
        ws = self.s.get("weapons") or []
        for w in ws:
            if w and w.get("kind") == "gun":
                return w
        return None

    def _pickup_value(self, kind):
        """How much we want pickup `kind` given what slot 1 holds (0 = no)."""
        w1 = self._w1()
        have = w1.get("kind") if w1 else None
        if kind not in WEAPON_RANK or kind == have:
            return 0
        if have is None:
            return WEAPON_RANK[kind]
        if WEAPON_RANK[kind] > WEAPON_RANK.get(have, 0):
            return WEAPON_RANK[kind] - WEAPON_RANK.get(have, 0)
        return 0

    def _maybe_take(self, obs):
        t, x, y, z = self.t, self.x, self.y, self.z
        w1 = self._w1()
        if w1 and w1.get("kind") == "laser" and fin(w1.get("charge", 0)) > 0:
            return None
        best, bv = None, 0
        for p in obs.get("pickups", []):
            if t - self.take_fail.get(p["id"], -99) < 1.0:
                continue
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) < self.reach - 0.4 and abs(p["pos"][2] - z) < 3.8:
                v = self._pickup_value(p.get("kind"))
                if v > bv:
                    bv, best = v, p
        if best is None:
            return None
        return (best["id"], 1 if w1 is not None else None)

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

    def _edge_time(self, obs, px, py):
        """Match time at which the moving zone edge reaches point p (inf if never this phase)."""
        zn = obs["zone"]
        cx, cy, r, ncx, ncy, nr, safe_r = self._zone(obs)
        t = self.t
        if math.hypot(px - cx, py - cy) > r:
            return t - 1.0
        if math.hypot(px - ncx, py - ncy) <= nr:
            return float("inf")
        ss, se = zn.get("shrink_start", t), zn.get("shrink_end", t)
        if se <= ss:
            return se
        for k in range(1, 21):
            tau = ss + (se - ss) * k / 20.0
            if tau < t:
                continue
            f = (tau - ss) / (se - ss)
            if math.hypot(px - (cx + (ncx - cx) * f), py - (cy + (ncy - cy) * f)) > r + (nr - r) * f:
                return tau
        return se

    def _zone_urgency(self, obs):
        """Seconds of slack before the zone edge reaches us (negative = already outside)."""
        cx, cy, r, ncx, ncy, nr, safe_r = self._zone(obs)
        if not self.s.get("in_zone", True):
            return -1.0
        te = self._edge_time(obs, self.x, self.y)
        if te == float("inf"):
            return 999.0
        need = max(0.0, math.hypot(self.x - ncx, self.y - ncy) - safe_r) / (self.walk_speed * 0.8)
        return (te - self.t) - need

    def _spot_zone_ok(self, obs, px, py, extra=0.0):
        """Can we stand at p for `extra` seconds and still reach safety in time?"""
        cx, cy, r, ncx, ncy, nr, safe_r = self._zone(obs)
        te = self._edge_time(obs, px, py)
        if te == float("inf"):
            return True
        need = max(0.0, math.hypot(px - ncx, py - ncy) - safe_r) / (self.walk_speed * 0.8)
        travel = math.hypot(px - self.x, py - self.y) / (self.walk_speed * 0.8)
        return (te - self.t) - need - travel - extra > 8

    # ------------------------------------------------------------------------------------
    # threats
    def _threats(self, obs):
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
            if "lightning" in ws:
                w += 0.2
            if m["id"] in self.attackers and self.t - self.attackers[m["id"]] < 6:
                w += 0.7
            out.append((m, w, True))
        for k, m in self.memory.items():
            if k in listed:
                continue
            age = self.t - m["t"]
            if age < 15:
                out.append((m, 0.6 * (1 - age / 15), False))
        if self.t - self.last_dmg_t < 3 and self.last_dmg_bearing is not None and not any(lv for _, _, lv in out):
            b = self.last_dmg_bearing
            ph = {"id": -1, "pos": [self.x + 90 * math.cos(b), self.y + 90 * math.sin(b), self.z], "vel": [0, 0, 0],
                  "aim": [wrap(b + PI), 0], "weapons": []}
            out.append((ph, 0.8, False))
        out.sort(key=lambda a: math.hypot(a[0]["pos"][0] - self.x, a[0]["pos"][1] - self.y))
        return out[:6]

    def _mortar_impacts(self, obs):
        pts = []
        g = self.mortar_g
        for p in obs.get("projectiles", []):
            if p.get("kind") != "mortar":
                continue
            px, py, pz = p["pos"]
            vx, vy, vz = p["vel"]
            a, b, c = -0.5 * g, vz, pz - self.z
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tt = (-b - math.sqrt(disc)) / (2 * a)
            if tt < 0 or tt > 8:
                continue
            pts.append((px + vx * tt, py + vy * tt, tt))
        return pts

    def _power(self):
        """Rough fighting strength of our mech (hp x damage multiplier x weapon bonus)."""
        s = self.s
        w1 = self._w1()
        k = 1.0
        if w1:
            k += {"laser": 0.5, "lightning": 0.4, "mortar": 0.25}.get(w1.get("kind"), 0)
        dm = 1 + 0.04 * min(s.get("cubes", 0), 10)
        return s["hp"] * dm * k

    @staticmethod
    def _enemy_power(m):
        hp = m.get("hp") or m.get("max_hp") or 1400
        k = 1.0
        for w in m.get("weapons") or []:
            k += {"laser": 0.5, "lightning": 0.4, "mortar": 0.25}.get(w, 0)
        return hp * (1 + 0.04 * min(m.get("cubes") or 0, 10)) * k

    # ------------------------------------------------------------------------------------
    def _movement(self, obs, heavy_ok):
        move, boost, jump = self._movement_core(obs, heavy_ok)
        try:
            return self._evasive(obs, move, boost, jump)
        except Exception:
            self._err("evasive")
            return move, boost, jump

    def _evasive(self, obs, move, boost, jump):
        """Strafe across the line of fire with random reversals and hop at random, all the time.
        Bullets fly 252 m/s (0.4-0.9 s at range): an enemy leading on our current velocity misses
        when we reverse; hops shift the vertical aim point."""
        t, x, y = self.t, self.x, self.y
        s = self.s
        if t < self.unstick_until or not STRAFE:
            return move, boost, jump
        grounded = bool(s.get("grounded", True))
        kind = self.goal_kind
        live = [m for m, w, lv in (self._cached_threats or []) if lv]
        threatened = bool(live) or t - self.last_dmg_t < 3
        # the direction shots come from
        if live:
            m = live[0]
            ref = math.atan2(y - m["pos"][1], x - m["pos"][0])
        elif t - self.last_dmg_t < 3 and self.last_dmg_bearing is not None:
            ref = self.last_dmg_bearing + PI
        elif move[0] or move[1]:
            ref = math.atan2(move[1], move[0])
        else:
            ref = s["aim"][0]
        if t > self.strafe_flip_t:
            # new leg of the weave: flip side (usually), random angle off the perpendicular,
            # random length, sometimes a short dead stop: no rhythm for a leading aimbot to learn
            if self.rng.random() < 0.8:
                self.strafe_side = -self.strafe_side
            self.strafe_ang = self.rng.uniform(-STRAFE_ANG, STRAFE_ANG)
            self.strafe_stop = self.rng.random() < STOP_P
            self.strafe_flip_t = t + (self.rng.uniform(0.12, 0.25) if self.strafe_stop else self.rng.uniform(0.25, 0.8))
        side = self.strafe_side
        a = ref + side * (PI / 2 + getattr(self, "strafe_ang", 0.0))
        px, py = math.cos(a), math.sin(a)
        M = self.map
        if M is not None and not M.walkable(x + px * 5, y + py * 5):
            self.strafe_side = -side
            px, py = -px, -py
        mx, my = move
        mag = math.hypot(mx, my)
        if kind in ("zone!",):
            amp = 0.35
        elif kind in ("mortar!", "dodge"):
            amp = 0.3
        elif threatened:
            amp = 0.0 if getattr(self, "strafe_stop", False) else 1.0
        elif kind in ("pickup", "drop", "cube"):
            amp = 0.4
        else:
            amp = STRAFE_IDLE
        if mag < 0.05:
            # holding a spot: weave around it, pulled back to the anchor
            ax, ay = getattr(self, "anchor", None) or (x, y)
            if self.goal is not None:
                ax, ay = self.goal
            self.anchor = (ax, ay)
            bx, by = (ax - x) * 0.25, (ay - y) * 0.25
            nx, ny = px * amp + bx, py * amp + by
        else:
            nx, ny = mx + px * amp, my + py * amp
        n = math.hypot(nx, ny)
        if n > 1:
            nx, ny = nx / n, ny / n
        # strafe at full speed: boost on the ground while under fire, with an energy reserve
        # (lightning shots and wall climbs need energy too)
        if threatened and STRAFE_BOOST and grounded and kind not in ("zone!",):
            w1 = self._w1()
            reserve = 45.0 if (w1 and w1.get("kind") == "lightning") else 30.0
            charging = w1 is not None and w1.get("kind") == "laser" and fin(w1.get("charge", 0)) > 0
            e = s.get("energy", 0)
            if self.boosting and e < reserve:
                self.boosting = False
            elif not self.boosting and e > reserve + 30:
                self.boosting = True
            if self.boosting and not charging and (nx or ny):
                boost = True
        # random hops (free: no energy); never while racing the zone on boost
        if grounded and kind != "zone!" and not (boost and kind == "zone!"):
            p = JUMP_P_THREAT if threatened else JUMP_P_IDLE
            if self.rng.random() < p:
                jump = True
        if not grounded and kind != "zone!":
            boost = False  # boost in the air is a jet: don't burn energy by accident
        return (nx, ny), boost, jump

    def _movement_core(self, obs, heavy_ok):
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
        engaged = t - self.last_dmg_t < 2.5
        live = [(m, w) for m, w, lv in threats if lv]
        if live or engaged:
            self.contact_pos, self.contact_t = (x, y), t
        boost = False

        # --- stuck recovery -------------------------------------------------------------
        if t < self.unstick_until:
            return self.unstick_dir, energy > 5, bool(s.get("grounded", True))
        if len(self.hist) >= 6 and all(h[3] for h in self.hist[-6:]):
            ht, hx, hy, _ = self.hist[-6]
            if math.hypot(x - hx, y - hy) < 1.0 and t - ht > 0.4:
                a = self.rng.uniform(-PI, PI)
                base = self._goal_dir() or (math.cos(a), math.sin(a))
                self.stuck_times = [st for st in getattr(self, "stuck_times", []) if t - st < 6] + [t]
                lm = self.last_move
                if M is not None and lm is not None and self.path:
                    fx, fy = x + lm[0] * 4.5, y + lm[1] * 4.5
                    if M.nidx(fx, fy) != M.nidx(x, y):
                        M.mark_blocked(fx, fy)
                if len(self.stuck_times) >= 2 and energy > 40:
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

        # --- dodges -----------------------------------------------------------------------
        for ix, iy, tt in impacts:
            fx, fy = x + s["vel"][0] * tt, y + s["vel"][1] * tt
            if math.hypot(fx - ix, fy - iy) < 11:
                ax, ay = x - ix, y - iy
                n = math.hypot(ax, ay) or 1.0
                self.goal_kind = "mortar!"
                return (ax / n, ay / n), energy > 5, False
        dodge = None
        for m, w in live:
            if not m.get("charging"):
                continue
            bx, by = x - m["pos"][0], y - m["pos"][1]
            bearing = math.atan2(by, bx)
            if abs(wrap(bearing - m["aim"][0])) < 0.15 and math.hypot(bx, by) < 300:
                hide = self._duck_spot(m) if M is not None else None
                if hide is not None:
                    dx, dy = hide[0] - x, hide[1] - y
                    n = math.hypot(dx, dy) or 1.0
                    dodge = (dx / n, dy / n)
                    break
                if not DODGE_STRAFE:
                    continue
                if t > self.dodge_switch_t:
                    self.dodge_side = self.rng.choice((-1.0, 1.0))
                    self.dodge_switch_t = t + self.rng.uniform(0.3, 0.7)
                dodge = (-math.sin(bearing) * self.dodge_side, math.cos(bearing) * self.dodge_side)
                break

        # --- choose goal --------------------------------------------------------------------
        urgent = slack < 6
        kind, goal = "", None
        fight_target = None
        if live:
            fight_target = self._fight_target(live, hp_frac)
        self.fight_target = fight_target
        if urgent:
            kind = "zone!"
            goal = self._safe_point(ncx, ncy, safe_r)
            boost = energy > 15 and (slack < 3 or not s.get("in_zone", True))
        elif live or engaged:
            kind = "fight" if fight_target is not None else "evade"
            goal = self._best_spot(obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok, fight_target)
            boost = engaged and energy > 35 and fight_target is None
        else:
            loot = None
            if hp_frac > 0.35 or t - self.last_dmg_t > 8:
                loot = self._loot_goal(obs, threats, hp_frac)
            if loot is not None:
                kind, goal = loot
            elif math.hypot(x - ncx, y - ncy) > safe_r + 2 and slack < 60:
                kind = "zone"
                goal = self._safe_point(ncx, ncy, safe_r)
            else:
                kind = "hide"
                if t - self.goal_score_t > 1.0 or self.goal is None or self.goal_kind != "hide":
                    goal = self._best_spot(obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok, None)
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
        return mv, boost, False

    def _duck_spot(self, m):
        """Closest spot within a few meters that a charging enemy cannot see (None if none)."""
        M = self.map
        x, y = self.x, self.y
        mx, my, mz = m["pos"]
        if not M.los(mx, my, mz + 8, x, y, self.z + 4.5):
            return None  # already blocked: stay
        best, bd = None, 1e9
        for rad in (4.0, 7.0, 10.0):
            for i in range(10):
                a = TAU * i / 10
                px, py = x + rad * math.cos(a), y + rad * math.sin(a)
                if not M.walkable(px, py) or not M.walk_line(x, y, px, py):
                    continue
                gz = M.ground(px, py)
                if M.los(mx, my, mz + 8, px, py, gz + 4.5) or M.los(mx, my, mz + 8, px, py, gz + 8.0):
                    continue
                if rad < bd:
                    bd, best = rad, (px, py)
            if best is not None:
                return best
        return None

    def _fight_target(self, live, hp_frac):
        """Pick a lone enemy we can beat, or None (then we evade)."""
        x, y = self.x, self.y
        # a badly hurt enemy nearby is worth chasing down: every kill is a place
        if PURSUE and hp_frac > 0.4:
            weak = [m for m, w in live if (m.get("hp") or 9999) < 0.35 * (m.get("max_hp") or 1400)
                    and math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 200]
            others = [m for m, w in live if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 150]
            if weak and len(others) <= 2:
                return min(weak, key=lambda m: m.get("hp") or 9999)
        near = [(m, w) for m, w in live if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < 170]
        if not near:
            return None
        if len(near) > 1:
            # two close enemies: only stand if one of them is nearly dead
            weak = min(near, key=lambda a: a[0].get("hp") or 9999)
            if (weak[0].get("hp") or 9999) < 250 and hp_frac > 0.5:
                return weak[0]
            return None
        m = near[0][0]
        mine, theirs = self._power(), self._enemy_power(m)
        if mine > theirs * FIGHT_RATIO or (m.get("hp") or 9999) < 300:
            return m
        return None

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

    # ------------------------------------------------------------------------------------
    def _loot_goal(self, obs, threats, hp_frac):
        """Nearest worthwhile loot when nobody is around: supply drop, weapon, cube, crate to shoot."""
        t, x, y = self.t, self.x, self.y
        M = self.map
        ghosts = [m for m, w, lv in threats]

        def risky(px, py):
            for m in ghosts:
                if math.hypot(m["pos"][0] - px, m["pos"][1] - py) < 90:
                    return True
            return False

        best, bscore = None, -1e9
        # supply drops: full heal + 3 cubes
        for d in obs.get("drops", []) or []:
            px, py = d["pos"][0], d["pos"][1]
            dist = math.hypot(px - x, py - y)
            eta = dist / self.walk_speed
            if dist > 260 or eta + 4 < d.get("lands_in", 0) - 20:
                continue
            holder = d.get("holder")
            if holder is not None and holder != self.slot and d.get("progress", 0) > 0.5:
                continue
            if risky(px, py) or not self._spot_zone_ok(obs, px, py, 8):
                continue
            sc = 120 - dist * 0.35 + (1 - hp_frac) * 60
            if sc > bscore:
                bscore, best = sc, ("drop", (px, py))
        cap = 140 if t < 200 else 90
        for k, it in self.items.items():
            px, py, pz = it["pos"]
            dist = math.hypot(px - x, py - y)
            if dist > cap or abs(pz - self.z) > 5:
                continue
            if risky(px, py) or not self._spot_zone_ok(obs, px, py, 3):
                continue
            if M is not None and not M.walkable(px, py) and M.nearest_walkable(px, py, 1) < 0:
                continue
            if it["cat"] == "cubes":
                sc = 80 - dist * 0.5
            elif it["cat"] == "pickups":
                v = self._pickup_value(it.get("kind"))
                if v <= 0 or t - self.take_fail.get(k, -99) < 5:
                    continue
                sc = 40 + 25 * v - dist * 0.5
            else:
                continue
            if sc > bscore:
                bscore, best = sc, (it["cat"][:-1], (px, py))
        # crates / NPC drones are shot from where we stand (see _combat): walk until we have a clear shot
        if best is None and self._farm_ok(hp_frac):
            muzzle = self.z + self.s.get("stats", {}).get("muzzle_height", 7.0)
            for k, it in self.items.items():
                cat = it["cat"]
                if cat not in ("crates", "npcs"):
                    continue
                px, py, pz = it["pos"]
                dist = math.hypot(px - x, py - y)
                if dist > 170 or abs(pz - self.z) > 6:
                    continue
                if risky(px, py) or not self._spot_zone_ok(obs, px, py, 5):
                    continue
                clear = dist < (95 if cat == "npcs" else 110) and (
                    M is None or M.los(x, y, muzzle, px, py, pz + (1.75 if cat == "npcs" else 0.8)))
                sc = -dist + (25 if cat == "npcs" else 0)
                if sc > bscore:
                    bscore = sc
                    best = (cat[:-1], (x, y) if clear else (px, py))
        return best

    # ------------------------------------------------------------------------------------
    def _best_spot(self, obs, threats, impacts, ncx, ncy, safe_r, cx, cy, r, heavy_ok, fight_target):
        """Score nearby walkable spots: out of enemy sight, inside the safe circle, near cover,
        away from enemies and NPCs, short to reach. In a fight, keep a line to the target."""
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
        dx, dy = ncx - x, ncy - y
        d = math.hypot(dx, dy)
        if d > 1:
            for rad in (15.0, 35.0):
                cands.append((x + dx / d * min(rad, d), y + dy / d * min(rad, d)))
        n_near = len(cands)
        live = [(m, w) for m, w, lv in threats if lv][:3]
        if FAR_ESCAPE and live and fight_target is None and heavy_ok:
            for rad in (70.0, 100.0):
                for i in range(8):
                    a = off + TAU * i / 8
                    cands.append((x + rad * math.cos(a), y + rad * math.sin(a)))
        ghosts = [(m, w) for m, w, lv in threats if not lv][:3]
        npcs = [m for m in obs.get("mechs", []) if m.get("kind") == "npc"]
        w1 = self._w1()
        wk = w1.get("kind") if w1 else None
        pref = {"laser": 110.0, "lightning": 32.0, "mortar": 120.0}.get(wk, 85.0)

        best, bs = None, -1e18
        for i, (px, py) in enumerate(cands):
            if M is not None:
                if not M.walkable(px, py):
                    continue
                if 0 < i < n_near and not M.walk_line(x, y, px, py):
                    continue
                gz = M.ground(px, py)
            else:
                gz = self.z
            sc = 0.0
            dz = math.hypot(px - ncx, py - ncy) - safe_r
            if dz > 0:
                sc -= 2.5 * dz
            dc = math.hypot(px - cx, py - cy) - (r - 5)
            if dc > 0:
                sc -= 20 * dc
            for m, w in live:
                mx, my, mz = m["pos"]
                dd = math.hypot(px - mx, py - my)
                is_t = fight_target is not None and m.get("id") == fight_target.get("id")
                if dd < 260:
                    seen = True
                    if M is not None:
                        seen = M.los(mx, my, mz + 8, px, py, gz + 8.5) or M.los(mx, my, mz + 8, px, py, gz + 4.5)
                    if is_t:
                        sc += 25 if seen else -10
                    elif seen:
                        sc -= 60 * w * (1.0 if dd < 200 else 0.5)
                if is_t:
                    sc -= abs(dd - pref) * 0.4
                else:
                    sc -= w * max(0.0, 150 - dd) * 1.0
            for m, w in ghosts:
                mx, my = m["pos"][0], m["pos"][1]
                dd = math.hypot(px - mx, py - my)
                sc -= w * max(0.0, 130 - dd) * 0.6
            for m in npcs:
                dd = math.hypot(px - m["pos"][0], py - m["pos"][1])
                if dd < 55:
                    sc -= (55 - dd) * 0.8
            for ix, iy, _ in impacts:
                if math.hypot(px - ix, py - iy) < 12:
                    sc -= 80
            if safe_r > 20 and fight_target is None:
                sc += 0.15 * min(math.hypot(px - ncx, py - ncy), safe_r)
            cp = getattr(self, "contact_pos", None)
            if RELOCATE and cp is not None and not live and self.t - self.contact_t < 40:
                sc -= max(0.0, 60 - math.hypot(px - cp[0], py - cp[1])) * 0.5
            if M is not None:
                sc += 2.0 * min(M.cover_at(px, py), 8)
                if fight_target is None:
                    sc += CANOPY_W * min(M.canopy_at(px, py), 2)
            sc -= 0.12 * math.hypot(px - x, py - y)
            if i == 1:
                sc += 6
            if sc > bs:
                bs, best = sc, (px, py)
        return best

    # ------------------------------------------------------------------------------------
    def _steer_to(self, goal, heavy_ok):
        M = self.map
        x, y, t = self.x, self.y, self.t
        gx, gy = goal
        d = math.hypot(gx - x, gy - y)
        if d < 1.2:
            self.path = None
            return (0.0, 0.0)
        if M is None or (d < 60 and M.walk_line(x, y, gx, gy)):
            self.path = None
            sp = 1.0 if d > 4 else max(0.35, d / 4)
            return ((gx - x) / d * sp, (gy - y) / d * sp)
        need_new = (self.path is None or self.path_goal is None
                    or math.hypot(self.path_goal[0] - gx, self.path_goal[1] - gy) > 12
                    or t - self.path_t > 4.0)
        if need_new:
            if self.astar is None or self.astar_goal is None or math.hypot(self.astar_goal[0] - gx, self.astar_goal[1] - gy) > 12:
                sk = M.nearest_walkable(x, y, 3)
                gk = M.nearest_walkable(gx, gy, 10)
                self.astar = nav.AStar(M, sk, gk, self._zone_now)
                self.astar_goal = (gx, gy)
            if self.astar.step(4000, 0.008 if heavy_ok else 0.003):
                if not self.astar.path and M.reset_marks():
                    # our learned blocks cut every route: forget them and search again next tick
                    self.astar = None
                    return ((gx - x) / d, (gy - y) / d)
                self.path = (self.astar.path + [(gx, gy)]) if self.astar.path else None
                self.path_i = 0
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
        return ((gx - x) / d, (gy - y) / d)

    # ------------------------------------------------------------------------------------
    # combat
    def _combat(self, obs):
        s = self.s
        t, x, y, z = self.t, self.x, self.y, self.z
        M = self.map
        st = s.get("stats", {})
        muzzle = z + st.get("muzzle_height", 7.0)
        gun = self._gun()
        w1 = self._w1()
        wk = w1.get("kind") if w1 else None
        gstats = (gun or {}).get("stats", self.gun)
        gspeed = gstats.get("speed", 252.0)
        grange = gstats.get("range", 230.0)
        engaged = t - self.last_dmg_t < 4.0
        hp_frac = s["hp"] / max(1.0, s.get("max_hp", 1400))
        mechs_visible = [m for m in obs.get("mechs", []) if m.get("kind") == "mech"]
        alone = not mechs_visible and t - self.last_dmg_t > 3
        max_range = grange
        if wk == "laser":
            max_range = max(max_range, w1.get("stats", {}).get("range", 280.0))

        # ---- choose a target ------------------------------------------------------------
        best, bscore = None, -1e9
        ft = getattr(self, "fight_target", None)
        for m in obs.get("mechs", []):
            mx, my, mz = m["pos"]
            d = math.hypot(mx - x, my - y)
            if d > max_range * 0.97:
                continue
            is_mech = m.get("kind") == "mech"
            scale = 1 + 0.05 * (m.get("cubes") or 0)
            cz = mz + (4.5 * scale if is_mech else 1.75)
            if M is not None and not M.los(x, y, muzzle, mx, my, cz):
                continue
            sc = -d
            if is_mech:
                sc += 400
                if m["id"] in self.attackers and t - self.attackers[m["id"]] < 5:
                    sc += 150
                if ft is not None and ft.get("id") == m["id"]:
                    sc += 200
                if m["id"] == self.target_id:
                    sc += STICKY  # finish what we started: spread damage just regenerates
                hp = m.get("hp")
                if hp is not None:
                    sc += (1 - hp / max(1.0, m.get("max_hp") or 1400)) * FOCUS_W
                aiming_at_us = abs(wrap(math.atan2(y - my, x - mx) - m["aim"][0])) < 0.35
                if abs(wrap(math.atan2(y - my, x - mx) - m["aim"][0])) > PI / 2:
                    sc += 60  # its back is to us: x1.5 damage
                if not (FIRE_ALWAYS or engaged or d < CLOSE_FIRE or (aiming_at_us and d < 160) or (hp or 9999) < 450 or ft is not None):
                    continue
            else:
                attacking = m["id"] in self.attackers and t - self.attackers[m["id"]] < 5
                if not (attacking or (alone and self._farm_ok(hp_frac) and d < 130)):
                    continue
            if sc > bscore:
                bscore, best = sc, (m, cz, d, "actor")
        if best is None and alone and self._farm_ok(hp_frac) and t - self.last_dmg_t > 5:
            # crates: 1 cube each, half drop a weapon
            for c in obs.get("crates", []):
                cx_, cy_, cz_ = c["pos"]
                d = math.hypot(cx_ - x, cy_ - y)
                if d > min(grange, 150):
                    continue
                if M is not None and not M.los(x, y, muzzle, cx_, cy_, cz_ + 0.8):
                    continue
                if not self._spot_zone_ok(obs, cx_, cy_, 2):
                    continue
                sc = -d - c.get("hp", 300) * 0.1
                if sc > bscore:
                    bscore, best = sc, ({"id": c["id"], "pos": c["pos"], "vel": [0, 0, 0], "kind": "crate"}, cz_ + 0.8, d, "crate")

        aim, fire0, fire1, reload_ = None, False, False, False
        if best is None:
            self.target_id = None
            aim = self._idle_aim(obs)
            # a laser still charging with nobody to shoot: release it after a while
            if wk == "laser" and fin(w1.get("charge", 0)) > 0 and t - self.target_seen_t < 1.2:
                fire1 = True
        else:
            m, cz, d, cat = best
            self.target_id = m.get("id")
            self.target_seen_t = t
            mx, my, mz = m["pos"]
            vx, vy, vz = m.get("vel", [0, 0, 0])
            is_mech = m.get("kind") == "mech"
            svx, svy, svz = s.get("vel", [0, 0, 0])
            ox, oy = x + svx * 0.08, y + svy * 0.08   # where we'll be when the shot leaves
            om = muzzle + svz * 0.08
            tx, ty, tz = self._predict(m, cz, 0.0)
            for _ in range(3):
                tt = math.hypot(tx - ox, ty - oy) / gspeed + 0.08
                tx, ty, tz = self._predict(m, cz, tt)
            gyaw = math.atan2(ty - oy, tx - ox)
            gpitch = math.atan2(tz - om, math.hypot(tx - ox, ty - oy))
            # hitscan aim (laser / lightning): the beam leaves after this decision is applied
            hx, hy, hz = self._predict(m, cz, 0.08)
            hyaw = math.atan2(hy - oy, hx - ox)
            hpitch = math.atan2(hz - om, max(1.0, math.hypot(hx - ox, hy - oy)))
            aim = (gyaw, gpitch)
            cur = s.get("aim", [gyaw, gpitch])
            tol = max(0.012, math.atan2(2.0 if is_mech else 1.2, max(d, 1.0)))

            # ---- slot 1 -------------------------------------------------------------
            if wk == "laser" and w1:
                ls = w1.get("stats", {})
                full = ls.get("charge", 0.75)
                ch = fin(w1.get("charge", 0))
                ready = fin(w1.get("cooldown", 0)) <= 0
                if ready and d < ls.get("range", 280) * 0.95:
                    if ch < full - 0.06:
                        # charging halves our speed: never start one while racing the zone
                        fire1 = ch > 0 or self.goal_kind != "zone!"
                    else:
                        aim = (hyaw, hpitch)
                        on = abs(wrap(cur[0] - hyaw)) < tol and abs(cur[1] - hpitch) < tol * 1.5 + 0.01
                        fire1 = not on  # release (fire) when on target
            elif wk == "lightning" and w1:
                ls = w1.get("stats", {})
                if d < ls.get("range", 45) * 0.95:
                    aim = (hyaw, hpitch)
                    cone = math.radians(ls.get("cone_deg", 30)) * 0.4
                    if fin(w1.get("cooldown", 0)) <= 0 and abs(wrap(cur[0] - hyaw)) < cone and s.get("energy", 0) > 6:
                        fire1 = True
            elif wk == "mortar" and w1 and cat == "actor" and 18 < d < 230:
                sol = self._mortar_solution(m, mz + 1.0, w1)
                if sol is not None and fin(w1.get("cooldown", 0)) <= 0.15 and (gun is None or gun.get("reload", 0) > 0 or d > 120 or not is_mech or self._gun_bad(gun)):
                    aim = sol
                    if fin(w1.get("cooldown", 0)) <= 0 and abs(wrap(cur[0] - sol[0])) < 0.03 and abs(cur[1] - sol[1]) < 0.03:
                        fire1 = True

            # ---- gun ------------------------------------------------------------------
            if gun is not None and gun.get("reload", 0) <= 0 and gun.get("ammo", 1) > 0 and d < grange * 0.95:
                if abs(wrap(cur[0] - gyaw)) < tol and abs(cur[1] - gpitch) < tol * 1.5 + 0.01:
                    fire0 = True
                    # don't waste a laser release window on gun alignment: both share the torso
        if gun is not None and gun.get("reload", 0) <= 0:
            ammo = gun.get("ammo", 12)
            if ammo <= 0 or (best is None and ammo < gstats.get("mag", 12) * 0.5 and t - self.last_dmg_t > 2):
                reload_ = True
        return aim, fire0, fire1, reload_

    def _predict(self, m, cz, tt):
        """Where m's aim point will be in tt seconds.
        Horizontal: average drift plus the current strafe component, decayed for random reversals
        (a strafer that flips every ~LEAD_TAU s is expected to come back toward its mean).
        Vertical: ballistic if airborne, never below the ground."""
        mx, my, mz = m["pos"]
        vx, vy, vz = m.get("vel", [0, 0, 0])
        if tt <= 0:
            return mx, my, cz
        ax, ay = vx, vy
        mem = self.memory.get(m.get("id")) if m.get("kind") == "mech" else None
        vh = mem.get("vh") if mem else None
        if vh and len(vh) >= 4:
            ax = sum(h[1] for h in vh) / len(vh)
            ay = sum(h[2] for h in vh) / len(vh)
        ox, oy = vx - ax, vy - ay
        k = (LEAD_TAU / 2.0) * (1 - math.exp(-2.0 * tt / LEAD_TAU))
        px = mx + ax * tt + ox * k
        py = my + ay * tt + oy * k
        dz = 0.0
        M = self.map
        g = self.mortar_g
        if m.get("kind") == "mech" and (abs(vz) > 0.3 or (M is not None and mz > M.ground(mx, my) + 0.6)):
            gr = M.ground(px, py) if M is not None else mz
            znew = max(gr, mz + vz * tt - 0.5 * g * tt * tt)
            if M is not None and mz > M.ground(mx, my) + 3.0 and abs(vz) < 0.3:
                znew = mz  # standing on a roof / bridge
            dz = znew - mz
        return px, py, cz + dz

    @staticmethod
    def _gun_bad(gun):
        return gun is None or gun.get("ammo", 12) <= 2

    def _mortar_solution(self, m, tz, w1):
        """[yaw, pitch] to land a shell on m (low arc if it has a clear path, else high arc)."""
        ms = w1.get("stats", {})
        v = ms.get("speed", 72.0)
        g = ms.get("gravity", self.mortar_g)
        x, y = self.x, self.y
        z0 = self.z + self.s.get("stats", {}).get("height", 9.0) + 0.5
        mx, my = m["pos"][0], m["pos"][1]
        vx, vy = m.get("vel", [0, 0, 0])[:2]
        tx, ty = mx, my
        sol = None
        for _ in range(3):
            D = math.hypot(tx - x, ty - y)
            h = tz - z0
            disc = v ** 4 - g * (g * D * D + 2 * h * v * v)
            if disc < 0 or D < 1:
                return None
            sq = math.sqrt(disc)
            lo = math.atan2(v * v - sq, g * D)
            hi = math.atan2(v * v + sq, g * D)
            pitch = lo
            M = self.map
            if M is not None and not M.los(x, y, z0 + 2, tx, ty, tz + 3):
                pitch = hi
            maxp = self.s.get("stats", {}).get("max_pitch", 1.3)
            if pitch > maxp:
                pitch = lo
            flight = D / max(1.0, v * math.cos(pitch))
            tx, ty = mx + vx * flight * 0.8, my + vy * flight * 0.8
            sol = (math.atan2(ty - y, tx - x), pitch)
        return sol

    def _idle_aim(self, obs):
        t, x, y = self.t, self.x, self.y
        if t - self.last_dmg_t < 3 and self.last_dmg_bearing is not None:
            self.scan_yaw = self.last_dmg_bearing
            return (self.last_dmg_bearing, 0.0)
        thr = self._cached_threats or []
        if thr:
            m = thr[0][0]
            yaw = math.atan2(m["pos"][1] - y, m["pos"][0] - x)
            self.scan_yaw = yaw
            return (yaw, 0.0)
        for snd in obs.get("sounds", []):
            if snd.get("kind") in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") \
                    and snd.get("dist") in ("near", "mid") and snd.get("age", 9) < 1.0:
                self.scan_yaw = snd["bearing"]
                return (snd["bearing"], 0.0)
        # moving somewhere: look where we go (and sweep a bit)
        if self.goal is not None and self.goal_kind in ("zone", "zone!", "cube", "pickup", "drop", "crate"):
            dx, dy = self.goal[0] - x, self.goal[1] - y
            if math.hypot(dx, dy) > 6:
                base = math.atan2(dy, dx)
                return (wrap(base + 0.6 * math.sin(t * 2.0)), 0.0)
        if self.scan_yaw is None:
            self.scan_yaw = self.s.get("aim", [0, 0])[0]
        self.scan_yaw = wrap(self.scan_yaw + 0.28)
        return (self.scan_yaw, 0.0)


if __name__ == "__main__":
    mf.run(Bot)
