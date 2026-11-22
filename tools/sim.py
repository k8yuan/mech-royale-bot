"""Crude local match simulator for regression testing (NOT the real game).

Our bot vs a 15-mech squad that never fights itself and hunts us (1 hunter early, more later, all late).
Models: walls (nav fine grid), zone, regen, rear armor, vision + memory, gun/laser squad fire,
NPC drones, crates, cubes (+hp/+dmg, upgrade offers), weapon pickups, our slot-1 laser/lightning/mortar,
supply drops. Ignores jumping, roofs, projectiles in obs.

usage: python3 tools/sim.py [n_matches] [seed0] [--bot module] [--trace] [--hunt fast|slow]
"""
from __future__ import annotations

import gzip
import importlib
import json
import math
import random
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "tools/archive")
sys.path.insert(0, "tools")
import nav  # noqa: E402

KIT = "kit/mechfight-kit"
PI, TAU = math.pi, 2 * math.pi


def wrap(a):
    return (a + PI) % TAU - PI


def load_hello():
    with gzip.open(f"{KIT}/check/stream.jsonl.gz", "rt") as f:
        h = json.loads(f.readline())
    h["map"]["dir"] = f"{KIT}/arena"
    return h


PHYS = "--nophys" not in sys.argv
PASSIVE = "--passive" in sys.argv
UPG = ["plating", "armor", "servos", "nanites", "sensors", "gyros", "capacitor", "reactor", "gun_damage", "gun_rate",
       "gun_mag", "laser_damage", "laser_charge"]


class Sim:
    def __init__(self, M, hello, bot_cls, seed, hunt="normal", intel="omni", tough=False):
        self.intel, self.tough = intel, tough
        self.phys = PHYS
        self.last_seen_pos = None
        self.last_seen_t = -99.0
        self.M, self.hello, self.cfg = M, hello, hello["config"]
        self.rng = random.Random(seed)
        self.t = 0.0
        self.tick = 0
        self.hunt = hunt
        zc = self.cfg["zone"]
        self.zone_phases = zc["phases"]
        self.zc0 = tuple(zc["start_center"])
        self.zr0 = zc["start_radius"]
        self._plan_zone()
        self.mechs = []
        tiers = ["easy"] * 4 + ["medium"] * 4 + ["hard"] * 4 + ["expert"] * 3
        spots = self._spawns(16, 130)
        for i in range(16):
            tier = None if i == 0 else tiers[i - 1]
            acc = {"easy": 0.25, "medium": 0.4, "hard": 0.55, "expert": 0.7}.get(tier, 0) + (0.15 if tough else 0.0)
            # everyone spawns with the gun; squad members find a laser some time later (tier-dependent)
            laser = False
            laser_t = 1e9
            if tier is not None and self.rng.random() < {"easy": 0.3, "medium": 0.5, "hard": 0.7, "expert": 0.9}[tier]:
                lo, hi = {"easy": (90, 300), "medium": (50, 220), "hard": (30, 150), "expert": (20, 100)}[tier]
                laser_t = self.rng.uniform(lo, hi)
            self.mechs.append(dict(id=i, kind="mech", pos=[spots[i][0], spots[i][1], M.ground(*spots[i])], vel=[0, 0, 0],
                                   aim=[self.rng.uniform(-PI, PI), 0], hp=1400.0, max_hp=1400.0, alive=True, tier=tier, acc=acc,
                                   laser=laser, energy=100.0, last_combat=-99.0, ammo=12, cd=0.0, reload=0.0, lcd=0.0, charge=0.0,
                                   hunter=False, wander=None, cmd=[0, 0], boost=False, aim_cmd=None, fire=False, fire1=False,
                                   death_t=None, seen_us=-99.0, cubes=0, w1=None, w1cd=0.0, equip=0.0, hp_mult=1.0, armor=0.0,
                                   regen=45.0, regen_delay=5.0, speed_mult=1.0, upg={}, radius=2.5, laser_t=laser_t))
        self.me = self.mechs[0]
        # world items
        self.npcs = []
        for k, p in enumerate(self._spawns(12, 40)):
            self.npcs.append(dict(id=100 + k, kind="npc", pos=[p[0], p[1], M.ground(*p)], home=p, vel=[0, 0, 0], aim=[0, 0],
                                  hp=350.0, max_hp=350.0, alive=True, target=None, cd=0.0, radius=1.8))
        self.crates = [dict(id=1000 + k, pos=[p[0], p[1], M.ground(*p)], hp=300.0, radius=1.5) for k, p in enumerate(self._spawns(36, 15))]
        self.cubes = []
        self.pickups = [dict(id=8000 + k, kind=self.rng.choice(["mortar", "laser", "lightning"]), pos=[p[0], p[1], M.ground(*p)])
                        for k, p in enumerate(self._spawns(40, 10))]
        self.next_id = 9000
        self.next_cube = 5000
        self.drops = []
        self.shells = []
        self.bullets = []
        h = dict(hello)
        h["slot"] = 0
        self.bot = bot_cls(h)
        self.events = []
        self.seen_t = {}
        self.max_ms = 0.0
        self.sum_ms = 0.0
        self.n_dec = 0
        self.zone_dmg = 0.0
        self.deaths = []
        self.stuck_ticks = 0
        self.offer = None
        self.offers_pending = 0
        self.kills = 0
        self.drops_won = 0

    # ---- zone -------------------------------------------------------------------------
    def _plan_zone(self):
        c, r = self.zc0, self.zr0
        t = 0.0
        self.zsched = []
        for ph in self.zone_phases:
            rr = ph["radius"]
            a = self.rng.uniform(-PI, PI)
            off = self.rng.uniform(0, max(0.0, r - rr)) * 0.8
            nc = (c[0] + off * math.cos(a), c[1] + off * math.sin(a))
            k = self.M.nearest_walkable(*nc, 20)
            if k >= 0:
                nc = self.M.ncenter(k)
            ss, se = t + ph["hold_s"], t + ph["hold_s"] + ph["shrink_s"]
            self.zsched.append((t, ss, se, c, r, nc, rr, ph["dps"]))
            c, r, t = nc, rr, se

    def zone(self):
        t = self.t
        for (t0, ss, se, c, r, nc, rr, dps) in self.zsched:
            if t < se:
                f = 0 if t < ss else (t - ss) / (se - ss)
                cc = (c[0] + (nc[0] - c[0]) * f, c[1] + (nc[1] - c[1]) * f)
                return dict(center=list(cc), radius=r + (rr - r) * f, next_center=list(nc), next_radius=rr,
                            shrink_start=ss, shrink_end=se, dps=dps)
        last = self.zsched[-1]
        return dict(center=list(last[5]), radius=0.0, next_center=list(last[5]), next_radius=0.0,
                    shrink_start=last[2], shrink_end=last[2], dps=last[7])

    def _main_comp(self):
        M = self.M
        if getattr(M, "_main", None) is None:
            nnx, nny = M.nnx, M.nny
            comp = bytearray(nnx * nny)
            best = None
            seen = bytearray(nnx * nny)
            for s0 in range(nnx * nny):
                if not M.walk[s0] or seen[s0]:
                    continue
                st, cells = [s0], []
                seen[s0] = 1
                while st:
                    k = st.pop()
                    cells.append(k)
                    i, j = k % nnx, k // nnx
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < nnx and 0 <= jj < nny:
                            kk = jj * nnx + ii
                            if M.walk[kk] and not seen[kk]:
                                seen[kk] = 1
                                st.append(kk)
                if best is None or len(cells) > len(best):
                    best = cells
            for k in best:
                comp[k] = 1
            M._main = comp
        return M._main

    def _spawns(self, n, sep):
        M = self.M
        main = self._main_comp()
        z = self.zone()
        out = []
        tries = 0
        while len(out) < n and tries < 20000:
            tries += 1
            a = self.rng.uniform(-PI, PI)
            rr = z["radius"] * 0.8 * math.sqrt(self.rng.random())
            x, y = z["center"][0] + rr * math.cos(a), z["center"][1] + rr * math.sin(a)
            k = M.nidx(x, y)
            if k < 0 or not main[k]:
                continue
            if all(math.hypot(x - p[0], y - p[1]) > sep - tries / 300 for p in out):
                out.append((x, y))
        return out

    # ---- helpers ------------------------------------------------------------------------
    def los_pt(self, a, b, za=8.0, zb=4.5):
        return self.M.los(a[0], a[1], a[2] + za, b[0], b[1], b[2] + zb)

    def los(self, a, b, za=8.0, zb=4.5):
        return self.los_pt(a["pos"], b["pos"], za, zb)

    def sees_pos(self, a, p, zb=4.5):
        dx, dy = p[0] - a["pos"][0], p[1] - a["pos"][1]
        d = math.hypot(dx, dy)
        if d > 200:
            return False
        if d > 45 and abs(wrap(math.atan2(dy, dx) - a["aim"][0])) > math.radians(60):
            return False
        if d > 45 and getattr(self.M, "canopy", None) is not None and self.M.canopy_at(p[0], p[1]) > 0:
            return False  # dense crown overhead (assumed model)
        return self.los_pt(a["pos"], p, 8.0, zb)

    def sees(self, a, b):
        return self.sees_pos(a, b["pos"], 4.5) or self.sees_pos(a, b["pos"], 8.5)

    def damage(self, victim, attacker, amount, weapon):
        if not victim["alive"]:
            return
        if victim.get("kind") == "crate":
            victim["hp"] -= amount
            if attacker is self.me:
                self.events.append({"type": "hit", "target": victim["id"], "amount": amount, "weapon": weapon})
            return
        bearing = math.atan2(attacker["pos"][1] - victim["pos"][1], attacker["pos"][0] - victim["pos"][0])
        rear = False
        if victim.get("kind") == "mech":
            rear = abs(wrap(bearing - victim["aim"][0])) > PI / 2
            if rear:
                amount *= 1.5
            amount *= (1 - victim.get("armor", 0))
        victim["hp"] -= amount
        victim["last_combat"] = self.t
        attacker["last_combat"] = self.t
        if victim.get("kind") == "npc" and attacker.get("kind") == "mech":
            victim["target"] = attacker["id"]
        if victim is self.me:
            self.events.append({"type": "damaged", "amount": amount, "weapon": weapon, "bearing": bearing, "by": attacker["id"], "rear": rear})
        if attacker is self.me:
            victim["hit_by_us_t"] = self.t
            self.events.append({"type": "hit", "target": victim["id"], "amount": amount, "weapon": weapon})

    def dmg_mult(self, m):
        return 1 + 0.04 * min(m.get("cubes", 0), 10)

    def move(self, m, dt, speed=11.7):
        M = self.M
        cmd = m["cmd"]
        air = m.get("zoff", 0.0) > 0.01 or m.get("vz", 0.0) > 0
        sm = m.get("speed_mult", 1.0)
        sp = (19.5 if (m["boost"] and m["energy"] > 1) else speed) * sm
        if m.get("charge", 0) > 0 and m is self.me:
            sp *= 0.5
        if m["boost"] and m["energy"] > 1 and (cmd[0] or cmd[1]):
            m["energy"] -= 17 * dt
        elif m.get("e_spent", -9) < self.t - 0.6:
            m["energy"] = min(100.0, m["energy"] + 18 * dt)
        tvx, tvy = cmd[0] * sp, cmd[1] * sp
        vx, vy = m["vel"][0], m["vel"][1]
        dvx, dvy = tvx - vx, tvy - vy
        dv = math.hypot(dvx, dvy)
        mx = (12.0 if air else 58.5) * dt
        if dv > mx:
            dvx, dvy = dvx / dv * mx, dvy / dv * mx
        vx, vy = vx + dvx, vy + dvy
        x, y = m["pos"][0], m["pos"][1]
        nx, ny = x + vx * dt, y + vy * dt
        climbing = m is self.me and m.get("jump_cmd") and m["boost"] and m["energy"] > 1
        if climbing and not self._free(nx, ny):
            k = M.fidx(nx, ny)
            if k >= 0 and M.ftop[k] - M.fground[k] < 25:
                m["energy"] -= 15 * dt
                m["pos"] = [nx, ny, M.ground(nx, ny)]
                m["vel"] = [vx, vy, 0.0]
                return
        if not self._free(x, y):
            x, y = nx, ny
        elif self._free(nx, ny):
            x, y = nx, ny
        elif self._free(nx, y):
            x, vy = nx, 0.0
        elif self._free(x, ny):
            y, vx = ny, 0.0
        else:
            vx, vy = 0.0, 0.0
        vz = m.get("vz", 0.0)
        zoff = m.get("zoff", 0.0)
        if m.get("jump_cmd") and zoff <= 0.01 and vz <= 0:
            vz = 11.0
            m["jump_cmd"] = False
        if zoff > 0.0 or vz > 0:
            zoff += vz * dt
            vz -= 22.0 * dt
            if zoff <= 0:
                zoff, vz = 0.0, 0.0
        m["vz"], m["zoff"] = vz, zoff
        m["pos"] = [x, y, M.ground(x, y) + zoff]
        m["vel"] = [vx, vy, vz]

    def _free(self, x, y):
        M = self.M
        for dx, dy in ((0, 0), (2.4, 0), (-2.4, 0), (0, 2.4), (0, -2.4), (1.7, 1.7), (-1.7, 1.7), (1.7, -1.7), (-1.7, -1.7)):
            k = M.fidx(x + dx, y + dy)
            if k < 0 or M.fblocked[k]:
                return False
        return True

    # ---- squad AI ------------------------------------------------------------------------
    def squad_think(self, m):
        me = self.me
        if not m["laser"] and self.t >= m["laser_t"]:
            m["laser"] = True
        z = self.zone()
        d = math.hypot(me["pos"][0] - m["pos"][0], me["pos"][1] - m["pos"][1])
        cansee = me["alive"] and d < 200 and self.sees(m, me)
        if cansee:
            for o in self.mechs[1:]:
                o["seen_us"] = self.t
            self.last_seen_pos, self.last_seen_t = (me["pos"][0], me["pos"][1]), self.t
        known = self.t - m["seen_us"] < 5 or (m["hunter"] and self.intel == "omni")
        if self.tough:
            # squad members farm too: slowly gain cubes
            want = min(8, int(self.t / 35) + (m["id"] % 3))
            if m["cubes"] < want:
                m["cubes"] += 1
                new = 1400 * (1 + 0.04 * m["cubes"])
                m["hp"] += new - m["max_hp"]
                m["max_hp"] = new
            if m["hp"] < 0.35 * m["max_hp"]:
                m["retreat"] = True
            elif m["hp"] > 0.85 * m["max_hp"]:
                m["retreat"] = False
        ncx, ncy = z["next_center"]
        dz = math.hypot(m["pos"][0] - ncx, m["pos"][1] - ncy)
        goal = None
        # where hunters think we are
        if self.intel == "omni":
            target_pos = (me["pos"][0], me["pos"][1])
        elif self.last_seen_pos is not None and self.t - self.last_seen_t < 30:
            target_pos = self.last_seen_pos
        else:
            target_pos = None
        if m.get("retreat") and me["alive"] and d < 220:
            a = math.atan2(m["pos"][1] - me["pos"][1], m["pos"][0] - me["pos"][0])
            goal = (m["pos"][0] + 40 * math.cos(a), m["pos"][1] + 40 * math.sin(a))
        elif dz > z["next_radius"] * 0.8 + 5 and (not m["hunter"] or dz > z["next_radius"]):
            goal = (ncx, ncy)
        elif m["hunter"] and me["alive"] and target_pos is not None and not (cansee and d <= 70):
            goal = target_pos
            if not cansee and math.hypot(target_pos[0] - m["pos"][0], target_pos[1] - m["pos"][1]) < 8:
                self.last_seen_pos = None  # searched there, nothing
        elif m["hunter"] and me["alive"] and cansee:
            if False:
                pass
            else:
                a = math.atan2(m["pos"][1] - me["pos"][1], m["pos"][0] - me["pos"][0]) + 0.6
                goal = (me["pos"][0] + 70 * math.cos(a), me["pos"][1] + 70 * math.sin(a))
        else:
            # some squad members go for supply drops
            dr = next((dd for dd in self.drops if dd["lands_in"] <= 5 and dd["holder"] != -2), None)
            if dr is not None and m["id"] % 3 == 0:
                goal = (dr["pos"][0], dr["pos"][1])
            else:
                if m["wander"] is None or math.hypot(m["wander"][0] - m["pos"][0], m["wander"][1] - m["pos"][1]) < 5 or self.rng.random() < 0.005:
                    a = self.rng.uniform(-PI, PI)
                    rr = z["next_radius"] * 0.7 * math.sqrt(self.rng.random())
                    m["wander"] = (ncx + rr * math.cos(a), ncy + rr * math.sin(a))
                goal = m["wander"]
        gx, gy = goal
        dx, dy = gx - m["pos"][0], gy - m["pos"][1]
        n = math.hypot(dx, dy)
        if n > 2:
            best = None
            for rot in (0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
                a = math.atan2(dy, dx) + rot
                if self._free(m["pos"][0] + 6 * math.cos(a), m["pos"][1] + 6 * math.sin(a)):
                    best = a
                    break
            if best is None:
                best = math.atan2(dy, dx) + self.rng.uniform(-PI, PI)
            m["cmd"] = [math.cos(best), math.sin(best)]
        else:
            m["cmd"] = [0, 0]
        m["boost"] = False
        if known and me["alive"] and (cansee or (d < 200 and self.los(m, me))):
            ft = d / 252.0
            lead = 0.0 if (m["laser"] and m["lcd"] <= 0) else ft
            px, py = me["pos"][0] + me["vel"][0] * lead, me["pos"][1] + me["vel"][1] * lead
            err = math.radians({"easy": 1.6, "medium": 1.1, "hard": 0.7, "expert": 0.45}[m["tier"]])
            m["aim_cmd"] = [math.atan2(py - m["pos"][1], px - m["pos"][0]) + self.rng.gauss(0, err),
                            math.atan2(me["pos"][2] + 4.5 - m["pos"][2] - 7, max(1.0, d))]
            provoked = self.t - m.get("hit_by_us_t", -99) < 10
            near = d < (50 if PASSIVE else 110)
            m["fire"] = cansee and (m["hunter"] or near or provoked) and not m.get("retreat")
            if PASSIVE and not (m["hunter"] or near or provoked):
                # a passive member looks where it walks, not at us
                m["aim_cmd"] = [math.atan2(m["cmd"][1], m["cmd"][0]) if (m["cmd"][0] or m["cmd"][1]) else m["aim"][0], 0]
            if self.tough and m["fire"]:
                if self.t > m.get("strafe_t", 0):
                    m["strafe"] = self.rng.choice((-1.0, 1.0))
                    m["strafe_t"] = self.t + self.rng.uniform(0.4, 1.0)
                a = math.atan2(me["pos"][1] - m["pos"][1], me["pos"][0] - m["pos"][0]) + PI / 2
                cx, cy = m["cmd"][0] * 0.5 + m["strafe"] * math.cos(a), m["cmd"][1] * 0.5 + m["strafe"] * math.sin(a)
                n = math.hypot(cx, cy) or 1.0
                m["cmd"] = [cx / n, cy / n]
        else:
            m["aim_cmd"] = [math.atan2(m["cmd"][1], m["cmd"][0]) if (m["cmd"][0] or m["cmd"][1]) else m["aim"][0], 0]
            m["fire"] = False

    def squad_fire(self, m, dt):
        me = self.me
        m["cd"] -= dt
        m["lcd"] -= dt
        if m["reload"] > 0:
            m["reload"] -= dt
            if m["reload"] <= 0:
                m["ammo"] = 12
        if not m["fire"] or not me["alive"]:
            m["charge"] = 0.0
            return
        d = math.hypot(me["pos"][0] - m["pos"][0], me["pos"][1] - m["pos"][1])
        if self.phys:
            if m["aim_cmd"] is None or abs(wrap(m["aim_cmd"][0] - m["aim"][0])) > 0.03:
                return
            if m["laser"] and m["lcd"] <= 0 and d < 280:
                m["charge"] += dt
                if m["charge"] >= 0.75:
                    if self.laser_hits(m, [me]) is me:
                        self.damage(me, m, 315, "laser")
                    m["charge"] = 0.0
                    m["lcd"] = 1.65
                return
            if m["cd"] <= 0 and m["ammo"] > 0 and d < 230 and m["reload"] <= 0:
                m["cd"] = 0.24
                m["ammo"] -= 1
                if m["ammo"] == 0:
                    m["reload"] = 2.4
                self.spawn_bullet(m, m["aim"][0], m["aim"][1], 45)
            return
        on = abs(wrap(math.atan2(me["pos"][1] - m["pos"][1], me["pos"][0] - m["pos"][0]) - m["aim"][0])) < 0.05
        if not on:
            return
        lateral = abs(me["vel"][0] * math.sin(m["aim"][0]) - me["vel"][1] * math.cos(m["aim"][0]))
        if m["laser"] and m["lcd"] <= 0 and d < 280:
            m["charge"] += dt
            if m["charge"] >= 0.75:
                p = 0.85 - 0.02 * lateral
                if self.rng.random() < p and self.los(m, me, 7, 4.5):
                    self.damage(me, m, 315, "laser")
                m["charge"] = 0.0
                m["lcd"] = 1.65
            return
        if m["cd"] <= 0 and m["ammo"] > 0 and d < 230 and m["reload"] <= 0:
            m["cd"] = 0.24
            m["ammo"] -= 1
            if m["ammo"] == 0:
                m["reload"] = 2.4
            p = m["acc"] * max(0.0, 1.0 - d / 260) * (1.0 - min(0.5, lateral / 40))
            if self.rng.random() < p and self.los(m, me, 7, 4.5):
                self.damage(me, m, 45, "gun")

    # ---- physical bullets ----------------------------------------------------------------
    def spawn_bullet(self, owner, yaw, pitch, dmg, spread_deg=0.6, speed=252.0, rng=230.0):
        yaw += self.rng.gauss(0, math.radians(spread_deg))
        pitch += self.rng.gauss(0, math.radians(spread_deg))
        cp = math.cos(pitch)
        dx, dy, dz = cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)
        p = owner["pos"]
        self.bullets.append(dict(owner=owner, pos=[p[0] + 3 * dx, p[1] + 3 * dy, p[2] + 7 + 3 * dz],
                                 vel=[speed * dx, speed * dy, speed * dz], ttl=rng / speed, dmg=dmg))

    def bullets_step(self, dt):
        keep = []
        tg = None
        for b in self.bullets:
            ox, oy, oz = b["pos"]
            vx, vy, vz = b["vel"]
            nx, ny, nz = ox + vx * dt, oy + vy * dt, oz + vz * dt
            b["ttl"] -= dt
            owner = b["owner"]
            if owner is self.me:
                if tg is None:
                    tg = self.targets()
                cands = tg
            else:
                cands = [self.me] if self.me["alive"] else []
            hit = None
            best_s = 2.0
            sx, sy = nx - ox, ny - oy
            L2 = sx * sx + sy * sy or 1e-9
            for m in cands:
                mx, my, mz = m["pos"]
                u = ((mx - ox) * sx + (my - oy) * sy) / L2
                u = max(0.0, min(1.0, u))
                cx, cy = ox + sx * u, oy + sy * u
                r = m.get("radius", 2.5) * (1 + 0.05 * m.get("cubes", 0))
                h = 9 if m.get("kind") == "mech" else (3.5 if m.get("kind") == "npc" else 1.6)
                zc = oz + (nz - oz) * u
                if (cx - mx) ** 2 + (cy - my) ** 2 < r * r and mz - 0.3 < zc < mz + h + 0.3 and u < best_s:
                    best_s, hit = u, m
            blocked = not self.M.los(ox, oy, oz, nx, ny, nz, 2.0)
            if hit is not None and blocked:
                hx, hy = ox + sx * best_s, oy + sy * best_s
                if not self.M.los(ox, oy, oz, hx, hy, oz + (nz - oz) * best_s, 2.0):
                    hit = None
            if hit is not None:
                if owner is self.me:
                    self.hit_target(hit, b["dmg"], "gun")
                else:
                    self.damage(self.me, owner, b["dmg"], "gun")
                continue
            if blocked or b["ttl"] <= 0:
                continue
            b["pos"] = [nx, ny, nz]
            keep.append(b)
        self.bullets = keep

    def laser_hits(self, shooter, victims):
        """Hitscan along shooter's actual aim: first victim whose cylinder the ray passes, with LOS."""
        yaw, pitch = shooter["aim"]
        sx, sy, sz = shooter["pos"][0], shooter["pos"][1], shooter["pos"][2] + 7
        dx, dy = math.cos(yaw), math.sin(yaw)
        best = None
        for m in victims:
            ex, ey = m["pos"][0] - sx, m["pos"][1] - sy
            along = ex * dx + ey * dy
            if along <= 0 or along > 280:
                continue
            perp = abs(-ex * dy + ey * dx)
            r = m.get("radius", 2.5) * (1 + 0.05 * m.get("cubes", 0))
            h = 9 if m.get("kind") == "mech" else (3.5 if m.get("kind") == "npc" else 1.6)
            zc = sz + math.tan(pitch) * along
            if perp < r and m["pos"][2] - 0.5 < zc < m["pos"][2] + h + 0.5:
                if (best is None or along < best[0]) and self.M.los(sx, sy, sz, sx + dx * along, sy + dy * along, zc):
                    best = (along, m)
        return best[1] if best else None

    # ---- NPCs ------------------------------------------------------------------------------
    def npc_step(self, n, dt):
        me = self.me
        n["cd"] -= dt
        hx, hy = n["home"]
        dh = math.hypot(n["pos"][0] - hx, n["pos"][1] - hy)
        tgt = None
        if n["target"] == 0 and me["alive"]:
            d = math.hypot(me["pos"][0] - n["pos"][0], me["pos"][1] - n["pos"][1])
            if d > 90 or dh > 60:
                n["target"] = None
            else:
                tgt = me
        if tgt is None and me["alive"] and dh < 30:
            d = math.hypot(me["pos"][0] - n["pos"][0], me["pos"][1] - n["pos"][1])
            if d < 45 and self.los(n, me, 3, 4.5):
                tgt = me
                n["target"] = 0
        if tgt is not None:
            d = math.hypot(tgt["pos"][0] - n["pos"][0], tgt["pos"][1] - n["pos"][1])
            gx, gy = tgt["pos"][0], tgt["pos"][1]
            if d < 30:
                gx, gy = n["pos"][0], n["pos"][1]
            if n["cd"] <= 0 and self.los(n, tgt, 3, 4.5):
                n["cd"] = 1.0
                p = max(0.1, 0.8 - d / 120)
                if self.rng.random() < p:
                    self.damage(tgt, n, 25, "npc")
        else:
            gx, gy = hx, hy
        dx, dy = gx - n["pos"][0], gy - n["pos"][1]
        dd = math.hypot(dx, dy)
        if dd > 2:
            nx, ny = n["pos"][0] + dx / dd * 5 * dt, n["pos"][1] + dy / dd * 5 * dt
            if self._free(nx, ny):
                n["pos"] = [nx, ny, self.M.ground(nx, ny)]
                n["vel"] = [dx / dd * 5, dy / dd * 5, 0]
        else:
            n["vel"] = [0, 0, 0]

    # ---- our bot -------------------------------------------------------------------------
    def visible_items(self, items, zb):
        me = self.me
        out = []
        for it in items:
            p = it["pos"]
            if abs(p[0] - me["pos"][0]) > 200 or abs(p[1] - me["pos"][1]) > 200:
                continue
            if self.sees_pos(me, p, zb):
                out.append(it)
        return out

    def obs(self):
        me = self.me
        z = self.zone()
        mechs = []
        for m in self.mechs[1:]:
            if not m["alive"]:
                continue
            if self.sees(me, m):
                self.seen_t[m["id"]] = self.t
            if self.t - self.seen_t.get(m["id"], -99) <= 5:
                w = ["gun", "laser"] if m["laser"] else ["gun"]
                mechs.append({"id": m["id"], "kind": "mech", "pos": list(m["pos"]), "vel": list(m["vel"]), "aim": list(m["aim"]),
                              "hp": m["hp"], "max_hp": m["max_hp"], "cubes": m["cubes"], "weapons": w, "charging": m["charge"] > 0})
        for n in self.npcs:
            if n["alive"] and self.sees_pos(me, n["pos"], 1.75):
                mechs.append({"id": n["id"], "kind": "npc", "pos": list(n["pos"]), "vel": list(n["vel"]), "aim": [0, 0],
                              "hp": n["hp"], "max_hp": 350})
        in_zone = math.hypot(me["pos"][0] - z["center"][0], me["pos"][1] - z["center"][1]) <= z["radius"]
        g = dict(self.cfg["weapons"]["gun"])
        g["damage"] = 45 * me.get("gun_dmg", 1.0)
        g["cooldown"] = me.get("gun_cd", 0.24)
        g["mag"] = me.get("mag", 12)
        w1 = None
        if me["w1"]:
            ws = dict(self.cfg["weapons"][me["w1"]])
            if me["w1"] == "laser":
                ws["charge"] = me.get("laser_full", 0.75)
            w1 = {"kind": me["w1"], "cooldown": max(0.0, me["w1cd"], me["equip"]), "stats": ws}
            if me["w1"] == "laser":
                w1["charge"] = me["charge"]
        stats = dict(self.cfg["mech"])
        stats["walk_speed"] = 11.7 * me["speed_mult"]
        stats["boost_speed"] = 19.5 * me["speed_mult"]
        o = {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": sum(m["alive"] for m in self.mechs),
             "self": {"id": 0, "pos": list(me["pos"]), "vel": list(me["vel"]), "aim": list(me["aim"]), "hp": me["hp"], "max_hp": me["max_hp"],
                      "energy": me["energy"], "max_energy": 100, "grounded": me.get("zoff", 0.0) <= 0.01, "cubes": me["cubes"],
                      "upgrades": [k for k, v in me["upg"].items() for _ in range(v)], "in_zone": in_zone,
                      "weapons": [{"kind": "gun", "cooldown": max(0.0, me["cd"], me.get("gequip", 0)), "ammo": me["ammo"],
                                   "reload": max(0.0, me["reload"]), "stats": g}, w1],
                      "stats": stats},
             "mechs": mechs, "projectiles": [], "beams": [], "sounds": [],
             "crates": [{"id": c["id"], "pos": list(c["pos"]), "hp": c["hp"]} for c in self.visible_items(self.crates, 0.8)],
             "cubes": [{"id": c["id"], "pos": list(c["pos"])} for c in self.visible_items(self.cubes, 0.3)],
             "pickups": [{"id": p["id"], "kind": p["kind"], "pos": list(p["pos"])} for p in self.visible_items(self.pickups, 0.3)],
             "events": self.events,
             "drops": [{"id": d["id"], "pos": list(d["pos"]), "radius": 8, "lands_in": max(0.0, d["lands_in"]), "capture_s": 4,
                        "progress": d["progress"], "holder": d["holder"] if d["holder"] is not None and d["holder"] >= 0 else None}
                       for d in self.drops if not d["taken"]],
             "zone": z, "offer": self.offer}
        self.events = []
        return o

    def bot_act(self):
        o = self.obs()
        if getattr(self, "trace", False) and self.tick % 10 == 0:
            me = self.me
            vis = [(m["id"], round(math.hypot(m["pos"][0] - me["pos"][0], m["pos"][1] - me["pos"][1])), "H" if m["id"] < 16 and self.mechs[m["id"]]["hunter"] else "") for m in o["mechs"]]
            dm = [(e["by"], round(e["amount"])) for e in o["events"] if e["type"] == "damaged"]
            print(f"t={self.t:5.1f} hp={me['hp']:6.0f}/{me['max_hp']:.0f} c={me['cubes']} w1={me['w1']} pos=({me['pos'][0]:.0f},{me['pos'][1]:.0f}) goal={getattr(self.bot, 'goal_kind', '')} vis={vis} dmg={dm}")
        t0 = time.perf_counter()
        a = self.bot.act(o)
        ms = (time.perf_counter() - t0) * 1000
        self.max_ms = max(self.max_ms, ms)
        self.sum_ms += ms
        self.n_dec += 1
        json.dumps(a, allow_nan=False)
        me = self.me
        mv = a.get("move", [0, 0])
        me["cmd"] = mv
        me["boost"] = a.get("boost", False)
        me["jump_cmd"] = bool(a.get("jump", False))
        if "aim" in a:
            me["aim_cmd"] = a["aim"]
        fr = a.get("fire", [False, False])
        me["fire"] = bool(fr[0])
        me["fire1"] = bool(fr[1]) if len(fr) > 1 else False
        if a.get("reload") and me["reload"] <= 0 and me["ammo"] < 12:
            me["reload"] = 2.4
        if (mv[0] or mv[1]) and math.hypot(*me["vel"][:2]) < 0.5:
            self.stuck_ticks += 1
        if "pick" in a and self.offer:
            ch = self.offer["choices"]
            i = int(a["pick"])
            if 0 <= i < len(ch):
                self.apply_upgrade(ch[i]["id"])
            self.offer = None
        if "take" in a:
            self.do_take(a["take"], a.get("drop"))

    def apply_upgrade(self, uid):
        me = self.me
        me["upg"][uid] = me["upg"].get(uid, 0) + 1
        if uid == "plating":
            me["hp_mult"] *= 1.15
            self.refresh_hp()
        elif uid == "armor":
            me["armor"] += 0.08
        elif uid == "servos":
            me["speed_mult"] *= 1.08
        elif uid == "gun_damage":
            me["gun_dmg"] = me.get("gun_dmg", 1.0) * 1.15
        elif uid == "gun_rate":
            me["gun_cd"] = me.get("gun_cd", 0.24) * 0.87
        elif uid == "gun_mag":
            me["mag"] = me.get("mag", 12) + 8
        elif uid == "laser_damage":
            me["laser_dmg"] = me.get("laser_dmg", 1.0) * 1.18
        elif uid == "laser_charge":
            me["laser_full"] = me.get("laser_full", 0.75) * 0.75
        elif uid == "nanites":
            me["regen"] *= 1.6
            me["regen_delay"] -= 1.5
        self.events.append({"type": "upgrade", "id": uid})

    def refresh_hp(self):
        me = self.me
        new = 1400 * (1 + 0.04 * min(me["cubes"], 10)) * me["hp_mult"]
        me["hp"] += new - me["max_hp"]
        me["max_hp"] = new

    def gain_cubes(self, n):
        me = self.me
        th = self.cfg["cubes"]["upgrade_thresholds"]
        for _ in range(n):
            me["cubes"] += 1
            self.refresh_hp()
            if me["cubes"] in th:
                self.offers_pending += 1
        self.events.append({"type": "pickup", "cubes": me["cubes"]})

    def do_take(self, pid, drop):
        me = self.me
        p = next((p for p in self.pickups if p["id"] == pid), None)
        if p is None:
            self.events.append({"type": "take_failed", "id": pid, "reason": "gone"})
            return
        if math.hypot(p["pos"][0] - me["pos"][0], p["pos"][1] - me["pos"][1]) > 4:
            self.events.append({"type": "take_failed", "id": pid, "reason": "range"})
            return
        if p["kind"] == me["w1"]:
            self.events.append({"type": "take_failed", "id": pid, "reason": "have"})
            return
        if me["w1"] is not None:
            if drop != 1:
                self.events.append({"type": "take_failed", "id": pid, "reason": "full"})
                return
            self.pickups.append(dict(id=self.next_id, kind=me["w1"], pos=list(me["pos"])))
            self.events.append({"type": "dropped", "id": self.next_id, "kind": me["w1"], "slot": 1})
            self.next_id += 1
        self.pickups.remove(p)
        me["w1"] = p["kind"]
        me["equip"] = max(1.0, self.cfg["weapons"][p["kind"]].get("cooldown", 0))
        me["charge"] = 0.0
        self.events.append({"type": "took", "id": pid, "kind": p["kind"], "slot": 1})

    def targets(self):
        out = [m for m in self.mechs[1:] if m["alive"]]
        out += [n for n in self.npcs if n["alive"]]
        out += [dict(c, kind="crate", alive=True, _ref=c) for c in self.crates if c["hp"] > 0]
        return out

    def ray_hit(self, yaw, pitch, rng, spread_deg=0.0, lead_noise=True, jitter_s=0.0):
        me = self.me
        yaw += self.rng.gauss(0, math.radians(spread_deg)) if spread_deg else 0
        best = None
        mz = me["pos"][2] + 7
        for m in self.targets():
            dx, dy = m["pos"][0] - me["pos"][0], m["pos"][1] - me["pos"][1]
            d = math.hypot(dx, dy)
            if d > rng:
                continue
            ft = d / 252 if lead_noise else 0
            f = self.rng.uniform(0.3, 1.0) if lead_noise else 0
            v = m.get("vel", [0, 0, 0])
            px, py = m["pos"][0] + v[0] * ft * f, m["pos"][1] + v[1] * ft * f
            if jitter_s and m.get("kind") == "mech":
                # it saw the charge coming / changed direction: lateral error up to jitter_s of travel
                a = self.rng.uniform(-PI, PI)
                j = self.rng.uniform(0, jitter_s) * 11.7
                px, py = px + j * math.cos(a), py + j * math.sin(a)
            ex, ey = px - me["pos"][0], py - me["pos"][1]
            along = ex * math.cos(yaw) + ey * math.sin(yaw)
            perp = abs(-ex * math.sin(yaw) + ey * math.cos(yaw))
            r = m.get("radius", 2.5) * (1 + 0.05 * m.get("cubes", 0))
            h = 9 if m.get("kind") == "mech" else (3.5 if m.get("kind") == "npc" else 1.6)
            zc = mz + math.tan(pitch) * along
            if along > 0 and perp < r and m["pos"][2] - 0.5 < zc < m["pos"][2] + h + 0.5:
                if (best is None or along < best[0]) and self.M.los(me["pos"][0], me["pos"][1], mz, px, py, m["pos"][2] + min(h, 4.5)):
                    best = (along, m)
        return best[1] if best else None

    def hit_target(self, m, amount, weapon):
        if m.get("kind") == "crate":
            ref = m["_ref"]
            ref["hp"] -= amount
            self.events.append({"type": "hit", "target": ref["id"], "amount": amount, "weapon": weapon})
            if ref["hp"] <= 0:
                self.crates.remove(ref)
                self.cubes.append(dict(id=self.next_cube, pos=list(ref["pos"])))
                self.next_cube += 1
                if self.rng.random() < 0.5:
                    self.pickups.append(dict(id=self.next_id, kind=self.rng.choice(["mortar", "laser", "lightning"]), pos=list(ref["pos"])))
                    self.next_id += 1
            return
        self.damage(m, self.me, amount, weapon)

    def our_fire(self, dt):
        me = self.me
        dm = self.dmg_mult(me)
        me["cd"] -= dt
        me["w1cd"] -= dt
        me["equip"] -= dt
        if me["reload"] > 0:
            me["reload"] -= dt
            if me["reload"] <= 0:
                me["ammo"] = me.get("mag", 12)
        elif me["fire"] and me["cd"] <= 0 and me["ammo"] > 0:
            me["cd"] = me.get("gun_cd", 0.24)
            me["ammo"] -= 1
            if me["ammo"] == 0:
                me["reload"] = 2.4
            if self.phys:
                self.spawn_bullet(me, me["aim"][0], me["aim"][1], 45 * dm * me.get("gun_dmg", 1.0))
            else:
                m = self.ray_hit(me["aim"][0], me["aim"][1], 230, 0.6)
                if m is not None:
                    self.hit_target(m, 45 * dm, "gun")
            if self.intel == "vision" and any(o["alive"] and math.hypot(o["pos"][0] - me["pos"][0], o["pos"][1] - me["pos"][1]) < 320 for o in self.mechs[1:]):
                self.last_seen_pos = (me["pos"][0] + self.rng.uniform(-20, 20), me["pos"][1] + self.rng.uniform(-20, 20))
                self.last_seen_t = self.t
        w = me["w1"]
        if w is None or me["equip"] > 0:
            me["charge"] = 0.0
            return
        if w == "laser":
            full = me.get("laser_full", 0.75)
            if me["fire1"] and me["w1cd"] <= 0:
                me["charge"] = min(full, me["charge"] + dt)
                me["hold"] = me.get("hold", 0) + (dt if me["charge"] >= full else 0)
                if me["hold"] < 2.0:
                    return
            if me["charge"] > 0 and me["w1cd"] <= 0:
                if me["charge"] < 0.15:
                    me["charge"] += dt
                    return
                frac = 0.35 + 0.65 * me["charge"] / full
                if self.phys:
                    m = self.laser_hits(me, self.targets())
                else:
                    m = self.ray_hit(me["aim"][0], me["aim"][1], 280, 0.0, lead_noise=False, jitter_s=0.25 if self.tough else 0.0)
                if m is not None:
                    self.hit_target(m, 315 * frac * dm * me.get("laser_dmg", 1.0), "laser")
                me["charge"] = 0.0
                me["hold"] = 0.0
                me["w1cd"] = 1.65
        elif w == "lightning":
            if me["fire1"] and me["w1cd"] <= 0 and me["energy"] >= 3:
                me["w1cd"] = 0.25
                me["energy"] -= 3
                me["e_spent"] = self.t
                best = None
                for m in self.targets():
                    dx, dy = m["pos"][0] - me["pos"][0], m["pos"][1] - me["pos"][1]
                    d = math.hypot(dx, dy)
                    if d < 45 and abs(wrap(math.atan2(dy, dx) - me["aim"][0])) < math.radians(15) and (best is None or d < best[0]):
                        if self.M.los(me["pos"][0], me["pos"][1], me["pos"][2] + 7, m["pos"][0], m["pos"][1], m["pos"][2] + 1.5):
                            best = (d, m)
                if best:
                    self.hit_target(best[1], 60 * dm, "lightning")
        elif w == "mortar":
            if me["fire1"] and me["w1cd"] <= 0:
                me["w1cd"] = 2.6
                y, p = me["aim"]
                v = 72
                self.shells.append(dict(pos=[me["pos"][0] + 3 * math.cos(y), me["pos"][1] + 3 * math.sin(y), me["pos"][2] + 9.5],
                                        vel=[v * math.cos(p) * math.cos(y), v * math.cos(p) * math.sin(y), v * math.sin(p)], t=0))

    def shells_step(self, dt):
        for s in list(self.shells):
            s["vel"][2] -= 22 * dt
            s["t"] += dt
            ox, oy, oz = s["pos"]
            s["pos"] = [ox + s["vel"][0] * dt, oy + s["vel"][1] * dt, oz + s["vel"][2] * dt]
            x, y, z = s["pos"]
            k = self.M.fidx(x, y)
            hit = k < 0 or z < self.M.ground(x, y) or (self.M.fbase[k] < z < self.M.ftop[k]) or s["t"] > 7
            for m in self.targets():
                if m.get("kind") == "crate":
                    continue
                if math.hypot(m["pos"][0] - x, m["pos"][1] - y) < m.get("radius", 2.5) and m["pos"][2] < z < m["pos"][2] + 9:
                    hit = True
            if hit:
                self.shells.remove(s)
                dm = self.dmg_mult(self.me)
                for m in self.targets() + [self.me]:
                    d = math.hypot(m["pos"][0] - x, m["pos"][1] - y)
                    if d < 7:
                        amt = 160 * (1 - 0.7 * d / 7) * dm
                        if d < m.get("radius", 2.5):
                            amt = 200 * dm
                        if m is self.me:
                            self.damage(self.me, self.me, amt * 0.5, "mortar")
                        else:
                            self.hit_target(m, amt, "mortar")

    # ---- drops ---------------------------------------------------------------------------
    def drops_step(self, dt):
        times = self.cfg.get("drops", {}).get("times", [])
        for k, tt in enumerate(times):
            if self.t >= tt and not any(d["id"] == 20000 + k for d in self.drops):
                z = self.zone()
                for _ in range(200):
                    a = self.rng.uniform(-PI, PI)
                    rr = z["next_radius"] * 0.7 * math.sqrt(self.rng.random())
                    x, y = z["next_center"][0] + rr * math.cos(a), z["next_center"][1] + rr * math.sin(a)
                    if self.M.walkable(x, y):
                        break
                self.drops.append(dict(id=20000 + k, pos=[x, y, self.M.ground(x, y)], lands_in=25.0, progress=0.0, holder=None, taken=False))
        for d in self.drops:
            if d["taken"]:
                continue
            d["lands_in"] -= dt
            if d["lands_in"] > 0:
                continue
            inside = [m for m in self.mechs if m["alive"] and math.hypot(m["pos"][0] - d["pos"][0], m["pos"][1] - d["pos"][1]) < 8]
            if len(inside) == 1:
                h = inside[0]["id"]
                if d["holder"] != h:
                    d["holder"], d["progress"] = h, 0.0
                d["progress"] += dt
                if d["progress"] >= 4:
                    d["taken"] = True
                    if h == 0:
                        me = self.me
                        self.gain_cubes(3)
                        me["hp"] = me["max_hp"]
                        me["energy"] = 100
                        self.drops_won += 1
                        self.events.append({"type": "drop", "id": d["id"], "cubes": me["cubes"]})
                    else:
                        m = self.mechs[h]
                        m["hp"] = m["max_hp"] = m["max_hp"] * 1.12
                        m["cubes"] += 3
            elif not inside:
                d["holder"], d["progress"] = None, 0.0

    # ---- main loop -----------------------------------------------------------------------
    def run(self):
        dt = 0.05
        hunt_plan = {"normal": [(20, 1), (150, 4), (270, 15)], "fast": [(10, 2), (90, 6), (180, 15)],
                     "slow": [(40, 1), (200, 3), (320, 15)], "squad": []}[self.hunt]
        order = list(range(1, 16))
        self.rng.shuffle(order)
        if self.hunt == "squad":
            # official rule: weakest first; one early, more as we thin them out, all of them late
            rank = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
            order.sort(key=lambda i: rank[self.mechs[i]["tier"]])
        me = self.me
        while self.t < 480:
            if self.hunt == "squad":
                if self.t >= 15:
                    alive = [i for i in order if self.mechs[i]["alive"]]
                    n = len(alive) if self.t >= 300 else 1 + self.kills
                    for i in alive[:n]:
                        self.mechs[i]["hunter"] = True
            for (ht, n) in hunt_plan:
                if self.t >= ht:
                    for i in order[:n]:
                        self.mechs[i]["hunter"] = True
            if self.offer is None and self.offers_pending > 0:
                self.offers_pending -= 1
                ch = self.rng.sample(UPG, 3)
                self.offer = {"id": self.tick, "expires": self.t + 5, "choices": [{"id": c, "name": c, "desc": c} for c in ch]}
            if self.tick % 2 == 0:
                if me["alive"]:
                    self.bot_act()
                for m in self.mechs[1:]:
                    if m["alive"]:
                        self.squad_think(m)
            for m in self.mechs:
                if not m["alive"]:
                    continue
                if m["aim_cmd"] is not None:
                    da = wrap(m["aim_cmd"][0] - m["aim"][0])
                    m["aim"][0] = wrap(m["aim"][0] + max(-4 * dt, min(4 * dt, da)))
                    dp = m["aim_cmd"][1] - m["aim"][1]
                    m["aim"][1] += max(-3 * dt, min(3 * dt, dp))
                self.move(m, dt)
            for n in self.npcs:
                if n["alive"]:
                    self.npc_step(n, dt)
            if me["alive"]:
                self.our_fire(dt)
            self.shells_step(dt)
            if self.phys:
                self.bullets_step(dt)
            for m in self.mechs[1:]:
                if m["alive"]:
                    self.squad_fire(m, dt)
            self.drops_step(dt)
            # cubes
            for c in list(self.cubes):
                if me["alive"] and math.hypot(c["pos"][0] - me["pos"][0], c["pos"][1] - me["pos"][1]) < 3.5:
                    self.cubes.remove(c)
                    self.gain_cubes(1)
            z = self.zone()
            for m in self.mechs:
                if not m["alive"]:
                    continue
                out = math.hypot(m["pos"][0] - z["center"][0], m["pos"][1] - z["center"][1]) > z["radius"]
                if out:
                    m["hp"] -= z["dps"] * dt
                    m["last_combat"] = max(m["last_combat"], self.t - 0.01)
                    if m is me:
                        self.zone_dmg += z["dps"] * dt
                        self.events.append({"type": "damaged", "amount": z["dps"] * dt, "weapon": "zone", "bearing": None, "by": None})
                elif self.t - m["last_combat"] > m["regen_delay"]:
                    m["hp"] = min(m["max_hp"], m["hp"] + m["regen"] * dt)
            for n in self.npcs:
                if n["alive"] and n["hp"] <= 0:
                    n["alive"] = False
                    for _ in range(2):
                        self.cubes.append(dict(id=self.next_cube, pos=[n["pos"][0] + self.rng.uniform(-1, 1), n["pos"][1] + self.rng.uniform(-1, 1), n["pos"][2]]))
                        self.next_cube += 1
            dead_now = [m for m in self.mechs if m["alive"] and m["hp"] <= 0]
            if dead_now:
                alive = sum(m["alive"] for m in self.mechs)
                for m in dead_now:
                    m["alive"] = False
                    m["death_t"] = self.t
                    m["place"] = alive
                    if m is not me:
                        self.kills += 1
                        for _ in range(m["cubes"] + 2):
                            self.cubes.append(dict(id=self.next_cube, pos=[m["pos"][0] + self.rng.uniform(-2, 2), m["pos"][1] + self.rng.uniform(-2, 2), m["pos"][2]]))
                            self.next_cube += 1
                        if m["laser"]:
                            self.pickups.append(dict(id=self.next_id, kind="laser", pos=list(m["pos"])))
                            self.next_id += 1
                        self.events.append({"type": "kill", "target": m["id"]})
                self.deaths.append((self.t, [m["id"] for m in dead_now]))
            if not me["alive"] or sum(m["alive"] for m in self.mechs) <= 1:
                break
            self.t += dt
            self.tick += 1
        if me["alive"]:
            alive = sorted([m for m in self.mechs if m["alive"]], key=lambda m: -m["hp"])
            me["place"] = 1 + alive.index(me)
        return me["place"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]
    n = int(args[0]) if args else 10
    seed0 = int(args[1]) if len(args) > 1 else 1
    mod = flags[flags.index("--bot") + 1] if "--bot" in flags else "bot"
    hunt = flags[flags.index("--hunt") + 1] if "--hunt" in flags else "normal"
    intel = flags[flags.index("--intel") + 1] if "--intel" in flags else "omni"
    tough = "--tough" in flags
    if "--intel" in flags:
        args = [a for a in args if a != intel]
    if "--bot" in flags:
        args = [a for a in args if a != mod]
    if "--hunt" in flags:
        args = [a for a in args if a != hunt]
    bot_mod = importlib.import_module(mod)
    hello = load_hello()
    M = nav.Map(hello["map"]["dir"])
    scores = []
    quiet = "--quiet" in flags
    for k in range(n):
        if not hasattr(M, "_walk0"):
            M._walk0 = bytes(M.walk)
        M.walk = bytearray(M._walk0)
        sim = Sim(M, hello, bot_mod.Bot, seed0 + k, hunt, intel, tough)
        sim.trace = "--trace" in flags
        place = sim.run()
        scores.append(16 - place)
        if not quiet:
            print("seed %d place %2d died@%3s zone %4d kills %d cubes %2d w1 %-9s drops %d stuck %3d max_ms %5.1f avg_ms %.2f" % (
                seed0 + k, place, round(sim.me["death_t"] or 480), round(sim.zone_dmg), sim.kills, sim.me["cubes"], sim.me["w1"],
                sim.drops_won, sim.stuck_ticks, sim.max_ms, sim.sum_ms / max(1, sim.n_dec)), flush=True)
    print(f"mean score {sum(scores) / len(scores):.2f} over {n}")


if __name__ == "__main__":
    main()
