"""Hit-rate stats: python3 tools/hits.py module n seed [--tough] [--intel X]"""
import sys, collections
sys.path.insert(0,'.'); sys.path.insert(0,'tools'); sys.path.insert(0,'tools/archive')
mod, n, seed0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import importlib, sim, nav
B = importlib.import_module(mod)
h=sim.load_hello(); M=nav.Map(h['map']['dir']); M._walk0=bytes(M.walk)
st=collections.Counter()
for k in range(n):
    M.walk=bytearray(M._walk0)
    s=sim.Sim(M,h,B.Bot,seed0+k,'normal','omni',True)
    osb=s.spawn_bullet
    def sb(owner,yaw,pitch,dmg,*a,s=s,osb=osb,**kw):
        st['our_shots' if owner is s.me else 'their_shots']+=1
        return osb(owner,yaw,pitch,dmg,*a,**kw)
    s.spawn_bullet=sb
    oh=s.hit_target
    def ht(m,amount,weapon,s=s,oh=oh):
        st[f'our_hit_{weapon}_{m.get("kind")}']+=1; st[f'our_dmg_{m.get("kind")}']+=amount
        return oh(m,amount,weapon)
    s.hit_target=ht
    od=s.damage
    def dm(victim,attacker,amount,weapon,s=s,od=od):
        if victim is s.me and attacker is not s.me: st[f'their_hit_{weapon}']+=1; st['their_dmg']+=amount
        if attacker is s.me and victim.get('kind')=='mech': st['our_mech_dmg_total']+=amount
        return od(victim,attacker,amount,weapon)
    s.damage=dm
    olh=s.laser_hits
    def lh(shooter,victims,s=s,olh=olh):
        r=olh(shooter,victims); st['our_laser_fired' if shooter is s.me else 'their_laser_fired']+=1
        if r is not None: st['our_laser_hit' if shooter is s.me else 'their_laser_hit']+=1
        return r
    s.laser_hits=lh
    p=s.run(); st['score']+=16-p; st['kills']+=s.kills; st['lifetime']+= (s.me['death_t'] or 480)
for k,v in sorted(st.items()): print(k, round(v/n,1))
