import sys, collections
mod = sys.argv[1]; flags = sys.argv[2:]
sys.path.insert(0,'.'); sys.path.insert(0,'tools'); sys.path.insert(0,'tools/archive'); sys.argv=['x']+flags
import importlib, sim, nav
B = importlib.import_module(mod)
h=sim.load_hello(); M=nav.Map(h['map']['dir']); M._walk0=bytes(M.walk)
src=collections.Counter(); n=0; modes=collections.Counter(); tiers=collections.Counter(); deathT=[]
for seed in range(400,440):
    M.walk=bytearray(M._walk0)
    s=sim.Sim(M,h,B.Bot,seed,'squad','omni',True)
    log=[]
    od=s.damage
    def dm(victim,attacker,amount,weapon,s=s,log=log,od=od):
        if victim is s.me and attacker is not s.me:
            prov = s.t - attacker.get('hit_by_us_t',-99) < 10
            who = 'H' if attacker.get('hunter') else ('npc' if attacker.get('kind')=='npc' else ('provoked' if prov else 'near'))
            log.append((who, amount, s.bot.goal_kind, attacker.get('tier'), weapon))
        return od(victim,attacker,amount,weapon)
    s.damage=dm
    s.run()
    dt = s.me['death_t']; deathT.append(round(dt) if dt else 480)
    if dt is not None and dt<90:
        n+=1
        for who,a,mode,tier,w in log: src[who]+=a; modes[mode]+=a; tiers[(who,tier,w)]+=a
print('early',n,'/40', sorted(src.items(),key=lambda kv:-kv[1]))
print(sorted(modes.items(),key=lambda kv:-kv[1]))
print(sorted(tiers.items(),key=lambda kv:-kv[1])[:8])
print('death times', sorted(deathT))
