import sys, collections
sys.path.insert(0,'.'); sys.path.insert(0,'tools'); sys.argv=['x']
import sim, nav, bot
h=sim.load_hello(); M=nav.Map(h['map']['dir']); M._walk0=bytes(M.walk)
hunt=sys.argv[1] if len(sys.argv)>1 else 'normal'
buckets=collections.Counter(); src=collections.Counter(); modes=collections.Counter(); places=[]
for seed in range(int(sys.argv[2]) if len(sys.argv)>2 else 9000, (int(sys.argv[2]) if len(sys.argv)>2 else 9000)+int(sys.argv[3]) if len(sys.argv)>3 else 9030):
    M.walk=bytearray(M._walk0)
    s=sim.Sim(M,h,bot.Bot,seed,hunt)
    log=[]
    orig=s.damage
    def dmgf(victim, attacker, amount, weapon, s=s, log=log, orig=orig):
        if victim is s.me and attacker is not s.me:
            log.append((s.t, 'H' if attacker.get('hunter') else ('npc' if attacker.get('kind')=='npc' else 'nonH'), weapon, amount, s.bot.goal_kind))
        return orig(victim, attacker, amount, weapon)
    s.damage=dmgf
    p=s.run(); places.append(p)
    dt=s.me['death_t']
    if dt is None: continue
    b='<60' if dt<60 else '<150' if dt<150 else '<270' if dt<270 else '<400' if dt<400 else 'late'
    buckets[b]+=1
    for (t,who,w,a,mode) in log:
        if t>dt-20: src[(b,who,w)]+=a; modes[(b,mode)]+=a
print('deaths by time',sorted(buckets.items()))
print('dmg last 20s',sorted(src.items(), key=lambda kv:-kv[1])[:12])
print('mode when hit',sorted(modes.items(), key=lambda kv:-kv[1])[:12])
print('mean score', sum(16-p for p in places)/len(places))
