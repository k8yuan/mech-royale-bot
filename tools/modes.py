import sys, collections
mod=sys.argv[1]; flags=sys.argv[2:]
sys.path.insert(0,'.'); sys.path.insert(0,'tools'); sys.path.insert(0,'tools/archive'); sys.argv=['x']+flags
import importlib, sim, nav
B=importlib.import_module(mod)
h=sim.load_hello(); M=nav.Map(h['map']['dir']); M._walk0=bytes(M.walk)
modes=collections.Counter(); dmg=collections.Counter(); kills=0; life=0; N=20
for seed in range(700,700+N):
    M.walk=bytearray(M._walk0)
    s=sim.Sim(M,h,B.Bot,seed,'squad','omni',True)
    oa=s.bot.act
    def act(o,s=s,oa=oa):
        a=oa(o); modes[s.bot.goal_kind]+=1; return a
    s.bot.act=act
    od=s.damage
    def dm(v,a,amt,w,s=s,od=od):
        if v is s.me and a is not s.me: dmg[s.bot.goal_kind]+=amt
        return od(v,a,amt,w)
    s.damage=dm
    s.run(); kills+=s.kills; life+=s.me['death_t'] or 480
tot=sum(modes.values())
print(mod,'kills',kills/N,'life',round(life/N))
print('time%',{k:round(100*v/tot,1) for k,v in modes.most_common()})
print('dmg taken by mode',{k:round(v/N) for k,v in dmg.most_common()})
