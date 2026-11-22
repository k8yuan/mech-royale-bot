import sys, math, collections
sys.path.insert(0,'.'); sys.path.insert(0,'tools'); sys.argv=['x']
import sim, nav, bot
h=sim.load_hello(); M=nav.Map(h['map']['dir']); M._walk0=bytes(M.walk)
agg=collections.Counter(); n=0; modes=collections.Counter()
for seed in range(5000,5040):
    M.walk=bytearray(M._walk0)
    s=sim.Sim(M,h,bot.Bot,seed,'normal')
    dmg=collections.Counter()
    orig=s.damage
    def dmgf(victim, attacker, amount, weapon, s=s, dmg=dmg, orig=orig):
        if victim is s.me and attacker is not s.me:
            hunter = attacker.get('hunter', False)
            dmg[('H' if hunter else 'nonH', weapon)] += amount
        return orig(victim, attacker, amount, weapon)
    s.damage=dmgf
    kinds=collections.Counter()
    oa=s.bot.act
    def act(o, s=s, kinds=kinds, oa=oa):
        a=oa(o); kinds[s.bot.goal_kind]+=1; return a
    s.bot.act=act
    place=s.run()
    if s.me['death_t'] is not None and s.me['death_t']<60:
        n+=1; agg.update(dmg); modes.update(kinds)
print('early deaths',n,'of 40'); print(agg.most_common()); print(modes.most_common())
