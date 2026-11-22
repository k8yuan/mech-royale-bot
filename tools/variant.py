"""Run sim with module-level overrides: python3 tools/variant.py NAME=VALUE ... -- simargs"""
import sys, importlib
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
i = sys.argv.index('--')
sets, rest = sys.argv[1:i], sys.argv[i + 1:]
import bot
for kv in sets:
    k, v = kv.split('=', 1)
    setattr(bot, k, eval(v))
import sim
sys.argv = ['sim'] + rest
sim.main()
