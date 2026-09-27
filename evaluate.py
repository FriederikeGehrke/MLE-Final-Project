import json, subprocess, sys
agent = sys.argv[1]
rounds = sys.argv[2] if len(sys.argv) > 2 else '300'
opponents = sys.argv[3:] or ['rule_based_agent'] * 3
out = f'eval_{agent}.json'
subprocess.run(['python', 'main.py', 'play', '--agents', agent, *opponents, '--scenario', 'classic',
                '--no-gui', '--seed', '1', '--n-rounds', rounds, '--save-stats', out],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for name, a in json.load(open(out))['by_agent'].items():
    r = a['rounds']
    print(f'{name:22s} score {a["score"]/r:5.2f} coins {a.get("coins",0)/r:5.2f} '
          f'kills {a.get("kills",0)/r:5.2f} suicides {a.get("suicides",0)/r:5.2f}')