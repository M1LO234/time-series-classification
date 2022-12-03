import os

for f in [s for s in os.listdir('./') if '.sh' in s]:
    os.remove(f)

for f in [s for s in os.listdir('./run_scripts') if '.sh' in s]:
    os.remove(f'./run_scripts{f}')
