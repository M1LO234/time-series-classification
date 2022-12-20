import os
import stat
import numpy as np

main_out = './run_all.sh'
initial_run_files = [f for f in os.listdir('./') if 'fcm_' in f]
initial_run_files.sort()
config_n = 0
for f_id, f in enumerate(initial_run_files):
    with open(f'./{f}', 'r') as curr_run:
        content = curr_run.read()
        scripts_list = content.split(" & ")[:-1]
        n_scripts = len(scripts_list)//8
        commands = ''
        max_confs = len(initial_run_files)*len(scripts_list)
        for script_id in range(len(scripts_list)):
            curr_run_script_id = script_id % 8
            commands = f'{scripts_list[script_id]}'
            out_file_name = f'run_scripts/run_class_{curr_run_script_id}.sh'
            with open(out_file_name, 'a') as fp:
                fp.write(f'{commands} & echo "Config {config_n}/{max_confs} started" & wait\n')
            st = os.stat(out_file_name)
            os.chmod(out_file_name, st.st_mode | stat.S_IEXEC)
            config_n += 1

all_run_files = [rf for rf in os.listdir('./run_scripts') if '.sh' in rf]
all_run_files.sort()
for run_file in all_run_files:
    with open(main_out, 'a') as fp:
        fp.write(f'./run_scripts/{run_file} & \n')
with open(main_out, 'a') as fp:
    fp.write('wait')
st = os.stat(main_out)
os.chmod(main_out, st.st_mode | stat.S_IEXEC)
    