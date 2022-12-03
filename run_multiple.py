import os
import stat
import numpy as np

# out_file_name = 'run_all.sh'
main_out = './run_all.sh'
initial_run_files = [f for f in os.listdir('./') if 'fcm_' in f]
initial_run_files.sort()
for f_id, f in enumerate(initial_run_files):
    with open(f'./{f}', 'r') as curr_run:
        content = curr_run.read()
        scripts_list = content.split(" & ")[:-1]
        n_scripts = len(scripts_list)//8
        commands = []

        for i in range(n_scripts):
            commands.append("")
        for script_id in range(len(scripts_list)):
            curr_run_script_id = script_id // 8
            commands[curr_run_script_id] += f'{scripts_list[script_id]} & \n'
            out_file_name = f'run_scripts/run_class{f_id}_{curr_run_script_id}.sh'
            if (script_id+1)%8==0:
                with open(out_file_name, 'a') as fp:
                    fp.write(f'{commands[curr_run_script_id]}wait')
                st = os.stat(out_file_name)
                os.chmod(out_file_name, st.st_mode | stat.S_IEXEC)

all_run_files = [rf for rf in os.listdir('./run_scripts') if '.sh' in rf]
all_run_files.sort()
for run_file in all_run_files:
    with open(main_out, 'a') as fp:
        fp.write(f'./run_scripts/{run_file}\n')
    st = os.stat(main_out)
    os.chmod(main_out, st.st_mode | stat.S_IEXEC)
    