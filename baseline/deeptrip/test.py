import subprocess
for i in range(402):
    command = f'python3 run.py {i}'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
