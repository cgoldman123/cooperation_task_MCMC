import sys, os, re, subprocess

subject_list_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/cooperation_prolific_IDs.csv'
results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific



if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")

subjects = []
with open(subject_list_path) as infile:
    for line in infile:
        if 'id' not in line:
            subjects.append(line.strip())

ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/cooperation_task_scripts_CMG/run_coop.ssub'

for subject in subjects:
    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"

    jobname = f'coop-fit-{subject}'
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {results} {experiment_mode}")

    print(f"SUBMITTED JOB [{jobname}]")
    


    ###python3 run_coop.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/coop_model_output/coop_fits_prolific_05-27 "prolific"


    ## joblist | grep coop | grep -Po 98.... | xargs scancel