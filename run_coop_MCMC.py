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

#subjects_string = ",".join(subjects)
for subject in subjects:
    #subjects = '65ea6d657bbd3689a87a1de6,565bff58c121fe0005fc390d,5590a34cfdf99b729d4f69dc'

    ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_MCMC_CMG/run_coop_MCMC.ssub'


    stdout_name = f"{results}/logs/{subject}-%J.stdout"
    stderr_name = f"{results}/logs/{subject}-%J.stderr"

    jobname = f'coop-fit-MCMC-{subject}'
    os.system(f"sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {subject} {results} {experiment_mode}")

    print(f"SUBMITTED JOB [{jobname}]")
    
    
    



###python3  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_MCMC_CMG/run_coop_MCMC.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_MCMC_model_output/coop_MCMC_simfit_7_19_24/ "prolific"


## joblist | grep coop | grep -Po 98.... | xargs scancel