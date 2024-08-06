import sys, os, re, subprocess, csv

# this script takes previously fit parameters in fit_results (e.g. fits using variational bayes) then runs a matlab script to simulate
# behavior with those parameters, then fit the simulated behavior

fit_results = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/data_analysis/compiled_model_fits_VB/coop_VB_prolific_7_19_24.csv'
results = sys.argv[1]
experiment_mode = sys.argv[2] # indicate inperson, mturk, or prolific



if not os.path.exists(results):
    os.makedirs(results)
    print(f"Created results directory {results}")

if not os.path.exists(f"{results}/logs"):
    os.makedirs(f"{results}/logs")
    print(f"Created results-logs directory {results}/logs")




ssub_path = '/media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_MCMC_CMG/run_coop_MCMC_simfit.ssub'


with open(fit_results, newline='') as csvfile:
    file = csv.DictReader(csvfile)

    for subject in file:
        stdout_name = f'{results}/logs/{subject["id"]}-%J.stdout'
        stderr_name = f'{results}/logs/{subject["id"]}-%J.stderr'

        jobname = f'coop-fit-{subject["id"]}'
        os.system(f'sbatch -J {jobname} -o {stdout_name} -e {stderr_name} {ssub_path} {results} {experiment_mode} {subject["id"]} {subject["posterior_alpha"]} {subject["posterior_eta"]} {subject["posterior_omega"]} {subject["posterior_p_a"]} {subject["posterior_cr"]} {subject["posterior_cl"]}') 

        print(f"SUBMITTED JOB [{jobname}]")
        
    
    
    



###python3  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/scripts/cooperation_task_scripts_MCMC_CMG/run_coop_MCMC_simfit.py  /media/labs/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_MCMC_model_output/coop_MCMC_simfit_using_VB_params_7-20/ "prolific"


## joblist | grep coop | grep -Po 98.... | xargs scancel