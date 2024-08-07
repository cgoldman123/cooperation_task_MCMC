% Main script for model fitting the cooperation task data 
dbstop if error
rng(23);
clear all;
SIMFIT = true;
SIM_PARAMS_PASSED_IN = false;

if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/modeling_output/coop_MCMC_model_output/'];
    
    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521"];
    elseif experiment_mode == "prolific"
        fit_list = ["65ea6d657bbd3689a87a1de6","565bff58c121fe0005fc390d","5590a34cfdf99b729d4f69dc"];
        fit_list = "5590a34cfdf99b729d4f69dc";
    end
    
    simfit_alpha = 3.1454473;
    simfit_eta = 0.4853053;
    simfit_omega = 0.46222943;
    simfit_pa = 0.68625881;
    simfit_cr = 6.9241437;
    simfit_cl = 4.4656905;
   
elseif isunix
    root = '/media/labs';
    fit_list = string(strsplit(getenv('SUBJECT'), ','))
    result_dir = getenv('RESULTS')
    experiment_mode = string(getenv('EXPERIMENT'))
    
    if SIM_PARAMS_PASSED_IN
        simfit_alpha = str2double(getenv('ALPHA'))
        simfit_cr = str2double(getenv('CR'))
        simfit_cl = str2double(getenv('CL'))
        simfit_eta = str2double(getenv('ETA'))
        simfit_omega = str2double(getenv('OMEGA'))
        simfit_pa = str2double(getenv('P_A'))
    end
      
    
end

%addpath([root '/rsmith/all-studies/core/matjags']);
addpath([root '/rsmith/lab-members/cgoldman/general/']);
addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);
addpath([root '/rsmith/all-studies/util/matlab-bayesian-estimation-master/']);
addpath([root '/rsmith/all-studies/util/matlab-bayesian-estimation-master/fileExchange/mcmcdiag']);


NS = length(fit_list);

% use first subjects' data to get forced choice rewards/actions
conf.num_forced_choices = 3;
conf.num_trials_per_block = 16;
conf.num_blocks = 15;


conf.nchains = 4;
conf.nburnin = 500;
conf.nsamples = 2000; 
conf.N = 501; % 1 - throwaway
%conf.nburnin = 10;
%conf.nsamples = 20;
conf.thin = 1;
conf.doparallel = 1;
conf.result_dir = result_dir;
conf.fit_list = fit_list;




merged_data = COP_merge_files(fit_list);

% even if passing in params to simfit, we must fit this subject to set up
% the mdp
[fits, samples, stats] = MCMC_fit(fit_list,merged_data, conf);
fields = setdiff(fieldnames(samples), {'deviance'});

% assess convergence in samples that were accepted (removing discarded
% samples based on conf.N)
discard_ratio = conf.N/conf.nsamples;
for i=1:length(fields)
    param = fields{i};
    R = mbe_gelmanPlot(samples.(param)');
    R = R(ceil(length(R)*discard_ratio):end);
    fits.([fields{i} '_max_rubin_stat']) = max(R);
end


if NS == 1
    if SIMFIT
        subject_id = char(fit_list(1));
        subject_data = merged_data(strcmp(cellfun(@char, merged_data.subject_id, 'UniformOutput', false), subject_id), :);
        
        if SIM_PARAMS_PASSED_IN
            params.alpha = simfit_alpha;
            params.eta = simfit_eta;
            params.omega = simfit_omega;
            params.pa = simfit_pa;
            params.cr = simfit_cr;
            params.cl = simfit_cl;    
            params.deviance = 'null';
            [sim_fit, sim_samples, sim_stats] = MCMC_simfit(subject_data, params,conf);
            
            % if passing in params, we don't care about the previously fit
            % params to set up the mdp
            for i = 1:length(fields)
                sim_fit.(['simulated_' fields{i}]) = params.(fields{i});
            end
            fits = sim_fit; samples = sim_samples; stats = sim_stats;
            fits.id = {subject_id};
        else
            params = stats.mean;
            [sim_fit, sim_samples, sim_stats] = MCMC_simfit(subject_data, params,conf);
            fields = setdiff(fieldnames(sim_fit), {'id'});
            for i = 1:length(fields)
                fits.(['simfit_' fields{i}]) = sim_fit.(fields{i});
            end
            fields = fieldnames(sim_samples);
            for i = 1:length(fields)
                samples.(['simfit_' fields{i}]) = sim_samples.(fields{i});
            end
            fields = fieldnames(sim_stats);
            for i = 1:length(fields)
                stats.(['simfit_' fields{i}]) = sim_stats.(fields{i});
            end
        end
        

        fit_type = 'simfit';
    else
        fit_type = 'fit';
    end
    writetable(struct2table(fits), [result_dir char(fit_list(1)) '_' fit_type '_coop_MCMC.csv']);
    save(fullfile([result_dir char(fit_list(1)) '_' fit_type '_coop_MCMC_samples.mat']), 'samples');
    save(fullfile([result_dir char(fit_list(1)) '_' fit_type '_coop_MCMC_stats.mat']), 'stats'); 

else
    writetable(struct2table(fits), [result_dir 'cooperation_task_MCMC_fits.csv']);
    save(fullfile([result_dir 'cooperation_task_MCMC_samples.mat']), 'samples');
    save(fullfile([result_dir 'cooperation_task_MCMC_stats.mat']), 'stats');
end




