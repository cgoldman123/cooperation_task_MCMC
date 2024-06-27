% Main script for model fitting the cooperation task data 
dbstop if error
rng(23);
clear all;
if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/coop_MCMC_model_output/'];
    
    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521","BV696","BV360"];
    elseif experiment_mode == "prolific"
        fit_list = ["65ea6d657bbd3689a87a1de6","565bff58c121fe0005fc390d","5590a34cfdf99b729d4f69dc"];
        fit_list = "5ee09c1a0c2ad1027f541f53";
    end
   
elseif isunix
    root = '/media/labs';
    fit_list = string(strsplit(getenv('FIT_LIST'), ','));
    result_dir = getenv('RESULTS');
    experiment_mode = string(getenv('EXPERIMENT'));
    (fit_list)
    (result_dir)
    (experiment_mode)
end

addpath([root '/rsmith/all-studies/core/matjags']);
addpath([root '/rsmith/all-studies/util/spm12/']);
addpath([root '/rsmith/all-studies/util/spm12/toolbox/DEM/']);

merged_data = COP_merge_files(fit_list);

NS = length(fit_list);

% use first subjects' data to get forced choice rewards/actions
num_forced_choices = 3;
num_trials_per_block = 16;
num_blocks = 30;


for sn = 1:NS
    subject_id = char(fit_list(sn));
    subject_data = merged_data(strcmp(cellfun(@char, merged_data.subject_id, 'UniformOutput', false), subject_id), :);

    observations = subject_data(:,{'o1', 'o2', 'o3','o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16'});
    observations = table2array(observations);
    all_observations(sn,1:size(observations,1),:) = observations;


    actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16'});
    actions = table2array(actions);
    all_actions(sn,1:size(actions,1),:) = actions;
end

all_actions = all_actions - 1;
all_observations = all_observations - 1;


datastruct = struct();
datastruct.NS = NS;
datastruct.result_dir = result_dir;
datastruct.subject = char(fit_list(1));
datastruct.num_trials_per_block = num_trials_per_block;
datastruct.num_blocks = num_blocks;
datastruct.all_actions = all_actions;
datastruct.all_observations = all_observations;






% nchains = 4;
% nburnin = 500;
% nsamples = 1000; 
% thin = 1;

nchains = 4;
nburnin = 500;
nsamples = 1000; 
thin = 1;

doparallel = 0;

clear S init0
for i=1:nchains

    S.pa(1:NS) = .25;
    S.eta(1:NS) = .5;
    S.cr(1:NS) = 4;
    S.cl(1:NS) = 4;
    S.alpha(1:NS) = 4;
    S.omega(1:NS) = .25;
    
    init0(i) = S;
end
currdir = pwd;

tic


monitor_params = {'pa','eta','cr','cl','alpha','omega'};

fprintf( 'Running JAGS\n' );
[samples, stats ] = matjags_cmg( ...
    datastruct, ...
    fullfile(currdir, 'coop_model_MCMC.txt'), ...
    init0, ...
    'doparallel' , doparallel, ...
    'nchains', nchains,...
    'nburnin', nburnin,...
    'nsamples', nsamples, ...
    'thin', thin, ...
    'monitorparams', ...
    monitor_params, ...
    'savejagsoutput' , 1 , ...
    'verbosity' , 1 , ...
    'cleanup' , 1  );
toc

% throw out first N-1 samples
N = 1;
stats.mean.pa = squeeze(mean(mean(samples.pa(:,N:end,:,:),2),1));
stats.mean.eta = squeeze(mean(mean(samples.eta(:,N:end,:,:),2),1));
stats.mean.cr = squeeze(mean(mean(samples.cr(:,N:end,:,:),2),1));
stats.mean.cl = squeeze(mean(mean(samples.cl(:,N:end,:,:),2),1));
stats.mean.omega = squeeze(mean(mean(samples.omega(:,N:end,:,:),2),1));
stats.mean.alpha = squeeze(mean(mean(samples.alpha(:,N:end,:,:),2),1));

 






fits = struct();
for si = 1:NS
    fits(si).id = {char(fit_list(si))};
    fits(si).pa = stats.mean.pa(si);  
    fits(si).eta = stats.mean.eta(si);  
    fits(si).cr = stats.mean.cr(si);  
    fits(si).cl = stats.mean.cl(si);  
    fits(si).omega = stats.mean.omega(si);  
    fits(si).alpha = stats.mean.alpha(si);  

    params = fits(si);
    params.forgetting_split = 0;
    params.learning_split = 0;
    params.NB = num_blocks;
    params.T = num_trials_per_block;

    for block=1:num_blocks
        choices = squeeze(all_observations(si,block,:))';
        rewards = squeeze(all_actions(si,block,:))';
        MDP_Block{block} = Simple_TAB_model(params, rewards, choices, 0);
        avg_act_prob(block) = sum(MDP_Block{block}.chosen_action_probabilities)/params.T;
        for trial = 1:params.T
            if MDP_Block{block}.chosen_action_probabilities(trial) == max(MDP_Block{block}.action_probabilities(:,trial))
                acc(block,trial) = 1;
            else
                acc(block,trial) = 0;
            end
        end
    end
    fits(si).average_action_probabilities = sum(avg_act_prob)/params.NB;
    fits(si).average_accuracy = (sum(sum(acc,2))/(params.NB*params.T));
end
if NS == 1
    writetable(struct2table(fits), [result_dir char(fit_list(1)) '_cooperation_task_MCMC_fit.csv']);
    save(fullfile([result_dir char(fit_list(1)) '_cooperation_task_MCMC_samples.mat']), 'samples');
    save(fullfile([result_dir char(fit_list(1)) '_cooperation_task_MCMC_stats.mat']), 'stats'); 
else
    writetable(struct2table(fits), [result_dir 'cooperation_task_MCMC_fits.csv']);
    save(fullfile([result_dir 'cooperation_task_MCMC_samples.mat']), 'samples');
    save(fullfile([result_dir 'cooperation_task_MCMC_stats.mat']), 'stats');
end




