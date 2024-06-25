% Main script for model fitting the cooperation task data 
dbstop if error
rng(23);
clear all;
if ispc
    root = 'L:';
    result_dir = [root '/rsmith/lab-members/cgoldman/Wellbeing/cooperation_task/coop_model_output/'];
    
    experiment_mode = "prolific";
    if experiment_mode == "local"
        fit_list = ["BW521","BV696","BV360"];
    elseif experiment_mode == "prolific"
        fit_list = ["65ea6d657bbd3689a87a1de6","565bff58c121fe0005fc390d","5590a34cfdf99b729d4f69dc"];
    end
   
elseif isunix
    root = '/media/labs';
    fit_list = string(getenv('SUBJECT'));
    result_dir = getenv('RESULTS');
    experiment_mode = string(getenv('EXPERIMENT'));
    (fit_list)
    (result_dir)
    (experiment_mode)
end
currdir = pwd;
addpath(['L:/rsmith/all-studies/core/matjags']);
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
    %observations = subject_data(:,{'o1', 'o2', 'o3','o4', 'o5', 'o6', 'o7', 'o8'});
    observations = table2array(observations);
    all_observations(sn,1:size(observations,1),:) = observations;


    actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16'});
    %actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8'});
    actions = table2array(actions);
    all_actions(sn,1:size(actions,1),:) = actions;
end

all_actions = all_actions - 1;
all_observations = all_observations - 1;


datastruct = struct();
datastruct.NS = NS;
datastruct.num_forced_choices = num_forced_choices;
datastruct.num_trials_per_block = num_trials_per_block;
datastruct.num_blocks = num_blocks;
datastruct.all_actions = all_actions;
datastruct.all_observations = all_observations;






nchains = 1;
nburnin = 500;
nsamples = 100; 
thin = 1;

doparallel = 0;

clear S init0
for i=1:nchains

    S.pa(1:NS) = 2.2;
    S.eta(1:NS) = .4;
    S.cr(1:NS) = 4;
    S.cl(1:NS) = 2;
    S.alpha(1:NS) = 2;
    S.omega(1:NS) = .3;
    
    init0(i) = S;
end

tic


monitor_params = {'pa','eta','cr','cl','alpha','omega'};
%monitor_params = {'alpha'};

fprintf( 'Running JAGS\n' );
[samples, stats ] = matjags( ...
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
        fits(si).id = {char(fit_list(si));};
        fits(si).pa = stats.mean.pa(si);  
        fits(si).eta = stats.mean.eta(si);  
        fits(si).cr = stats.mean.cr(si);  
        fits(si).cl = stats.mean.cl(si);  
        fits(si).omega = stats.mean.omega(si);  
        fits(si).alpha = stats.mean.alpha(si);  
    end
    
fits = struct2table(fits);
