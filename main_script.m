% Main script for model fitting the cooperation task data 
dbstop if error
rng(23);
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
num_trials_per_block = 8;
num_blocks = 30;


for sn = 1:NS
    subject_id = char(fit_list(sn));
    subject_data = merged_data(strcmp(cellfun(@char, merged_data.subject_id, 'UniformOutput', false), subject_id), :);

    %observations = subject_data(:,{'o1', 'o2', 'o3','o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16'});
    observations = subject_data(:,{'o1', 'o2', 'o3','o4', 'o5', 'o6', 'o7', 'o8'});
    observations = table2array(observations);
    all_observations(sn,1:size(observations,1),:) = observations;


    %actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16'});
    actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8'});
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
nsamples = 1; 
thin = 1;

doparallel = 0;

clear S init0
for i=1:nchains

    S.p_a(1:NS) = 2.2;
    S.eta(1:NS) = .4;
    S.cr(1:NS) = 4;
    S.cl(1:NS) = 2;
    S.alpha(1:NS) = 2;
    S.omega(1:NS) = .3;
    
    init0(i) = S;
end

tic


monitor_params = {'p_a','eta','cr','cl','alpha','omega'};
%monitor_params = {'alpha'};

fprintf( 'Running JAGS\n' );
[samples, stats ] = matjags( ...
    datastruct, ...
    fullfile(currdir, 'coop_model_MCMC2'), ...
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



for subject = fit_list
    %Fit_file = strcat("./", fit_list(s), "-T1-_COP_R1-_BEH.csv");
    subject
    if experiment_mode == "local"
        fit_results = TAB_fit_simple_local(subject);
    elseif experiment_mode == "prolific"
        fit_results = TAB_fit_simple_prolific(subject);
    end
    save([result_dir '/' char(subject) '_fit_results.mat'], "fit_results");
    
    % assemble output table
    priorFields = fieldnames(fit_results.prior);
    priorValues = struct2cell(fit_results.prior);
    priorTable = cell2table(priorValues', 'VariableNames', strcat('prior_', priorFields));

    postFields = fieldnames(fit_results.prior);
    postValues = struct2cell(fit_results.parameters);
    postTable = cell2table(postValues', 'VariableNames', strcat('posterior_', postFields));
    % Extract additional values
    additionalValuesTable = table({char(subject)}, fit_results.file, fit_results.average_action_probabilities, ...
                                  fit_results.average_accuracy, (fit_results.has_practice_effects), ...
                                  'VariableNames', {'subject', 'file', 'average_action_probabilities', ...
                                                    'average_accuracy', 'has_practice_effects'});
    % Concatenate all tables horizontally
    resultTable = [additionalValuesTable, priorTable, postTable];
    
    writetable(resultTable, [result_dir '/coop_fit_' char(subject) '.csv']);

    
end

