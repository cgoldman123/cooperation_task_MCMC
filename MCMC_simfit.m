function [sim_fits, sim_samples, sim_stats] = MCMC_simfit(subject_data, params,conf)
    dbstop if error;
    MDP = rmfield(params, 'deviance');
    MDP.forgetting_split = 0;
    MDP.learning_split = 0;
    MDP.NB = conf.num_blocks;
    MDP.T = conf.num_trials_per_block;
    NS = 1;

    

    observations = subject_data(:,{'o1', 'o2', 'o3','o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11', 'o12', 'o13', 'o14', 'o15', 'o16'});
    observations = table2array(observations);
    all_observations(1,1:size(observations,1),:) = observations;

    actions = subject_data(:,{'u1', 'u2', 'u3','u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16'});
    actions = table2array(actions);
    all_actions(1,1:size(actions,1),:) = actions;



    
    
    
    % get block probabilities from schedule
    schedule = readtable('../../task_schedule/prolific_30_block_schedule.xlsx');
    block_probs = zeros(3,3,MDP.NB);
    for i = 1:MDP.NB
        block_probs(:,1,i) = str2double(strsplit(schedule.good_probabilities{i},'_'))';
        block_probs(:,2,i) = str2double(strsplit(schedule.safe_probabilities{i},'_'))';
        block_probs(:,3,i) = str2double(strsplit(schedule.bad_probabilities{i},'_'))';
    end
    
    
    
    
    %% SIMULATE BEHAVIOR
    fprintf("Simulating behavior\n");
    MDP_Block = cell(1, MDP.NB);
    for block = 1:MDP.NB
        MDP.BlockProbs = block_probs(:,:,block);
        rewards = squeeze(all_observations(1,block,:))' - 1;
        choices = squeeze(all_actions(1,block,:))' - 1;
        MDP_Block{block} = Simple_TAB_model_v2(MDP, rewards, choices, 1);
    end
    % Assemble datastruct with simulated actions and observations

    
    simmed_choices = cell2mat(cellfun(@(c) c.choices', MDP_Block, 'UniformOutput', false));
    simmed_outcomes = cell2mat(cellfun(@(c) c.outcomes', MDP_Block, 'UniformOutput', false));
    datastruct.all_actions = reshape(simmed_choices', 1, conf.num_blocks, conf.num_trials_per_block);
    datastruct.all_observations = reshape(simmed_outcomes', 1, conf.num_blocks, conf.num_trials_per_block);
    datastruct.NS = NS;
    datastruct.result_dir = conf.result_dir;
    datastruct.subject = char(conf.fit_list(1));
    datastruct.num_trials_per_block = conf.num_trials_per_block;
    datastruct.num_blocks = conf.num_blocks;

    monitor_params = {'opt','eta','cr','cl','alpha','omega'};
    
    nchains = conf.nchains;
    nburnin = conf.nburnin;
    nsamples = conf.nsamples;
    thin = conf.thin;
    doparallel = conf.doparallel;
    
    
    clear S init0
    for i=1:nchains

        S.opt(1:NS) = .5;
        S.eta(1:NS) = .5;
        S.cr(1:NS) = 1;
        S.cl(1:NS) = 1;
        S.alpha(1:NS) = 4;
        S.omega(1:NS) = .5;

        init0(i) = S;
    end
    currdir = pwd;
   
    %% FIT SIMULATED BEHAVIOR
    fprintf("Fitting simulated behavior!\n");
    tic

    fprintf( 'Running JAGS\n' );
    [sim_samples, sim_stats ] = matjags_cmg( ...
        datastruct, ...
        fullfile(currdir, 'coop_model_MCMC_v2.txt'), ...
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
    
     % MEAN
    % throw out first N-1 samples
    N = 1;
    for i=1:length(monitor_params)
        sim_stats.mean.(monitor_params{i}) = squeeze(mean(mean(sim_samples.(monitor_params{i})(:,N:end,:,:),2),1));
    end


    % MODE
    % Loop through each parameter, compute the mode after rounding, and store in stats
%     for i = 1:length(monitor_params)
%         data_vector = round(sim_samples.(monitor_params{i})(:), 2);
%         [unique_values, ~, idx] = unique(data_vector);
%         frequency = accumarray(idx, 1);
%         [~, max_idx] = max(frequency);
%         sim_stats.mode.(monitor_params{i}) = unique_values(max_idx);
%     end



    for si = 1:NS 
        sim_fits(si).id = {char(conf.fit_list(si))};
        %mean_or_mode = {'mean','mode'};
        mean_or_mode = {'mean'};
        for i = 1:length(mean_or_mode)
            for fn = fieldnames(sim_stats.(mean_or_mode{i}))'
                sim_fits(si).([mean_or_mode{i} '_' (fn{1})]) = sim_stats.(mean_or_mode{i}).(fn{1});
            end
            params = sim_stats.(mean_or_mode{i}); 
            params.forgetting_split = 0;
            params.learning_split = 0;
            params.NB = conf.num_blocks;
            params.T = conf.num_trials_per_block;

            for block=1:conf.num_blocks
                rewards = squeeze(datastruct.all_observations(si,block,:))';
                choices = squeeze(datastruct.all_actions(si,block,:))';
                MDP_Block{block} = Simple_TAB_model_v2(params, rewards, choices, 0);
                avg_act_prob(block) = sum(MDP_Block{block}.chosen_action_probabilities(4:end))/(params.T-3);
                for trial = 4:params.T
                    if MDP_Block{block}.chosen_action_probabilities(trial) == max(MDP_Block{block}.action_probabilities(:,trial))
                        acc(block,trial-3) = 1;
                    else
                        acc(block,trial-3) = 0;
                    end
                end
            end
            sim_fits(si).(['avg_act_prob_' mean_or_mode{i} '_params'])  = sum(avg_act_prob)/params.NB;
            sim_fits(si).(['model_acc_' mean_or_mode{i} '_params'])  = (sum(sum(acc,2))/(params.NB*(params.T-3)));
        end
        sim_fits(si).nchains = nchains;
        sim_fits(si).nburnin = nburnin;
        sim_fits(si).nsamples = nsamples;
        sim_fits(si).thin = thin;
        sim_fits(si).throwaway = N-1;

    end
    
    

end