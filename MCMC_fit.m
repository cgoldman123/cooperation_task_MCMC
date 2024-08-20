function [fits, samples, stats] = MCMC_fit(fit_list,merged_data, conf)
    monitor_params = {'opt','eta','cr','cl','alpha','omega'};

    NS = length(fit_list);
    result_dir = conf.result_dir;
    num_trials_per_block = conf.num_trials_per_block;
    num_blocks = conf.num_blocks;
    
    nchains = conf.nchains;
    nburnin = conf.nburnin;
    nsamples = conf.nsamples;
    thin = conf.thin;
    doparallel = conf.doparallel;



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



    clear S init0
    for i=1:nchains

        S.opt(1:NS) = .5;
        S.eta(1:NS) = .5;
        S.cr(1:NS) = 1;
        S.cl(1:NS) = 1;
        S.alpha(1:NS) = 4;
        S.omega(1:NS) = .2;

        init0(i) = S;
    end
    currdir = pwd;

    tic

    fprintf( 'Running JAGS\n' );
    [samples, stats ] = matjags_cmg( ...
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
    N = conf.N;
    for i=1:length(monitor_params)
        stats.mean.(monitor_params{i}) = squeeze(mean(mean(samples.(monitor_params{i})(:,N:end,:,:),2),1));
    end

    
    

    % MODE
    % Loop through each parameter, compute the mode after rounding, and store in stats
%     for i = 1:length(monitor_params)
%         data_vector = round(samples.(monitor_params{i})(:), 2);
%         [unique_values, ~, idx] = unique(data_vector);
%         frequency = accumarray(idx, 1);
%         [~, max_idx] = max(frequency);
%         stats.mode.(monitor_params{i}) = unique_values(max_idx);
%     end



    for si = 1:NS 
        fits(si).id = {char(fit_list(si))};
        %mean_or_mode = {'mean','mode'};
        mean_or_mode = {'mean'};

        for i = 1:length(mean_or_mode)
            for fn = fieldnames(stats.(mean_or_mode{i}))'
                fits(si).([mean_or_mode{i} '_' (fn{1})]) = stats.(mean_or_mode{i}).(fn{1});
            end
            params = stats.(mean_or_mode{i}); 
            params.forgetting_split_matrix = 0;
            params.forgetting_split_row = 0;
            params.learning_split = 0;
            params.NB = num_blocks;
            params.T = num_trials_per_block;

            for block=1:num_blocks
                rewards = squeeze(all_observations(si,block,:))';
                choices = squeeze(all_actions(si,block,:))';
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
            fits(si).(['avg_act_prob_' mean_or_mode{i} '_params'])  = sum(avg_act_prob)/params.NB;
            fits(si).(['model_acc_' mean_or_mode{i} '_params'])  = (sum(sum(acc,2))/(params.NB*(params.T-3)));
        end
        fits(si).nchains = nchains;
        fits(si).nburnin = nburnin;
        fits(si).nsamples = nsamples;
        fits(si).thin = thin;
        fits(si).throwaway = N-1;

    end
end