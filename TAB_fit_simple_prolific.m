function fit_results = TAB_fit_simple_prolific(subject)
    %% Add Subj Data (Parse the data files)
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    file_path = [root 'NPC/DataSink/StimTool_Online/WB_Cooperation_Task/'];
    
    
    has_practice_effects = false;
    % Manipulate Data
    directory = dir(file_path);
    % sort by date
    dates = datetime({directory.date}, 'InputFormat', 'dd-MMM-yyyy HH:mm:ss');
    % Sort the dates and get the sorted indices
    [~, sortedIndices] = sort(dates);
    % Use the sorted indices to sort the structure array
    sortedDirectory = directory(sortedIndices);

    index_array = find(arrayfun(@(n) contains(sortedDirectory(n).name, strcat('cooperation_task_',subject, '_T1_Pilot_R1_NEW')),1:numel(sortedDirectory)));
    if length(index_array) > 1
        disp("WARNING, MULTIPLE BEHAVIORAL FILES FOUND FOR THIS ID. USING THE FIRST FULL ONE")
    end

    for k = 1:length(index_array)
        file_index = index_array(k);
        file = [file_path sortedDirectory(file_index).name];

        subdat = readtable(file);

        if any(cellfun(@(x) isequal(x, 'MAIN_START'), subdat.trial_type))
            if size(subdat,1)<2020
                has_practice_effects = true;
                continue;
            end
        end
        
        %% 12. Set up model structure
        %==========================================================================
        %==========================================================================

    
        TpB = 16;     % trials per block
        NB  = 30;     % number of blocks
        N   = TpB*NB; % trials per block * number of blocks

        first_game_trial = min(find(ismember(subdat.trial_type, 'MAIN_START'))) +2;
        clean_subdat = subdat(first_game_trial:end, :);

        trial_types = clean_subdat.trial_type(clean_subdat.event_type==3,:);
        location_code = zeros(NB, 3);
        force_choice = zeros(NB, 3);
        force_outcome = zeros(NB, 3);

        location_map = containers.Map({'g', 's', 'b'}, [2, 3, 4]);
        force_choice_map = containers.Map({'g', 's', 'b'}, [1, 2, 3]);
        force_outcome_map = containers.Map({'W', 'N', 'L'}, [1, 2, 3]);

        for i = 1:length(trial_types)
            underscore_indices = strfind(trial_types{i}, '_');
            letters = trial_types{i}(underscore_indices(1)+1:underscore_indices(1)+3);
            location_code(i, :) = arrayfun(@(c) location_map(c), letters);
            forced_letters = trial_types{i}(underscore_indices(2)+1:underscore_indices(2)+3);
            force_choice(i, :) = arrayfun(@(c) force_choice_map(c), forced_letters);
            forced_outcome_letters = trial_types{i}(underscore_indices(3)+1:underscore_indices(3)+3);
            force_outcome(i, :) = arrayfun(@(c) force_outcome_map(c), forced_outcome_letters);
        end



        %BlockProbs = load('BlockProbs_all.mat').BlockProbs_all;
        %force_choice = load('force_choice.mat').force_choice;
        %force_outcome = load('force_outcome.mat').force_outcome;
        %location_code = load('location_code.mat').location_code;

        params.force_choice = force_choice;
        params.force_outcome = force_outcome;
            %--------------------------------------------------------------------------
        learning = 1; %fit eta?
        forgetting = 1; %fit omega?

        params.T = TpB;
        params.p_a = .25; %inverse information sensitivity (& lower bound on forgetting)
        params.cr = 4; %Reward Seeking
        params.cl = 1;
        params.alpha = 4; %Action Precision
        params.eta = 1; %Learning rate
        params.omega = 1; %Forgetting rate

        %if splitting forgetting rates
        params.forgetting_split = 0; % 1 = separate wins/losses, 0 = not
             %params.omega_win = 1;
             %params.omega_loss = 1;

        %if splitting learning rates
         params.learning_split = 0; % 1 = separate wins/losses, 0 = not
             %params.eta_win = 1;
             %params.eta_loss = 1;

        % specify true reward probabilities if simulating (won't influence fitting)
        p1 = .9;
        p2 = .9;
        p3 = .9;

        true_probs = [p1   p2   p3   ;
                      1-p1 1-p2 1-p3];

        % simulating or fitting
        sim = 0; %1 = simulating, 0 = fitting
        %sim.true_probs = true_probs;


        % if not fitting (specify if fitting)
        rewards = [];
        choices = [];

        % rewards = [1 1 2...]; % length T
        % choices = [1 2 3...]; % length T



        % parse observations and actions
        sub.o = clean_subdat.result(clean_subdat.event_type == 5);
        sub.u = clean_subdat.response(clean_subdat.event_type == 5);

        for i = 1:N
            if sub.o{i,1} == "positive"
                sub.o{i,1} = 2;
            elseif sub.o{i,1} == "neutral"
                sub.o{i,1} = 3;
            elseif sub.o{i,1} == "negative"
                sub.o{i,1} = 4;
            end
        end
        sub.o = cell2mat(sub.o);

        for i = 1:NB
            for j = 1:TpB
                if sub.u{16*(i-1)+j,1}(1) == 'l' %== "left"
                    sub.u{16*(i-1)+j,1} = location_code(i,1);
                elseif sub.u{16*(i-1)+j,1}(1) == 'u' %== "up"
                    sub.u{16*(i-1)+j,1} = location_code(i,2);
                elseif sub.u{16*(i-1)+j,1}(1) == 'r' %== "right"
                    sub.u{16*(i-1)+j,1} = location_code(i,3);
                end
            end
        end

        sub.u = cell2mat(sub.u);

        o_all = [];
        u_all = [];

        for n = 1:NB
            o_all = [o_all sub.o((n*TpB-(TpB-1)):TpB*n,1)];
            u_all = [u_all sub.u((n*TpB-(TpB-1)):TpB*n,1)];
        end
        %% 6.2 Invert model and try to recover original parameters:
        %==========================================================================

        %--------------------------------------------------------------------------
        % This is the model inversion part. Model inversion is based on variational
        % Bayes. The basic idea is to maximise (negative) variational free energy
        % wrt to the free parameters (here: alpha and cr). This means maximising
        % the likelihood of the data under these parameters (i.e., maximise
        % accuracy) and at the same time penalising for strong deviations from the
        % priors over the parameters (i.e., minimise complexity), which prevents
        % overfitting.
        % 
        % You can specify the prior mean and variance of each parameter at the
        % beginning of the TAB_spm_dcm_mdp script.
        %--------------------------------------------------------------------------

      %  params.BlockProbs = BlockProbs;
        params.NB = NB;


        DCM.MDP    = params;                  % MDP model

        if forgetting==1 & learning==1
                if params.forgetting_split==1 & params.learning_split==1
                DCM.field  = {'alpha' 'cr' 'eta_win' 'eta_loss','omega_win' 'omega_loss', 'p_a'}; % Parameter field

                elseif params.forgetting_split==1 & params.learning_split==0
                DCM.field  = {'alpha' 'cr' 'eta','omega_win' 'omega_loss', 'p_a'}; % Parameter field

                elseif params.forgetting_split==0 & params.learning_split==1
                DCM.field  = {'alpha' 'cr' 'eta_win' 'eta_loss','omega', 'p_a'}; % Parameter field

                else

                DCM.field  = {'alpha' 'cr' 'cl' 'eta' 'omega' 'p_a'}; % Parameter field
                end
        elseif forgetting==0 & learning==1
                if params.learning_split==1
                DCM.field  = {'alpha' 'cr' 'eta_win' 'eta_loss', 'p_a'}; % Parameter field

                else 
                DCM.field  = {'alpha' 'cr' 'eta','p_a'}; % Parameter field
                end
        elseif forgetting==1 & learning==0
                if params.forgetting_split==1
                DCM.field  = {'alpha' 'cr' 'omega_win' 'omega_loss', 'p_a'}; % Parameter field

                else 
                DCM.field  = {'alpha' 'cr' 'omega','p_a'}; % Parameter field
                end
        end


        DCM.U      = {o_all};              % trial specification (stimuli)
        DCM.Y      = {u_all};              % responses (action)

        DCM        = TAB_inversion_simple(DCM);   % Invert the model

        %% 6.3 Check deviation of prior and posterior means & posterior covariance:
        %==========================================================================

        %--------------------------------------------------------------------------
        % re-transform values and compare prior with posterior estimates
        %--------------------------------------------------------------------------
        field = fieldnames(DCM.M.pE);
        for i = 1:length(field)
            if strcmp(field{i},'eta_neu')
                prior.eta_neu = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_neu = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'eta_win')
                prior.eta_win = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_win = 1/(1+exp(-DCM.Ep.(field{i})));  
            elseif strcmp(field{i},'eta_loss')
                prior.eta_loss = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta_loss = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'eta')
                prior.eta = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.eta = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega') 
                prior.omega = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega_win')
                prior.omega_win = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega_win = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'omega_loss')
                prior.omega_loss = 1/(1+exp(-DCM.M.pE.(field{i})));
                mdp.omega_loss = 1/(1+exp(-DCM.Ep.(field{i}))); 
            elseif strcmp(field{i},'alpha')
                prior.alpha = exp(DCM.M.pE.(field{i}));
                mdp.alpha = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'cr')
                prior.cr = exp(DCM.M.pE.(field{i}));
                mdp.cr = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'cl')
                prior.cl = exp(DCM.M.pE.(field{i}));
                mdp.cl = exp(DCM.Ep.(field{i}));
            elseif strcmp(field{i},'p_a')
                prior.p_a = exp(DCM.M.pE.(field{i}));
                mdp.p_a = exp(DCM.Ep.(field{i}));
            end
        end


        all_MDPs = [];

        U_Block = DCM.U{:}-1;
        rewards = reshape(U_Block,params.T,params.NB)';

        Y_Block = DCM.Y{:}-1;
        choices = reshape(Y_Block,params.T,params.NB)';

        mdp.T = params.T;
        mdp.learning_split = params.learning_split; % 1 = separate wins/losses, 0 = not
        mdp.forgetting_split = params.forgetting_split; % 1 = separate wins/losses, 0 = not

        %     %if splitting learning rates
        %          params.omega_win = 1;
        %          params.omega_loss = 1;
        % 
        %     %if splitting learning rates
        %          params.eta_win = .5;
        %          params.eta_loss = .5;
        %         
        % %Simulate beliefs using fitted values
        for block=1:params.NB
            mdp.force_choice = params.force_choice(block,:);
            mdp.force_outcome = params.force_outcome(block,:);
            MDP_Block{block} = Simple_TAB_model(mdp, rewards(block,:), choices(block,:), 0);
            avg_act_prob(block) = sum(MDP_Block{block}.chosen_action_probabilities)/mdp.T;

            for trial = 1:params.T
                if MDP_Block{block}.chosen_action_probabilities(trial) == max(MDP_Block{block}.action_probabilities(:,trial))
                    acc(block,trial) = 1;
                else
                    acc(block,trial) = 0;
                end
            end
        end

        
        average_action_probabilities = sum(avg_act_prob)/params.NB;
        average_accuracy = (sum(sum(acc,2))/(params.NB*params.T));
        
        fieldsToRemove = {'T', 'force_outcome', 'force_choice', 'forgetting_split', 'learning_split'};
        % Remove the specified fields from the struct mdp
        mdp = rmfield(mdp, fieldsToRemove);

        fit_results.file = {file};
        fit_results.prior = prior;
        fit_results.parameters = mdp;
        fit_results.param_names = DCM.field;
        fit_results.DCM = DCM;
        fit_results.simulations = MDP_Block;
        fit_results.average_action_probabilities = average_action_probabilities;
        fit_results.average_accuracy = average_accuracy;
        fit_results.has_practice_effects = has_practice_effects;
        break;
        
        
    end
        

end