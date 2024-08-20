% % % Simple TAB Model
%
% % Generative model to run with Simple TAB model
%
% clear all
% close all
%
% %Parameters
% params.T = 16;
% params.p_a = .25; %inverse information sensitivity (& lower bound on forgetting)
% params.cr = 2; %Reward Seeking
% params.alpha = 32; %Action Precision
% params.eta = .5; %Learning rate
% params.omega = 1; %Forgetting rate
%
%     %if splitting forgetting rates
%     params.forgetting_split = 0; % 1 = separate wins/losses, 0 = not
%          params.omega_win = .9;
%          params.omega_loss = .8;
%
%     %if splitting learning rates
%      params.learning_split = 0; % 1 = separate wins/losses, 0 = not
%          params.eta_win = .9;
%          params.eta_loss = .95;
%
% % specify true reward probabilities if simulating (won't influence fitting)
% p1 = .9;
% p2 = .5;
% p3 = .1;
%
% true_probs = [p1   p2   p3   ;
%               1-p1 1-p2 1-p3];
%
% % if simulating
% sim.toggle = 1;
% sim.true_probs = true_probs;
%
% % if not fitting (specify if fitting)
% rewards = [];
% choices = [];
%
% % rewards = [1 1 2...]; % length T
% % choices = [1 2 3...]; % length T


function [model_output] = Simple_TAB_model_v2(params, rewards, choices, sim)


a_0 = [params.opt params.opt params.opt;
       1/2*(1-params.opt) 1/2*(1-params.opt) 1/2*(1-params.opt);
       1/2*(1-params.opt) 1/2*(1-params.opt) 1/2*(1-params.opt)];

B_pi = [1 0 0;
        0 1 0;
        0 0 1];

C =  spm_softmax([params.cr+params.cl+eps params.cl+eps eps]');

outcome_vector = zeros(3,params.T);
F = nan(3,params.T);
G_error = nan(1,params.T);
chosen_action_probs = nan(1,params.T);

for t = 1:params.T
    if t <= 3
        if t == 1
            a{t} = a_0;
        end
%         actions(t) = params.force_choice(t);
%         outcomes(t) = params.force_outcome(t);
%         outcome_vector(params.force_outcome(t),t) = 1;
        
        actions(t) = choices(t); 
        outcomes(t) = rewards(t);    
        outcome_vector(outcomes(t),t) = 1; 
        % only accumulate concentration parameters
        % learning part
        if params.learning_split
            if outcomes(t) == 1
                eta = params.eta_win;
            elseif outcomes(t) == 2
                eta = params.eta_neutral;
            elseif outcomes(t) == 3
                eta = params.eta_loss;
            end
        else
        eta = params.eta;
        end

        %forgetting part
        if params.forgetting_split_matrix
            if outcomes(t) == 1
                omega = params.omega_win;
            elseif outcomes(t) == 2
                omega = params.omega_neutral;
            elseif outcomes(t) == 3
                omega = params.omega_loss;
            end
            a{t+1} = (a{t} - a_0)*(1-omega)+ a_0;

        elseif params.forgetting_split_row
            omega = [1-params.omega_win; 1-params.omega_neutral; 1-params.omega_loss];
            a{t+1} = (a{t} - a_0).* repmat(omega, 1, 3)+ a_0;
        else
            omega = params.omega;
            a{t+1} = (a{t} - a_0)*(1-omega)+ a_0;
        end 
        
        a{t+1} = a{t+1}+eta*(B_pi(:,actions(t))*outcome_vector(:,t)')';
        %learning_rate
       
    elseif t > 3
        
        A{t} = spm_norm(a{t});
      
        
        
        % compute information gain
        a_sums{t} = [sum(a{t}(:,1)) sum(a{t}(:,2)) sum(a{t}(:,3));
                     sum(a{t}(:,1)) sum(a{t}(:,2)) sum(a{t}(:,3));
                     sum(a{t}(:,1)) sum(a{t}(:,2)) sum(a{t}(:,3))];
        
        info_gain = .5*((a{t}.^-1)-(a_sums{t}.^-1));
        
        % compute expected free energies
        for pol = 1:3
            epistemic_value(pol,t) = dot(A{t}*B_pi(:,pol),info_gain*B_pi(:,pol));
            pragmatic_value(pol,t) = dot(A{t}*B_pi(:,pol),log(C));
            G(pol,t) = -epistemic_value(pol,t) - pragmatic_value(pol,t);
        end
        
        % compute action probabilities
        action_probs(:,t) = spm_softmax(params.alpha*-G(:,t));
        
        
        
        % select actions
        if sim == 1 %.toggle == 1
            actions(t) = find(rand < cumsum(action_probs (:,t)),1);
        else
            actions(t) = choices(t);
        end
        
        chosen_action_probs(:,t) = action_probs(actions(t),t);
        
        % get outcomes
        if sim == 1%.toggle == 1
            outcomes(t) = find(rand < cumsum(params.BlockProbs(:, actions(t))),1);%cumsum(sim.true_probs(:,actions(t))),1);
            % outcome ==1 : win, outcome ==2 : neutral, outcome == 3: loss
            outcome_vector(outcomes(t),t) = 1;
        else
            outcomes(t) = rewards(t);
            outcome_vector(rewards(t),t) = 1;
        end
        
        % Get surprise
        F(:, t) = -log(A{t}'*outcome_vector(:,t));
        action_probs_post = spm_softmax(params.alpha*-G(:,t) - F(:,t));
        G_error(t) = (action_probs_post - action_probs(:,t))'*G(:,t);
        
        % learning
        if params.learning_split
            if outcomes(t) == 1
                eta = params.eta_win;
            elseif outcomes(t) == 2
                eta = params.eta_neutral;
            elseif outcomes(t) == 3
                eta = params.eta_loss;
            end
        else
            eta = params.eta;
        end

        % forgetting part
        if params.forgetting_split_matrix
            if outcomes(t) == 1
                omega = params.omega_win;
            elseif outcomes(t) == 2
                omega = params.omega_neutral;
            elseif outcomes(t) == 3
                omega = params.omega_loss;
            end
            a{t+1} = (a{t} - a_0)*(1-omega)+ a_0;

        elseif params.forgetting_split_row
            omega = [1-params.omega_win; 1-params.omega_neutral; 1-params.omega_loss];
            a{t+1} = (a{t} - a_0).* repmat(omega, 1, 3)+ a_0;
        else
            omega = params.omega;
            a{t+1} = (a{t} - a_0)*(1-omega)+ a_0;
        end 
        
      % learning part
        a{t+1} = a{t+1}+ eta*(B_pi(:,actions(t))*outcome_vector(:,t)')';
        
    end
    
end

% Store model variables for export
model_output.choices = actions;
model_output.outcomes = outcomes;
model_output.outcome_vector = outcome_vector;
model_output.action_probabilities = action_probs;
model_output.chosen_action_probabilities = chosen_action_probs;
model_output.learned_reward_probabilities = a;
model_output.params = params;
model_output.EFE = G;
model_output.epistemic_value = epistemic_value;
model_output.pragmatic_value = pragmatic_value;
model_output.G_error = G_error;

end

function A  = spm_norm(A)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
A(isnan(A)) = 1/size(A,1);
end