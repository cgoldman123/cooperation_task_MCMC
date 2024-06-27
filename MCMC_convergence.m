function MCMC_convergence()
    load('L:\rsmith\lab-members\cgoldman\Wellbeing\cooperation_task\coop_MCMC_model_output\coop_MCMC_fits_1\5a348ea750833c0001ee9550_cooperation_task_MCMC_samples');

    % Define the parameters
   % Define the parameters
    parameters = {'pa', 'eta', 'cr', 'cl', 'alpha', 'omega', 'deviance'};
    iterations = 4;
    samples_per_iteration = 1000;

    % Define colors for the iterations
    colors = {'b', 'g', 'r', 'c'};

    % Plotting the samples for each parameter
    for idx = 1:length(parameters)
        param = parameters{idx};
        figure;
        hold on;
        for i = 1:iterations
            plot(samples.(param)(i, :), 'Color', colors{i}, 'DisplayName', ['Iteration ' num2str(i)]);
        end
        hold off;
        title(['Samples for ' param]);
        legend('show');
        xlabel('Sample Index');
        ylabel('Value');
    end

end