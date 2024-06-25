function merged_file = COP_merge_files(fit_list)
    merged_file = [];
    for subject = fit_list
        root = 'L:/';
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

            % continue to next file if no MAIN_START
            if ~any(cellfun(@(x) isequal(x, 'MAIN_START'), subdat.trial_type))
                continue;
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

            % parse observations and actions
            sub.o = clean_subdat.result(clean_subdat.event_type == 5);
            sub.u = clean_subdat.response(clean_subdat.event_type == 5);
            
            % if they got past "MAIN_START" but don't have the right number
            % of trials, indicate that they have practice effects and advance 
            % them to the next file
            if size(sub.o,1) ~= 480 && size(sub.u,1) ~= 480
                has_practice_effects = true;
                continue
            end
               
            

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
            
            o_all_transposed = o_all';
            u_all_transposed = u_all';
            
            data_struct = struct();
            data_struct.subject_id = repmat({subject}, NB, 1);
            data_struct.block_num = num2cell(1:NB)';
            data_struct.trial_types = trial_types;
            for i = 1:TpB
                data_struct.(['o', num2str(i)]) = o_all_transposed(:, i);
                data_struct.(['u', num2str(i)]) = u_all_transposed(:, i);
            end

            % Convert struct to table
            T = struct2table(data_struct);
            % Create a table
            % Append the table to the merged_file table
            merged_file = [merged_file; T];

        end
    end
end