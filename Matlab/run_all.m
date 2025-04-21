%% Dataset Paths
dataset_paths = {
    'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/COIL20.mat', 'COIL20';
    'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/AR.mat', 'AR';
    'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/MSRA25.mat', 'MSRA25';
    'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/YaleB.mat', 'YaleB';
};

%% Global Params
pca_dim = 95;
reduced_dim = 40;
h = 5;
r = 2;
alpha = 1;
maxiter = 10;
n_run = 10;
label_percents = [10, 20, 30, 40, 50];

for d = 1:size(dataset_paths,1)
    data_path = dataset_paths{d,1};
    dataset_name = dataset_paths{d,2};
    
    fprintf('\n==============================================\n');
    fprintf('Processing dataset: %s\n', dataset_name);
    fprintf('==============================================\n');

    %% Load Data
    load(data_path);  % expects 'fea' and 'gnd' or 'X' and 'Y'
    if exist('fea', 'var') && exist('gnd', 'var')
        X = fea';
        Y = gnd;
    elseif exist('X', 'var') && exist('Y', 'var')
        X = X';
    else
        error('Unrecognized variable names in %s', data_path);
    end

    n_class = length(unique(Y));
    acc_allma = zeros(length(label_percents), n_run);
    acc_allma_semi = zeros(length(label_percents), n_run);

    for p = 1:length(label_percents)
        percent = label_percents(p);
        fprintf('\nRunning for %d%% labeled data...\n', percent);

        for run = 1:n_run
            fprintf('Run %d/%d...\n', run, n_run);

            %% PCA
            meanX = mean(X,2);
            X_centered = X - meanX;
            [U, ~, ~] = svd(X_centered, 'econ');
            X_pca = U(:, 1:pca_dim)' * X_centered;

            %% Split
            rng(run);
            train_idx = [];
            test_idx = [];
            for i = 1:n_class
                idx = find(Y == i);
                idx = idx(randperm(length(idx)));
                n_labeled = round(length(idx) * (percent/100));
                if n_labeled == 0 || n_labeled >= length(idx)
                    continue;
                end
                train_idx = [train_idx, idx(1:n_labeled)];
                test_idx = [test_idx, idx(n_labeled+1:end)];
            end

            X_train = X_pca(:, train_idx);
            Y_train = Y(train_idx(:));
            X_test = X_pca(:, test_idx);
            Y_test = Y(test_idx(:));

            %% ALLDA
            [~, W_allma, ~] = ALLDA(X_train, Y_train, reduced_dim, h, r, 1e-5);
            Z_train_allma = W_allma' * X_train;
            Z_test_allma = W_allma' * X_test;

            %% ALLDA_semi
            [W_semi, ~, ~, ~] = ALLDA_semi(X_train, Y_train, [X_train, X_test], h, h, reduced_dim, alpha, maxiter);
            Z_train_semi = W_semi' * X_train;
            Z_test_semi = W_semi' * X_test;

            %% 1-NN Accuracy
            mdl1 = fitcknn(Z_train_allma', Y_train', 'NumNeighbors', 1);
            pred1 = predict(mdl1, Z_test_allma');
            acc_allma(p, run) = sum(pred1(:) == Y_test(:)) / length(Y_test);

            mdl2 = fitcknn(Z_train_semi', Y_train', 'NumNeighbors', 1);
            pred2 = predict(mdl2, Z_test_semi');
            acc_allma_semi(p, run) = sum(pred2(:) == Y_test(:)) / length(Y_test);
        end
    end

    %% Final Result per Dataset
    fprintf('\n---- %s Results ----\n', dataset_name);
    for p = 1:length(label_percents)
        fprintf('Labeling %2d%% -> ALLDA: %.4f ± %.4f | ALLDA_semi: %.4f ± %.4f\n', ...
            label_percents(p), ...
            mean(acc_allma(p,:)), std(acc_allma(p,:)), ...
            mean(acc_allma_semi(p,:)), std(acc_allma_semi(p,:)));
    end
end
