% Running ALLDA and ALLDA_semi on different datasets
% and evaluating their performance using 1-NN classifier.
% with different label parcentages.

%% 1. Load data
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat', 'AR';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat', 'COIL20';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat', 'MSRA25';
data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat', 'YaleB';


load(data_path);

X = X';
%fprintf('X size: %d x %d\n', size(X,1), size(X,2));
%fprintf('Y size: %d x %d\n', size(Y,1), size(Y,2));
n_class = length(unique(Y));
n = size(X, 2);
n_run = 9;

%% Parameters
pca_dim = 95;
reduced_dim = 40;
h1 = 2;
h2 = 10;
r = 2;
alpha = 0.1;
maxiter = 10;
%label_percents = [10, 20, 30, 40, 50];
label_percents = [ 20, 30, 40, 50, 60]; %For AR dataset only
n_perc = length(label_percents);

acc_allma = zeros(n_perc, n_run);        % For supervised ALLDA
acc_allma_semi = zeros(n_perc, n_run);   % For semi-supervised ALLDA_semi

for p = 1:n_perc
    percent = label_percents(p);
    fprintf('\nRunning for %d%% labeled data...\n', percent);

    for run = 1:n_run
        fprintf('Run %d/%d...\n', run, n_run);

        %% 2. PCA
        meanX = mean(X,2);
        X_centered = X - meanX;
        [U, ~, ~] = svd(X_centered, 'econ');
        X_pca = U(:, 1:pca_dim)' * X_centered;

        %% 3. Balanced Train/Test Split with Percentage Labeling
        rng(run);  % for reproducibility

        % Find the number of samples in each class
        class_counts = histcounts(Y, n_class);
        min_samples = min(class_counts);  % smallest class size

        % Calculate number of labeled samples based on percentage
        n_labeled = round(min_samples * (percent/100));
        n_labeled = max(1, n_labeled);  % Ensure at least one labeled sample
        n_unlabeled = min_samples - n_labeled;

        % Error check for unlabeled samples
        if n_unlabeled < 1
            error('Not enough samples for unlabeled set with current percentage');
        end

        train_idx = [];
        test_idx = [];

        for i = 1:n_class
            idx = find(Y == i);
            idx = idx(randperm(length(idx)));  % shuffle

            % Take the determined number of samples
            if length(idx) >= min_samples
                train_idx = [train_idx, idx(1:n_labeled)];
                test_idx = [test_idx, idx(n_labeled+1:n_labeled+n_unlabeled)];
            else
                % For classes smaller than min_samples (shouldn't happen due to earlier check)
                warning('Class %d has fewer samples (%d) than min_samples (%d)', i, length(idx), min_samples);
                continue;
            end
        end

        X_train = X_pca(:, train_idx);
        Y_train = Y(train_idx(:));
        X_test = X_pca(:, test_idx);
        Y_test = Y(test_idx(:));

        %% 4. Run ALLDA
        [~, W_allma, ~] = ALLDA(X_train, Y_train, reduced_dim, h1, r, 1e-5);
        Z_train_allma = W_allma' * X_train;
        Z_test_allma = W_allma' * X_test;

        %% 5. Run ALLDA_semi
        [W_semi, ~, ~, ~] = ALLDA_semi(X_train, Y_train, [X_train, X_test], h1, h2, reduced_dim, alpha, maxiter);
        Z_train_semi = W_semi' * X_train;
        Z_test_semi = W_semi' * X_test;

        %% 6. Evaluate using 1-NN
        mdl1 = fitcknn(Z_train_allma', Y_train', 'NumNeighbors', 1);
        pred1 = predict(mdl1, Z_test_allma');
        pred1 = pred1(:);
        Y_test = Y_test(:);
        acc_allma(p, run) = sum(pred1 == Y_test) / length(Y_test);

        mdl2 = fitcknn(Z_train_semi', Y_train', 'NumNeighbors', 1);
        pred2 = predict(mdl2, Z_test_semi');
        pred2 = pred2(:);
        acc_allma_semi(p, run) = sum(pred2 == Y_test) / length(Y_test);
    end
end

%% Print Final Results
fprintf('\n---- Final Results ----\n');
for p = 1:n_perc
    fprintf('\nLabeling: %d%%\n', label_percents(p));
    fprintf('ALLDA mean acc:       %.4f ± %.4f\n', mean(acc_allma(p,:)), std(acc_allma(p,:)));
    fprintf('ALLDA_semi mean acc:  %.4f ± %.4f\n', mean(acc_allma_semi(p,:)), std(acc_allma_semi(p,:)));
end
