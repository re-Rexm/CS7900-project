%% 1. Load data
data_path = 'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/AR.mat', 'AR';
%data_path = 'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/COIL20.mat', 'COIL20';
%data_path = 'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/MSRA25.mat', 'MSRA25';
%data_path = 'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/YaleB.mat', 'YaleB';
load(data_path);  

X = X';
n_class = length(unique(Y));
n = size(X, 2);
n_run = 10;

%% Parameters
pca_dim = 95;
reduced_dim = 30;
h1 = 2;
h2 = 10;
r = 2;
alpha = 0.1;
maxiter = 10;
label_percents = [10, 20, 30, 40, 50];
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

        %% 3. Train/Test Split
        rng(run);  % for reproducibility
        train_idx = [];
        test_idx = [];
        for i = 1:n_class
            idx = find(Y == i);
            if length(idx) < 2  % Need at least 2 samples per class
                continue;
            end
            idx = idx(randperm(length(idx)));
            n_labeled = round(length(idx) * (percent/100));
            n_labeled = max(1, n_labeled);  % Ensure at least one labeled sample

            if length(idx) - n_labeled < 1
                continue;  % Skip this class if not enough for test
            end
            train_idx = [train_idx, idx(1:n_labeled)];
            test_idx = [test_idx, idx(n_labeled+1:end)];
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
