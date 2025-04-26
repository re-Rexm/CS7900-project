% Running ALLDA and ALLDA_semi on different datasets
% and evaluating their performance using 1-NN classifier.

%% 1. Load data

data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\colon.mat', 'Colon';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\lung_discrete.mat', 'Lung';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\Prostate.mat', 'Prostate';

load(data_path);  
disp(['X size: ', num2str(size(X))]);
disp(['Y size: ', num2str(size(Y))]);
disp(['Unique classes: ', num2str(length(unique(Y)))]);

X = X';
n_class = length(unique(Y));
n = size(X, 2);
n_run = 10;

acc_allma = zeros(1, n_run);
acc_allma_semi = zeros(1, n_run);

%% Parameters
pca_dim = 30;
reduced_dim = 2;
h1 = 1;
h2 = 5;
r = 2;
alpha = 0.01;
maxiter = 10;

for run = 1:n_run
    fprintf('Run %d/%d...\n', run, n_run);
    
    %% 2. PCA
    meanX = mean(X,2);
    X_centered = X - meanX;
    [U, ~, ~] = svd(X_centered, 'econ');
    X_pca = U(:, 1:pca_dim)' * X_centered;
    
    %% 3. 70/30 train test split (no class balancing)
    rng(run);  % for reproducibility
    
    % Shuffle all samples
    idx = randperm(n);
    split_point = round(0.7 * n);  % 70% for training
    
    train_idx = idx(1:split_point);
    test_idx = idx(split_point+1:end);
    
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
    acc_allma(run) = sum(pred1 == Y_test) / length(Y_test);
    
    mdl2 = fitcknn(Z_train_semi', Y_train', 'NumNeighbors', 1);
    pred2 = predict(mdl2, Z_test_semi');
    acc_allma_semi(run) = sum(pred2 == Y_test) / length(Y_test);
end

%% Print Results
fprintf('\nALLDA mean acc: %.4f ± %.4f\n', mean(acc_allma), std(acc_allma));
fprintf('ALLDA_semi mean acc: %.4f ± %.4f\n', mean(acc_allma_semi), std(acc_allma_semi));