% Running ALLDA and ALLDA_semi on different datasets
% and evaluating their performance using 1-NN classifier.


%% 1. Load data
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat', 'AR';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat', 'COIL20';
data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat', 'MSRA25';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat', 'YaleB';
load(data_path);  

X = X';
%fprintf('X size: %d x %d\n', size(X,1), size(X,2));
%fprintf('Y size: %d x %d\n', size(Y,1), size(Y,2));
n_class = length(unique(Y));
n = size(X, 2);
n_run = 10;

acc_allma = zeros(1, n_run);
acc_allma_semi = zeros(1, n_run);

%% Parameters
pca_dim = 95;
reduced_dim = 40;
h1 = 2;
h2 = 10;
r = 2;
alpha = 0.1;
maxiter = 10;

for run = 1:n_run
    fprintf('Run %d/%d...\n', run, n_run);
    
    %% 2. PCA
    meanX = mean(X,2);
    X_centered = X - meanX;
    [U, ~, ~] = svd(X_centered, 'econ');
    X_pca = U(:, 1:pca_dim)' * X_centered;
    
    %% 3. 50/50 train test split
    rng(run);  % for reproducibility seed = numofrun

    % Find the number of samples in each class
    class_counts = histcounts(Y, n_class);
    min_samples = min(class_counts);  % smallest class size
    %fprintf('Min sample: %d \n', min_samples);

    % Determine how many samples to take from each class for train/test
    train_per_class = floor(min_samples/2);
    test_per_class = min_samples - train_per_class;

    train_idx = [];
    test_idx = [];
    for i = 1:n_class
        idx = find(Y == i);
        idx = idx(randperm(length(idx)));

        % Take the determined number of samples
        train_idx = [train_idx, idx(1:train_per_class)];
        test_idx = [test_idx, idx(train_per_class+1:train_per_class+test_per_class)];
    end
    
    
    X_train = X_pca(:, train_idx);
    Y_train = Y(train_idx(:));
    %fprintf('X size: %d x %d\n', size(X_train,1), size(X_train,2));
    %fprintf('Y size: %d x %d\n', size(Y_train,1), size(Y_train,2));
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
    %fprintf('Pred1 size: %d x %d\n', size(pred1,1), size(pred1,2));
    %fprintf('Y size: %d x %d\n', size(Y_test,1), size(Y_test,2));
    acc_allma(run) = sum(pred1 == Y_test) / length(Y_test);
    
    mdl2 = fitcknn(Z_train_semi', Y_train', 'NumNeighbors', 1);
    pred2 = predict(mdl2, Z_test_semi');
    acc_allma_semi(run) = sum(pred2 == Y_test) / length(Y_test);
end

%% Print Results
fprintf('\nALLDA mean acc: %.4f ± %.4f\n', mean(acc_allma), std(acc_allma));
fprintf('ALLDA_semi mean acc: %.4f ± %.4f\n', mean(acc_allma_semi), std(acc_allma_semi));
