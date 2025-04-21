
%% 1. Load data
data_path = 'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/AR.mat', 'AR';
load(data_path);  % assumes variables 'fea' and 'gnd'

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
    rng(run);  % for reproducibility
    train_idx = [];
    test_idx = [];
    for i = 1:n_class
        idx = find(Y == i);
        idx = idx(randperm(length(idx)));
        split = floor(length(idx)/2);
        if split == 0 || split >= length(idx)
            continue;  
        end
        train_idx = [train_idx, idx(1:split)];
        test_idx = [test_idx, idx(split+1:end)];
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
