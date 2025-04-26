% Load COIL20 dataset
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat', 'AR';
data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat', 'COIL20';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat', 'MSRA25';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat', 'YaleB';


load(data_path);

% Take only first 5 classes, 10 samples per class
X = X';
Y = Y(:);
mask = Y <= 5;  % First 5 classes
X = X(:, mask);
Y = Y(mask);
X = X(:, 1:7:end);  % Take every 7th sample
Y = Y(1:7:end);

% Parameters
pca_dim = 15;
reduced_dim = 10;
h1 = 2;
h2 = 3;
r = 2;
alpha = 0.1;
maxiter = 5;

fprintf('MATLAB Results:\n');
fprintf('%s\n', repmat('-', 1, 50));
fprintf('Input X shape: %dx%d\n', size(X));
fprintf('Input Y shape: %dx%d\n', size(Y));
fprintf('Y unique values: '); fprintf('%d ', unique(Y)); fprintf('\n');
fprintf('Samples per class:\n');
tabulate(Y);

% PCA
meanX = mean(X, 2);
X_centered = X - meanX;
[U, ~, ~] = svd(X_centered, 'econ');
X_pca = U(:, 1:pca_dim)' * X_centered;

fprintf('\nAfter PCA:\n');
fprintf('Mean X first 3 values: '); fprintf('%.6f ', meanX(1:3)); fprintf('\n');
fprintf('X_pca shape: %dx%d\n', size(X_pca));
fprintf('X_pca first 3x3 values:\n');
disp(X_pca(1:3,1:3));

% Run ALLDA
[OBJ, W_allma, S1] = ALLDA(X_pca, Y, reduced_dim, h1, r, 1e-5);
Z_train_allma = W_allma' * X_pca;

fprintf('\nALLDA results:\n');
fprintf('W_allma shape: %dx%d\n', size(W_allma));
fprintf('W_allma first 3x3 values:\n');
disp(W_allma(1:3,1:3));
fprintf('Z_train_allma shape: %dx%d\n', size(Z_train_allma));
fprintf('Z_train_allma first 3x3 values:\n');
disp(Z_train_allma(1:3,1:3));

% Run ALLDA_semi
[W_semi, p, S, Obj] = ALLDA_semi(X_pca, Y, X_pca, h1, h2, reduced_dim, alpha, maxiter);
Z_train_semi = W_semi' * X_pca;

fprintf('\nALLDA_semi results:\n');
fprintf('W_semi shape: %dx%d\n', size(W_semi));
fprintf('W_semi first 3x3 values:\n');
disp(W_semi(1:3,1:3));
fprintf('Z_train_semi shape: %dx%d\n', size(Z_train_semi));
fprintf('Z_train_semi first 3x3 values:\n');
disp(Z_train_semi(1:3,1:3));


% Load Python results for comparison 
py_results = load('coil20_results.mat');
fprintf('\nComparison with Python results:\n');
fprintf('X_pca max difference: %.10f\n', max(abs(X_pca(:) - py_results.X_pca(:))));
fprintf('W_allma max difference: %.10f\n', max(abs(W_allma(:) - py_results.W_allma(:))));
fprintf('Z_train_allma max difference: %.10f\n', max(abs(Z_train_allma(:) - py_results.Z_train_allma(:))));
fprintf('W_semi max difference: %.10f\n', max(abs(W_semi(:) - py_results.W_semi(:))));
fprintf('Z_train_semi max difference: %.10f\n', max(abs(Z_train_semi(:) - py_results.Z_train_semi(:))));