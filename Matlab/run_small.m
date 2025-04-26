% Load COIL20 dataset
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat', 'AR';
data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat', 'COIL20';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat', 'MSRA25';
%data_path = 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat', 'YaleB';


load(data_path);

% Convert X to double explicitly after loading
X = double(X');
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

% Enhanced comparison function
function compare_matrices(name, mat1, mat2, num_elements)
    if nargin < 4
        num_elements = 3;  % Default to showing first 3 elements
    end
    
    % Convert inputs to double to ensure compatibility
    mat1 = double(mat1);
    mat2 = double(mat2);
    
    fprintf('\n=== Comparing %s ===\n', name);
    fprintf('Shape: MATLAB %dx%d, Python %dx%d\n', size(mat1), size(mat2));
    
    % Show first few elements of both matrices
    fprintf('First %d elements:\n', num_elements);
    fprintf('MATLAB: ');
    fprintf('%.6f ', mat1(1:min(num_elements, numel(mat1))));
    fprintf('\nPython: ');
    fprintf('%.6f ', mat2(1:min(num_elements, numel(mat2))));
    fprintf('\n');
    
    % Calculate differences
    if isequal(size(mat1), size(mat2))
        abs_diff = abs(mat1 - mat2);
        max_diff = max(abs_diff(:));
        mean_diff = mean(abs_diff(:));
        fprintf('Max difference: %.10f\n', max_diff);
        fprintf('Mean difference: %.10f\n', mean_diff);
        
        % Show location and values of maximum difference
        [max_row, max_col] = find(abs_diff == max_diff, 1);
        if ~isempty(max_row)
            fprintf('Max difference location: [%d, %d]\n', max_row, max_col);
            fprintf('MATLAB value: %.10f\n', mat1(max_row, max_col));
            fprintf('Python value: %.10f\n', mat2(max_row, max_col));
        end
    else
        fprintf('ERROR: Matrix sizes do not match!\n');
    end
end

% Load Python results and perform detailed comparisons
py_results = load('coil20_results.mat');

% Compare initial data
compare_matrices('Input X', X, py_results.X);
compare_matrices('Input Y', Y, py_results.Y);

% Compare PCA steps
compare_matrices('meanX', meanX, py_results.meanX);
compare_matrices('X_centered', X_centered, py_results.X_centered);
compare_matrices('U (PCA components)', U(:,1:pca_dim), py_results.U);
compare_matrices('X_pca', X_pca, py_results.X_pca);

% Compare ALLDA results
compare_matrices('W_allma', W_allma, py_results.W_allma);
compare_matrices('S1', S1, py_results.S1);
compare_matrices('Z_train_allma', Z_train_allma, py_results.Z_train_allma);
compare_matrices('OBJ', OBJ, py_results.OBJ_allda);

% Compare ALLDA_semi results
compare_matrices('W_semi', W_semi, py_results.W_semi);
compare_matrices('S', S, py_results.S);
compare_matrices('Z_train_semi', Z_train_semi, py_results.Z_train_semi);
compare_matrices('Obj', Obj, py_results.Obj_semi);

% If p struct comparison is needed
if isstruct(p) && isfield(py_results, 'p') && isstruct(py_results.p)
    p_fields = fieldnames(p);
    for i = 1:length(p_fields)
        field = p_fields{i};
        if isfield(py_results.p, field)
            compare_matrices(['p.' field], p.(field), py_results.p.(field));
        end
    end
end