function  S0 =  construct_S0( idx, h, ni)
% initial a graph in which the weight of nearest
% neighborhood is one.
S0 = zeros(size(idx,1), ni);
    for j = 1:size(idx,1)
        for k = 1:h
            if k+1 <= size(idx,2)  % Check bounds
                S0(j, idx(j,k+1)) = 1/h;
            end
        end
    end