function S = setsparse(S, I, J, V, fun)
% Sparse sub-assignment with accumulation
% Drop-in for Bruno Luong's sparse-sub-access toolbox
if nargin < 5, fun = @plus; end
tmp = sparse(I(:), J(:), V(:), size(S,1), size(S,2));
S = S + tmp;
end
