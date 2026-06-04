function C = mntimes(A, B, dimA, dimB)
% Batched matrix multiplication using pagemtimes
% Drop-in for Darin Koblick's mntimes toolbox
if nargin < 3, dimA = 1; end
if nargin < 4, dimB = 2; end
if ndims(A) == 2
    C = A * B;
else
    C = pagemtimes(A, B);
end
end
