function C = mntimes2(A, B)
% Batched matrix multiplication using pagemtimes
% Drop-in for mntimes2 from mntimes toolbox
C = pagemtimes(A, B);
end
