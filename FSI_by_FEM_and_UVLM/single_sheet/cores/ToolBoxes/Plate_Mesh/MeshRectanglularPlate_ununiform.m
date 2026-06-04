function [coordinates, nodes] = MeshRectanglularPlate_ununiform(x_vec, y_vec)
% Generate structured rectangular plate mesh with non-uniform spacing
% Drop-in replacement for KSSV's Plate_Mesh toolbox
%
% Node ordering: x varies in outer loop, y in inner loop
%   Node (i,j) -> linear index: (i-1)*(Ny+1) + j
%   So nodes 1:(Ny+1) are at x=x_vec(1) (leading edge)
%   Adjacent nodes in x: offset by (Ny+1)

Nx = length(x_vec) - 1;
Ny = length(y_vec) - 1;

% Build coordinates: x outer loop, y inner loop
% So first (Ny+1) nodes are at x=x_vec(1)
coordinates = zeros((Nx+1)*(Ny+1), 2);
for i = 1:(Nx+1)
    for j = 1:(Ny+1)
        idx = (i-1)*(Ny+1) + j;
        coordinates(idx, :) = [x_vec(i), y_vec(j)];
    end
end

% Build element connectivity
% Element (i,j): bottom-left at node (i-1)*(Ny+1)+j, counter-clockwise
nodes = zeros(Nx*Ny, 4);
elem = 0;
for i = 1:Nx
    for j = 1:Ny
        elem = elem + 1;
        n1 = (i-1)*(Ny+1) + j;        % bottom-left
        n2 = i*(Ny+1) + j;             % bottom-right (next x column)
        n3 = i*(Ny+1) + j + 1;         % top-right
        n4 = (i-1)*(Ny+1) + j + 1;     % top-left
        nodes(elem, :) = [n1, n2, n3, n4];
    end
end
end
