%% Quick test: verify mntimes behavior
clc; clear;

% Setup paths
script_dir = 'E:\DATA\vscode\FSI_by_FEM_and_UVLM\single_sheet';
cd(script_dir);
addpath ./cores/ToolBoxes/mntimes;
addpath ./cores/ToolBoxes/Plate_Mesh;
addpath ./cores;
addpath ./cores/functions;
addpath ./cores/functions/structure;
addpath ./cores/functions/fluid;
addpath ./cores/solver;
addpath ./cores/solver/structure;
addpath ./cores/solver/fluid;

% Set minimal params
Nx = 3; Ny = 2;
Length = 1.0; Width = 1.0;
thick = 1e-3;
mu_m = 1.0; nu = 0.3; Ua = 25;
eta_m = mu_m/Ua^2;
Aa = Width*thick;
Ia = Width*thick^3/12;
zeta_m = Aa/Ia*eta_m*Length^2;
n_LW = 1.0;
N_gauss = 3;
theta_a = 0;
N_element = Nx*Ny;
x_vec = ((0:Nx)/Nx).^n_LW*Length;
y_vec = (0:Ny)/Ny*Width;
Dp_mat = 1/(1-nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

flag_output = 0;

generate_shape_function;

[coordinates, nodes] = MeshRectanglularPlate_ununiform(x_vec, y_vec);

% Element sizes
dL_vec = zeros(1,N_element);
dW_vec = dL_vec;
for ii = 1:N_element
    x_2 = coordinates(nodes(ii,2),1);
    x_1 = coordinates(nodes(ii,1),1);
    dL_vec(ii) = x_2 - x_1;
    y_4 = coordinates(nodes(ii,4),2);
    y_1 = coordinates(nodes(ii,1),2);
    dW_vec(ii) = y_4 - y_1;
end

N_q = size(Sc_mat(0,0,0,0), 2);
N_qi = N_q/4;

N_q_all = size(coordinates,1)*N_qi;

I_vec = [];
J_vec = [];
for ii = 1:N_element
    i_vec = repmat((N_qi*(nodes(ii,:)-1)+1).', [1 N_qi]).' + repmat(0:N_qi-1, [length(nodes(ii,:)) 1]).';
    i_vec_v{ii} = i_vec(:).'; %#ok<SAGROW>
    I_vec = [I_vec repmat(i_vec_v{ii}, [1 N_q])]; %#ok<AGROW>
    J_vec = [J_vec kron(i_vec_v{ii}, ones(1,N_q))]; %#ok<AGROW>
end

% Test Gauss quadrature
[p_vec, w_vec] = Gauss(N_gauss);

% Test mntimes with actual shape function data
dL = dL_vec(1);
dW = dW_vec(1);

% Build dx_Sc_mat_v for element 1
dx_Sc_mat_v_test = zeros(3, N_q, length(p_vec), length(p_vec));
i_xi_a = 1;
for xi_a = p_vec
    i_eta_a = 1;
    for eta_a = p_vec
        x_i = dL*(xi_a+1)/2;
        y_i = dW*(eta_a+1)/2;
        dx_Sc_mat_v_test(:,:,i_xi_a,i_eta_a) = dx_Sc_mat(x_i, y_i, dL, dW);
        i_eta_a = i_eta_a+1;
    end
    i_xi_a = i_xi_a+1;
end

fprintf('dx_Sc_mat_v_test size: [%s]\n', num2str(size(dx_Sc_mat_v_test)));

% Permute as in generate_matrices line 76
A_test = permute(dx_Sc_mat_v_test, [2 1 3 4]);
B_test = dx_Sc_mat_v_test;

fprintf('A_test (permuted) size: [%s]\n', num2str(size(A_test)));
fprintf('B_test size: [%s]\n', num2str(size(B_test)));

% Call mntimes
C_test = mntimes(A_test, B_test, 1, 2);
fprintf('C_test (mntimes result) size: [%s]\n', num2str(size(C_test)));

% Expected: [N_q x N_q x p_vec x p_vec]
fprintf('Expected: [%d x %d x %d x %d]\n', N_q, N_q, length(p_vec), length(p_vec));

% Also test with MATLAB pagemtimes if available
try
    C_ref = pagemtimes(permute(A_test, [1 2 4 3]), B_test);
    fprintf('pagemtimes result size: [%s]\n', num2str(size(C_ref)));
catch
    fprintf('pagemtimes not available or dimension mismatch\n');
end

fprintf('\nTest complete.\n');
