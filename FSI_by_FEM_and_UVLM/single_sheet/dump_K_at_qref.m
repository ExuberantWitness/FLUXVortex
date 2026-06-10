function dump_K_at_qref
% Dump MATLAB ANCF tangent stiffness K_global at the reference (flat plate) state
% for entry-by-entry comparison with the Python implementation.
%
% Output: ./fixtures/K_at_qref.mat with variables:
%   - q_vec, dt_q_vec        (initial state at flat plate)
%   - coordinates, nodes
%   - dL_vec, dW_vec
%   - M_global               (mass matrix, sparse)
%   - Qe_global              (membrane internal force, dense)
%   - Qk_global              (bending internal force, dense)
%   - dq_Qe_global           (membrane tangent K_mem, sparse)
%   - dq_Qk_global           (bending  tangent K_bend, sparse)
%   - i_vec_v                (per-element 36-DOF index list)
%   - i_vec_bc               (Dirichlet DOF indices)
%   - Nx, Ny, N_q, N_q_all, N_element

add_pathes;
param_setting;

generate_shape_function;
generate_elements;
generate_matrices;

% --- Initial state: flat plate with unit slopes (matches initial_values.m) ---
q_vec    = zeros(N_q_all, 1);
dt_q_vec = zeros(N_q_all, 1);

idx_rxy   = reshape([1:N_qi:N_q_all; 2:N_qi:N_q_all], 1, []);
q_vec(idx_rxy)  = reshape(coordinates.', [], 1);

idx_dx_r  = reshape([4:N_qi:N_q_all; 5:N_qi:N_q_all; 6:N_qi:N_q_all], 1, []);
q_vec(idx_dx_r) = repmat([1 0 0].', [N_node 1]);

idx_dy_r  = reshape([7:N_qi:N_q_all; 8:N_qi:N_q_all; 9:N_qi:N_q_all], 1, []);
q_vec(idx_dy_r) = repmat([0 1 0].', [N_node 1]);

% --- Force theta_a=1 so generate_stiff_matrices populates Qd_*_mat_i ---
theta_a_save = theta_a;
theta_a = 1;
flag_output = 1;
generate_stiff_matrices;
theta_a = theta_a_save;

% --- Assemble membrane K (zeta_m * Qd_eps_mat_i) ---
dq_Qe_mem_global = sparse(N_q_all, N_q_all);
for ii = 1:N_element
    i_vec = i_vec_v{ii};
    dq_Qe_mem_global(i_vec, i_vec) = dq_Qe_mem_global(i_vec, i_vec) ...
        + squeeze(zeta_m * Qd_eps_mat_i(:, :, ii));
end

% --- Assemble bending K (eta_m * Qd_k_mat_i) ---
dq_Qk_global = sparse(N_q_all, N_q_all);
for ii = 1:N_element
    i_vec = i_vec_v{ii};
    dq_Qk_global(i_vec, i_vec) = dq_Qk_global(i_vec, i_vec) ...
        + squeeze(eta_m * Qd_k_mat_i(:, :, ii));
end

% --- BC indices (Dirichlet on leading edge: r, dx_r, dy_r) ---
i_r  = reshape((repmat((N_qi*(node_r_0  - 1) + 1).', [1 3]) + repmat(0:2, [length(node_r_0)  1])).', 1, []);
i_dx = reshape((repmat((N_qi*(node_dxr_0 - 1) + 4).', [1 3]) + repmat(0:2, [length(node_dxr_0) 1])).', 1, []);
i_dy = reshape((repmat((N_qi*(node_dyr_0 - 1) + 7).', [1 3]) + repmat(0:2, [length(node_dyr_0) 1])).', 1, []);
i_vec_bc = [i_r i_dx i_dy];

if ~exist('./fixtures', 'dir'); mkdir('./fixtures'); end
out_path = './fixtures/K_at_qref.mat';
save(out_path, ...
    'q_vec', 'dt_q_vec', 'coordinates', 'nodes', 'dL_vec', 'dW_vec', ...
    'M_global', 'Qe_global', 'Qk_global', ...
    'dq_Qe_global', 'dq_Qe_mem_global', 'dq_Qk_global', ...
    'Qd_eps_mat_i', 'Qd_k_mat_i', ...
    'i_vec_v', 'i_vec_bc', 'Nx', 'Ny', 'N_q', 'N_q_all', 'N_element', ...
    'mu_m', 'zeta_m', 'eta_m', 'Length', 'Width', 'thick', 'nu', ...
    '-v7');

fprintf('\n=== Dumped %s ===\n', out_path);
fprintf('  N_node=%d  N_element=%d  N_q_all=%d\n', N_node, N_element, N_q_all);
fprintf('  |dq_Qe_global|_F   = %.6e  (full K from generate_stiff_matrices, theta_a=1)\n', norm(full(dq_Qe_global), 'fro'));
fprintf('  |dq_Qe_mem_global|_F = %.6e  (assembled zeta_m * Qd_eps_mat_i)\n', norm(full(dq_Qe_mem_global), 'fro'));
fprintf('  |dq_Qk_global|_F   = %.6e  (assembled eta_m * Qd_k_mat_i)\n', norm(full(dq_Qk_global), 'fro'));
fprintf('  |M_global|_F       = %.6e\n', norm(full(M_global), 'fro'));
fprintf('  |Qe_global|        = %.6e\n', norm(Qe_global));
fprintf('  |Qk_global|        = %.6e\n', norm(Qk_global));
end
