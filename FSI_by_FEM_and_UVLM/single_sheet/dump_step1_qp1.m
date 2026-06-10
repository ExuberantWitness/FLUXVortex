function dump_step1_qp1
% Dump MATLAB stage-0 (predictor) q_p1 at step 1, with no fluid forces (only pulse).
% This isolates the structural Newmark predictor — same inputs as Python's diag_step1_pure.

add_pathes;
param_setting;

% Override d_t to match Python's dt_struct=2e-4 s (= 2e-3 non-dim, since L/V=0.1)
d_t = 2e-3;
fprintf('Override: d_t = %.4e (non-dim) → %.4e s physical\n', d_t, d_t*0.1);

generate_shape_function;
generate_elements;
generate_matrices;
generate_Qf_time_mat;

% Initial state — flat plate
time_m = 0:d_t:End_Time;
initial_values;

q_vec = h_X_vec(1:N_q_all, 1);
dt_q_vec = 0 * q_vec;
X_vec = [q_vec; dt_q_vec];

% Force theta_a=1 so generate_stiff_matrices ALSO populates Qd_k_mat_i (bending tangent)
% Then we will build K_total = K_mem + K_bend manually and pass to Newmark
% to test whether MATLAB with full K matches Python.
theta_a = 1;
flag_output = 1;
generate_stiff_matrices;
theta_a = 0;     % restore for Newmark call

Qe_global_n = Qe_global;
Qk_global_n = Qk_global;

% Assemble full K = membrane + bending
dq_Qk_global = sparse(N_q_all, N_q_all);
for ii = 1:N_element
    i_vec = i_vec_v{ii};
    dq_Qk_global(i_vec, i_vec) = dq_Qk_global(i_vec, i_vec) + squeeze(eta_m * Qd_k_mat_i(:, :, ii));
end
dq_Qe_mem_only = dq_Qe_global;
dq_Qe_full     = dq_Qe_global + dq_Qk_global;

% Run Newmark TWICE: once with K_mem only (MATLAB default), once with full K (Python)
dq_Qe_global_n = dq_Qe_mem_only;
Qd_global_n = Qd_global;
Qd_theta_global_n = Qd_theta_global;
J_global_1_n = J_global_1;
J_global_2_n = J_global_2;

% PULSE ONLY at t = d_t (matches Python's diag_step1_pure)
time = d_t;
Qf_pulse = Qf_time_global * q_in_norm(time);

% Newmark inputs — no aero, no added mass, no damping
m_global_struct.M_global = M_global + J_global_1_n;
qf_global_struct.Qf_global = Qf_global + Qf_pulse;     % Qf_global=0 in Yamano
dq_qe_global_struct.dq_Qe_global = dq_Qe_global_n;
qe_global_struct.Qe_global = Qe_global_n + Qk_global_n;
qd_global_struct.Qd_global = Qd_global_n + Qd_theta_global_n + J_global_2_n;

% Stage 0 (predictor) WITH K_mem only (MATLAB default)
[X_vec_p, out1] = new_X_func_FAST(X_vec, m_global_struct, qf_global_struct, ...
    dq_qe_global_struct, qe_global_struct, qd_global_struct, var_param, 0, []);

q_p1 = X_vec_p(1:N_q_all);
dq_p1 = X_vec_p(N_q_all+1:end);

% Also run with FULL K (Python-style) for comparison
dq_qe_global_struct_full.dq_Qe_global = dq_Qe_full;
[X_vec_p_full, out1_full] = new_X_func_FAST(X_vec, m_global_struct, qf_global_struct, ...
    dq_qe_global_struct_full, qe_global_struct, qd_global_struct, var_param, 0, []);
q_p1_full = X_vec_p_full(1:N_q_all);
dq_p1_full = X_vec_p_full(N_q_all+1:end);

% Stage 1 (corrector) — recompute Qk at q_p1
q_vec_save = q_vec;
q_vec = q_p1;
flag_output = 0;
generate_stiff_matrices;
Qk_global_np1 = Qk_global;
q_vec = q_vec_save;

% Apply stage 1 with bending averaged
qe_global_struct.Qe_global = Qe_global_n + (Qk_global_n + Qk_global_np1)/2;
qf_global_struct.Qf_global = Qf_global + Qf_pulse;     % pulse not averaged
new_X_vec = new_X_func_FAST(X_vec, m_global_struct, qf_global_struct, ...
    dq_qe_global_struct, qe_global_struct, qd_global_struct, var_param, 1, out1);

q_new = new_X_vec(1:N_q_all);
dq_new = new_X_vec(N_q_all+1:end);

if ~exist('./fixtures', 'dir'); mkdir('./fixtures'); end
out_path = './fixtures/step1_qp1.mat';
% Also dump M_global (with corrections), and BC indices for direct system solve check
M_for_newmark = M_global + J_global_1_n;     % no Qf_p_mat (=0)
% Recompute i_vec_bc (same as inside new_X_func_FAST)
i_r  = reshape((repmat((N_qi*(node_r_0  - 1) + 1).', [1 3]) + repmat(0:2, [length(node_r_0)  1])).', 1, []);
i_dx_bc = reshape((repmat((N_qi*(node_dxr_0 - 1) + 4).', [1 3]) + repmat(0:2, [length(node_dxr_0) 1])).', 1, []);
i_dy_bc = reshape((repmat((N_qi*(node_dyr_0 - 1) + 7).', [1 3]) + repmat(0:2, [length(node_dyr_0) 1])).', 1, []);
i_vec_bc_full = [i_r i_dx_bc i_dy_bc];

save(out_path, ...
    'q_vec', 'q_p1', 'dq_p1', 'q_new', 'dq_new', ...
    'q_p1_full', 'dq_p1_full', ...
    'Qf_pulse', 'Qe_global_n', 'Qk_global_n', 'Qk_global_np1', ...
    'dq_Qe_mem_only', 'dq_Qe_full', ...
    'M_for_newmark', 'i_vec_bc_full', ...
    'coordinates', 'nodes', 'Nx', 'Ny', 'N_q_all', 'N_qi', 'N_element', ...
    'mu_m', 'zeta_m', 'eta_m', 'Length', 'd_t', 'thick', ...
    '-v7');

% Also report q_p1_full for direct comparison
trail_node = Nx*(Ny+1) + floor(Ny/2) + 1;
z_dof = (trail_node-1)*N_qi + 3;
fprintf('  tip z (K_mem only)    = %+.4e\n', q_p1(z_dof));
fprintf('  tip z (K_mem + K_bend)= %+.4e   ← Python-equivalent\n', q_p1_full(z_dof));

fprintf('\n=== Dumped %s ===\n', out_path);
fprintf('  |q_p1 - q_ref|_max  = %.4e   (stage 0 predictor)\n', max(abs(q_p1 - q_vec)));
fprintf('  |q_new - q_ref|_max = %.4e   (stage 1 corrector)\n', max(abs(q_new - q_vec)));
fprintf('  |Qk_global_n|_F     = %.4e\n', norm(Qk_global_n));
fprintf('  |Qk_global_np1|_F   = %.4e\n', norm(Qk_global_np1));

% Trailing-edge tip values
trail_node = Nx*(Ny+1) + floor(Ny/2) + 1;
z_dof = (trail_node-1)*N_qi + 3;
fprintf('  tip z (q_p1)  = %+.4e   tip z (q_new)  = %+.4e\n', q_p1(z_dof), q_new(z_dof));
fprintf('  tip dxrz (q_p1)= %+.4e   tip dxrz (q_new)= %+.4e\n', q_p1(z_dof+3), q_new(z_dof+3));
end
