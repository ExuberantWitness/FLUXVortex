function dump_step1_direct
% Bypass new_X_func_FAST — solve Newmark stage 0 system DIRECTLY in MATLAB.
% Compare with my hand-derived A1 X = A2 X_n + dt[0; F] formula.

add_pathes;
param_setting;
d_t = 2e-3;            % match Python dt_struct=2e-4 s
var_param.d_t = d_t;   % CRITICAL: var_param caches d_t at param_setting time

generate_shape_function;
generate_elements;
generate_matrices;
generate_Qf_time_mat;

time_m = 0:d_t:End_Time;
initial_values;

q_vec = h_X_vec(1:N_q_all, 1);
dt_q_vec = 0 * q_vec;

% Force theta_a=1 to compute K_mem only here (skip Qd_k populating)
theta_a = 0;
flag_output = 1;
generate_stiff_matrices;

% BC indices
i_r  = reshape((repmat((N_qi*(node_r_0  - 1) + 1).', [1 3]) + repmat(0:2, [length(node_r_0)  1])).', 1, []);
i_dx_bc = reshape((repmat((N_qi*(node_dxr_0 - 1) + 4).', [1 3]) + repmat(0:2, [length(node_dxr_0) 1])).', 1, []);
i_dy_bc = reshape((repmat((N_qi*(node_dyr_0 - 1) + 7).', [1 3]) + repmat(0:2, [length(node_dyr_0) 1])).', 1, []);
i_vec_bc = [i_r i_dx_bc i_dy_bc];
free = setdiff(1:N_q_all, i_vec_bc);
nf = length(free);

% Build M, K, F at free DOFs
M_ff = M_global(free, free);
K_ff = dq_Qe_global(free, free);
Qf_pulse_full = Qf_time_global * q_in_norm(d_t);
F_free = Qf_pulse_full(free);

% Build Newmark stage 0 system (alpha=0.5, c_damp=2, Qd=0 since theta_a=0)
alpha_v = 0.5;
C_damp = 2;

I_sp = speye(nf);
O_sp = sparse(nf, nf);
D_bot_left = C_damp * d_t / 2 * K_ff;
D_mat = [I_sp, O_sp; D_bot_left, M_ff];
X2_mat = [O_sp, I_sp; O_sp, O_sp];
A1 = D_mat - alpha_v * d_t * X2_mat;
A2 = D_mat + (1 - alpha_v) * d_t * X2_mat;

q_n_free = q_vec(free);
dq_n_free = dt_q_vec(free);
X_n_free = [q_n_free; dq_n_free];
A2Xn = A2 * X_n_free;
RHS = A2Xn + d_t * [zeros(nf, 1); F_free];
X_p1_free = A1 \ RHS;

q_p1_free = X_p1_free(1:nf);
dq_p1_free = X_p1_free(nf+1:end);

q_p1 = q_vec;
dq_p1 = dt_q_vec;
q_p1(free) = q_p1_free;
dq_p1(free) = dq_p1_free;

% Trap rule check at tip
trail_node = Nx*(Ny+1) + floor(Ny/2) + 1;
z_dof = (trail_node-1)*N_qi + 3;

fprintf('\n=== Direct MATLAB Newmark stage 0 (bypassing new_X_func_FAST) ===\n');
fprintf('  d_t = %.4e\n', d_t);
fprintf('  tip z (q_p1)   = %+.5e\n', q_p1(z_dof));
fprintf('  tip z (q_ref)  = %+.5e\n', q_vec(z_dof));
fprintf('  δq_z           = %+.5e\n', q_p1(z_dof) - q_vec(z_dof));
fprintf('  tip dz (dq_p1) = %+.5e\n', dq_p1(z_dof));
fprintf('  trap check: 0.5*dt*dq_p1 = %+.5e\n', 0.5 * d_t * dq_p1(z_dof));
fprintf('  ratio (δq / trap_expected) = %.4f\n', (q_p1(z_dof) - q_vec(z_dof)) / (0.5*d_t*dq_p1(z_dof)));

% Now run new_X_func_FAST and compare
X_vec = [q_vec; dt_q_vec];
m_global_struct.M_global = M_global;
qf_global_struct.Qf_global = Qf_global + Qf_pulse_full;
dq_qe_global_struct.dq_Qe_global = dq_Qe_global;
qe_global_struct.Qe_global = Qe_global + Qk_global;
qd_global_struct.Qd_global = Qd_global + Qd_theta_global + 0*J_global_2;

[X_vec_p, ~] = new_X_func_FAST(X_vec, m_global_struct, qf_global_struct, ...
    dq_qe_global_struct, qe_global_struct, qd_global_struct, var_param, 0, []);
q_p1_via_func = X_vec_p(1:N_q_all);
dq_p1_via_func = X_vec_p(N_q_all+1:end);

fprintf('\n=== Via new_X_func_FAST (same M, K, F, dq_n, q_n) ===\n');
fprintf('  tip z (q_p1)   = %+.5e\n', q_p1_via_func(z_dof));
fprintf('  tip dz (dq_p1) = %+.5e\n', dq_p1_via_func(z_dof));
fprintf('  ratio (δq / 0.5*dt*dq_p1) = %.4f\n', q_p1_via_func(z_dof) / (0.5*d_t*dq_p1_via_func(z_dof)));

fprintf('\n=== Diff: direct vs new_X_func_FAST ===\n');
fprintf('  |q_p1_direct - q_p1_func|_max = %.4e\n', max(abs(q_p1 - q_p1_via_func)));
fprintf('  tip z direct = %+.5e, via func = %+.5e, diff = %+.3e\n', ...
    q_p1(z_dof), q_p1_via_func(z_dof), q_p1(z_dof) - q_p1_via_func(z_dof));

% Inspect dx_rz at i=1 (chord position 1, second column from LE)
fprintf('\n=== dx_rz at i=1 (first interior chord column), j=5 (mid-span) ===\n');
% Node index for (i=1, j=5) in MATLAB i-outer/j-inner: k = i*(Ny+1) + j = 1*11 + 5 = 16 (1-idx)
i1_j5_node = 1 * (Ny+1) + 5 + 1;  % 1-indexed
dxrz_dof_i1_j5 = (i1_j5_node-1)*N_qi + 6;  % DOF 6 = dx_rz
fprintf('  node (i=1,j=5), 1-indexed = %d, dx_rz DOF = %d\n', i1_j5_node, dxrz_dof_i1_j5);
fprintf('  dx_rz (q_p1 direct)       = %+.5e\n', q_p1(dxrz_dof_i1_j5));
fprintf('  dx_rz (q_p1 via_func)     = %+.5e\n', q_p1_via_func(dxrz_dof_i1_j5));

% Also tip dx_rz
fprintf('\n=== dx_rz at tip (trailing edge, j=5) ===\n');
fprintf('  dx_rz (q_p1 direct)   = %+.5e\n', q_p1(z_dof+3));
fprintf('  dx_rz (q_p1 via_func) = %+.5e\n', q_p1_via_func(z_dof+3));

% Save for Python comparison
out_path2 = './fixtures/step1_pure_with_slopes.mat';
q_p1_pure = q_p1;
dq_p1_pure = dq_p1;
save(out_path2, 'q_p1_pure', 'dq_p1_pure', 'q_vec', 'dt_q_vec', ...
    'Nx', 'Ny', 'N_qi', 'N_q_all', 'd_t', '-v7');
fprintf('\nSaved %s\n', out_path2);
end
