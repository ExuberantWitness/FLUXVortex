function dump_Q_at_perturb
% Dump MATLAB Qe, Qk at q_ref + δq where δq is a small z-perturbation.
% Used to compare Python's _internal_forces_separated vs MATLAB at a non-trivial
% (deformed) state — the case that matters for Newmark stage-1 correction.

add_pathes;
param_setting;

generate_shape_function;
generate_elements;
generate_matrices;

% Initial state — flat plate with unit slopes
q_ref    = zeros(N_q_all, 1);
dt_q_vec = zeros(N_q_all, 1);

idx_rxy   = reshape([1:N_qi:N_q_all; 2:N_qi:N_q_all], 1, []);
q_ref(idx_rxy)  = reshape(coordinates.', [], 1);

idx_dx_r  = reshape([4:N_qi:N_q_all; 5:N_qi:N_q_all; 6:N_qi:N_q_all], 1, []);
q_ref(idx_dx_r) = repmat([1 0 0].', [N_node 1]);

idx_dy_r  = reshape([7:N_qi:N_q_all; 8:N_qi:N_q_all; 9:N_qi:N_q_all], 1, []);
q_ref(idx_dy_r) = repmat([0 1 0].', [N_node 1]);

% Perturbation: parabolic z deflection w(x) = eps * (x/L)^2
% applied to every node's r_z DOF + corresponding dx_r_z slope DOF
eps = 1e-6;
q_perturb = q_ref;
for nn = 1:N_node
    x = coordinates(nn, 1);
    rz_dof  = (nn - 1) * N_qi + 3;
    dxrz    = (nn - 1) * N_qi + 6;
    q_perturb(rz_dof)  = eps * x^2;
    q_perturb(dxrz)    = 2 * eps * x;
end

% Compute Qe, Qk at perturbation
q_vec = q_perturb;
theta_a_save = theta_a;
theta_a = 1;
flag_output = 1;
generate_stiff_matrices;
theta_a = theta_a_save;

if ~exist('./fixtures', 'dir'); mkdir('./fixtures'); end
out_path = './fixtures/Q_at_perturb.mat';
save(out_path, ...
    'q_ref', 'q_perturb', 'eps', 'coordinates', 'nodes', ...
    'Qe_global', 'Qk_global', ...
    'Nx', 'Ny', 'N_q', 'N_q_all', 'N_element', 'N_node', 'N_qi', ...
    'mu_m', 'zeta_m', 'eta_m', 'Length', 'Width', 'thick', 'nu', ...
    '-v7');

fprintf('\n=== Dumped %s ===\n', out_path);
fprintf('  eps = %.3e (z-perturbation amplitude)\n', eps);
fprintf('  |q_perturb - q_ref|_max = %.3e\n', max(abs(q_perturb - q_ref)));
fprintf('  |Qe_global|_F = %.6e  (membrane internal force at perturb)\n', norm(Qe_global));
fprintf('  |Qk_global|_F = %.6e  (bending  internal force at perturb)\n', norm(Qk_global));
fprintf('  max|Qe_global| = %.6e\n', max(abs(Qe_global)));
fprintf('  max|Qk_global| = %.6e\n', max(abs(Qk_global)));
end
