function dump_traj_long(end_time)
% Run the full Yamano FSI coupled loop to a longer time and save the tip
% trajectory (+ periodic wake snapshots) for long-time Python<->MATLAB fidelity
% comparison. Mirrors run_sim.m but with the benchmark Nx=15, Ny=10, d_t=2e-3.
%
% Usage: dump_traj_long(1.0)   % t* up to 1.0 (500 steps)
% Output: ./fixtures/traj_long_t<end>.mat  with z_tip(time), time_m, h_X_vec, wake snaps.

if nargin < 1, end_time = 1.0; end

script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);
add_pathes;

%% ---- Parameters (Yamano benchmark, d_t=2e-3 to match fixtures) ----
param_ver = 12.0;
End_Time = end_time;
d_t = 2e-3;
core_num = 4; speed_check = 0; alpha_v = 0.5; coupling_flag = 1;
Ma = 1.0; Ua = 25;
theta_v = 7.8e-4/25; theta_a = 0*(Ua/Ma^2)*theta_v; C_theta_a = 0*1e-2; J_a = 0;
dt_rz_end = 0*3e-1;
q_in_norm = @(time)( 0.5*sin(pi*time/0.2).*(time < 0.2) );
q_in_vec = [0 0 1].';
mode_num = 5;
i_snapshot = 50; Snapshot_tmin = 3; Snapshot_tmax = End_Time;
panel_node_plot = 0; pressure_interp_plot = 0; movie_format = 'avi';

mu_m = 1/Ma; nu = 0.3; Length = 1.0; Width = 1.00*Length; thick = 1e-3;
Aa = Width*thick; Ia = Width*thick^3/12; eta_m = mu_m/Ua^2; zeta_m = Aa/Ia*eta_m*Length^2;
Nx = 15; Ny = 10;                 %% Yamano benchmark grid
n_LW = 1.0; N_gauss = 5;
F_in = 0*[0 1 0].';
x_vec = ((0:Nx)/Nx).^n_LW*Length; y_vec = (0:Ny)/Ny*Width; N_element = Nx*Ny;
Dp_mat = 1/(1 - nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];
flag_fluid_bench = 0; flag_output = 0;

k_omega = 1/2; Theta_pitch = pi/6.0; omega_pitch = 2*k_omega;
theta_pitch_time = @(time)(Theta_pitch*sin(omega_pitch*time));
dt_theta_pitch_time = @(time)(Theta_pitch*omega_pitch.*cos(omega_pitch*time));
dtt_theta_pitch_time = @(time)(-Theta_pitch*omega_pitch.^2.*sin(omega_pitch*time));
L_pitch_center = Length/2;
R_pitch = @(theta)[cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
dt_R_pitch = @(theta, omega) omega*[-sin(theta) 0 cos(theta); 0 0 0; -cos(theta) 0 -sin(theta)];
H_1_2nd = @(k)besselh(1, 2, k); H_0_2nd = @(k)besselh(0, 2, k);
C_theodorsen = @(k)(H_1_2nd(k)/(H_1_2nd(k) + 1i*H_0_2nd(k)));

U_in = 1.0; V_in = ones(N_element,1)*U_in*[1 0 0];
r_eps.fine = 1e-6; r_eps.rough = 10e-2; Ncore = 2; eps_v = 1e-9;
dL_vec = diff(x_vec); dt_wake = dL_vec(end)/U_in; dt_wake_per_dt = ceil(dt_wake/d_t);

node_r_0 = [1:Ny+1]; node_dxr_0 = [1:Ny+1]; node_dyr_0 = [1:Ny+1];
node_dxr_theta_c = []; node_dyr_theta_c = []; element_C_theta = 1:Ny;
R_wake_x_threshold = 5.5*Length; R_wake_x_threshold_no_change = R_wake_x_threshold - 1.5*Length;

global var_param
var_param.Length = Length; var_param.Nx = Nx; var_param.d_t = d_t; var_param.alpha_v = alpha_v;
var_param.theta_a = theta_a; var_param.C_theta_a = C_theta_a; var_param.J_a = J_a;
var_param.node_r_0 = node_r_0; var_param.node_dxr_0 = node_dxr_0; var_param.node_dyr_0 = node_dyr_0;
var_param.node_dxr_theta_c = node_dxr_theta_c; var_param.node_dyr_theta_c = node_dyr_theta_c;
var_param.r_eps = r_eps; var_param.Ncore = Ncore; var_param.eps_v = eps_v;

maxNumCompThreads(core_num);

%% ---- Setup ----
generate_shape_function;
generate_elements;
generate_panel;
generate_matrices;
generate_Qf_time_mat;
time_m = 0:d_t:End_Time;
initial_values;
solve_mode;

%% ---- Wake snapshot checkpoints (t* values) ----
snap_times = [0.068, 0.2, 0.5, end_time*0.8];
snap_times = snap_times(snap_times <= end_time);
snaps = struct('time', {}, 'Gamma', {}, 'Gamma_wake', {}, 'n_wake', {}, 'q_vec', {});

%% ---- Time loop (identical to run_full/run_sim) ----
fprintf('[traj] integrating to t*=%.2f (%d steps), dt_wake_per_dt=%d ...\n', End_Time, length(time_m), dt_wake_per_dt);
tic
i_time = 1; i_time_cnt = 1; i_wake_time = 1; time_fluid = 0;
d_t_wake = d_t*dt_wake_per_dt; fluid_compute_flag = 1; time = 0;
next_snap = 1;

while time <= time_m(end) || ~fluid_compute_flag
    time = i_time*d_t;
    if mod(i_time, max(1, floor(End_Time/d_t/20))) == 0
        fprintf('  t*=%.3f / %.1f  (%.0f%%)  n_wake=%d\n', time, End_Time, 100*time/End_Time, ...
            exist('Gamma_wake','var')*numel(who('Gamma_wake')));
    end

    if flag_fluid_bench, rigid_structure; else, solve_structure; end

    if mod(i_time, dt_wake_per_dt) == 1
        if fluid_compute_flag
            old_Qf_p_global = Qf_p_global; old_Qf_p_mat_global = Qf_p_mat_global;
            old_Qf_p_mat0_global = Qf_p_mat0_global; old_Qf_p_lift2_mat_global = Qf_p_lift2_mat_global;
            solve_fluid;
            i_wake_time = i_wake_time + 1;
        else
            Qf_p_global_a = Qf_p_global; Qf_p_mat_global_a = Qf_p_mat_global;
            Qf_p_mat0_global_a = Qf_p_mat0_global; Qf_p_lift2_mat_global_a = Qf_p_lift2_mat_global;
            time_fluid = time;
        end
    end

    solve_energy;

    % Wake snapshot
    if next_snap <= numel(snap_times) && time >= snap_times(next_snap) && fluid_compute_flag
        s = numel(snaps) + 1;
        snaps(s).time = time;
        if exist('Gamma','var'), snaps(s).Gamma = Gamma; else, snaps(s).Gamma = []; end
        if exist('Gamma_wake','var'), snaps(s).Gamma_wake = Gamma_wake; snaps(s).n_wake = numel(Gamma_wake)/Ny;
        else, snaps(s).Gamma_wake = []; snaps(s).n_wake = 0; end
        snaps(s).q_vec = h_X_vec(1:N_q_all, max(1,i_time));
        next_snap = next_snap + 1;
    end

    if mod(i_time, dt_wake_per_dt) == 1
        if fluid_compute_flag
            i_time = i_time - i_time_cnt; fluid_compute_flag = 0;
        else
            i_time_cnt = 0; fluid_compute_flag = 1;
        end
    end
    i_time = i_time + 1; i_time_cnt = i_time_cnt + 1;
end
fprintf('[traj] done in %.1f s\n', toc);

%% ---- Extract tip z trajectory (correct 9-DOF indexing) ----
trail_node = Nx*(Ny+1) + floor(Ny/2) + 1;        % 1-indexed
z_dof = 9*(trail_node - 1) + 3;                   % z position DOF
z_tip = h_X_vec(z_dof, 1:length(time_m));

if ~exist('./fixtures','dir'); mkdir('./fixtures'); end
out = sprintf('./fixtures/traj_long_t%.1f.mat', End_Time);
save(out, 'z_tip', 'time_m', 'z_dof', 'trail_node', 'Nx', 'Ny', 'd_t', ...
     'dt_wake_per_dt', 'N_q_all', 'h_X_vec', 'snaps', '-v7');
fprintf('[traj] saved %s\n', out);
fprintf('  z_tip(t*=0.2)=%.6e  z_tip(end t*=%.2f)=%.6e  max|z_tip|=%.6e\n', ...
    z_tip(min(101,end)), End_Time, z_tip(end), max(abs(z_tip)));
end
