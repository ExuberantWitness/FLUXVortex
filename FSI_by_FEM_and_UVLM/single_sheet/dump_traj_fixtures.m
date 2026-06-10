%% ========================================================================
%  dump_fixtures_multi.m — Phase 3b multi-checkpoint fixture generator
%  Mirrors dump_fixture_run.m but dumps at MULTIPLE times to support layered
%  comparison (Mf2_vec1 must match across the whole wake-growth transient).
%
%  Checkpoints (t*): 0.0680, 0.1360, 0.1995, 0.2640
%   - 0.0680: first fluid step, NO wake yet
%   - 0.1360: 1 wake row
%   - 0.1995: Yamano benchmark reference (3 wake rows)
%   - 0.2640: 4 wake rows, transitioning to steady
%
%  Outputs: ./fixtures/fixture_step{i}_t{time}.mat per checkpoint
%% ========================================================================

clc; clear all; close all hidden;

%% Setup paths
script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);
addpath ./save;
addpath ./cores;
addpath ./cores/functions;
addpath ./cores/functions/structure;
addpath ./cores/functions/fluid;
addpath ./cores/solver;
addpath ./cores/solver/structure;
addpath ./cores/solver/fluid;
addpath ./cores/ToolBoxes/Plate_Mesh;
addpath ./cores/ToolBoxes/mpg_write/src;
addpath ./cores/ToolBoxes/mmwrite;
addpath ./cores/ToolBoxes/mntimes;
addpath ./cores/ToolBoxes/TriStream;
addpath ./cores/ToolBoxes/quiver5;
addpath ./cores/ToolBoxes/FEM_sparse/FEM_sparse;

fprintf('=== Multi-checkpoint Phase 3b fixture dump ===\n');

%% Output dir
CHECKPOINT_DIR_LOCAL = fullfile(script_dir, 'fixtures_traj');
if ~exist(CHECKPOINT_DIR_LOCAL, 'dir')
    mkdir(CHECKPOINT_DIR_LOCAL);
end

%% Globals to drive multi-checkpoint hook in calc_fluid_force.m
global TIME_FOR_DUMP CHECKPOINT_TIMES CHECKPOINT_DIR CHECKPOINTS_DONE
global DUMP_FIXTURE_PATH DUMP_DONE
% Clear single-mode globals so multi-mode wins
DUMP_FIXTURE_PATH = [];
DUMP_DONE = [];

CHECKPOINT_TIMES = [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.8000, 1.0000];
CHECKPOINT_DIR = CHECKPOINT_DIR_LOCAL;
CHECKPOINTS_DONE = false(size(CHECKPOINT_TIMES));
TIME_FOR_DUMP = 0;

%% Parameters (Yamano-matched: U*=25, M*=1, AR=1, 15x10)
param_ver = 12.0;
End_Time = 1.02;        % full trajectory to large deformation
d_t = 2e-3;
core_num = 4;
speed_check = 0;
alpha_v = 0.5;

Ma = 1.0;
Ua = 25;
coupling_flag = 1;

theta_v = 7.8e-4/25;
theta_a = 0*(Ua/Ma^2)*theta_v;
C_theta_a = 0*1e-2;
J_a = 0;
dt_rz_end = 0*3e-1;

q_in_norm = @(time)( 0.5*sin(pi*time/0.2).*(time < 0.2) );
q_in_vec = [0 0 1].';

mode_num = 5;

i_snapshot = 50;
Snapshot_tmin = 3;
Snapshot_tmax = End_Time;
panel_node_plot = 0;
pressure_interp_plot = 0;
movie_format = 'avi';

mu_m = 1/Ma;
nu = 0.3;
Length = 1.0;
Width = 1.00*Length;
thick = 1e-3;
Aa = Width*thick;
Ia = Width*thick^3/12;
eta_m = mu_m/Ua^2;
zeta_m = Aa/Ia*eta_m*Length^2;

Nx = 15;
Ny = 10;

n_LW = 1.0;
N_gauss = 5;
k_gravity = 0.136^4/(1.21^3*4.989e-4);
F_in = 0*[0 1 0].';

x_vec = ((0:Nx)/Nx).^n_LW*Length;
y_vec = (0:Ny)/Ny*Width;
N_element = Nx*Ny;

Dp_mat = 1/(1 - nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

flag_fluid_bench = 0;
flag_output = 0;

k_omega = 1/2;
Theta_pitch = pi/6.0;
omega_pitch = 2*k_omega;
theta_pitch_time = @(time)(Theta_pitch*sin(omega_pitch*time));
dt_theta_pitch_time = @(time)(Theta_pitch*omega_pitch.*cos(omega_pitch*time));
dtt_theta_pitch_time = @(time)(-Theta_pitch*omega_pitch.^2.*sin(omega_pitch*time));
L_pitch_center = Length/2;
R_pitch = @(theta)[cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
dt_R_pitch = @(theta, omega) omega*[-sin(theta) 0 cos(theta); 0 0 0; -cos(theta) 0 -sin(theta)];
H_1_2nd = @(k)besselh(1, 2, k);
H_0_2nd = @(k)besselh(0, 2, k);
C_theodorsen = @(k)(H_1_2nd(k)/(H_1_2nd(k) + 1i*H_0_2nd(k)));

U_in = 1.0;
V_in = ones(N_element,1)*U_in*[1 0 0];
r_eps.fine = 1e-6;
r_eps.rough = 10e-2;
Ncore = 2;
eps_v = 1e-9;

dL_vec = diff(x_vec);
dt_wake = dL_vec(end)/U_in;
dt_wake_per_dt = ceil(dt_wake/d_t);

node_r_0 = [1:Ny+1];
node_dxr_0 = [1:Ny+1];
node_dyr_0 = [1:Ny+1];
node_dxr_theta_c = [];
node_dyr_theta_c = [];
element_C_theta = 1:Ny;

R_wake_x_threshold = 5.5*Length;
R_wake_x_threshold_no_change = R_wake_x_threshold - 1.5*Length;

global var_param
var_param.Length = Length;
var_param.Nx = Nx;
var_param.d_t = d_t;
var_param.alpha_v = alpha_v;
var_param.theta_a = theta_a;
var_param.C_theta_a = C_theta_a;
var_param.J_a = J_a;
var_param.node_r_0 = node_r_0;
var_param.node_dxr_0 = node_dxr_0;
var_param.node_dyr_0 = node_dyr_0;
var_param.node_dxr_theta_c = node_dxr_theta_c;
var_param.node_dyr_theta_c = node_dyr_theta_c;
var_param.r_eps = r_eps;
var_param.Ncore = Ncore;
var_param.eps_v = eps_v;

exe_ver = 12.0;
fprintf('Mesh: %dx%d, End_Time: %.2f, %d checkpoints\n', Nx, Ny, End_Time, length(CHECKPOINT_TIMES));

%% Setup
maxNumCompThreads(core_num);
generate_shape_function;
generate_elements;
generate_panel;
generate_matrices;
generate_Qf_time_mat;

time_m = 0:d_t:End_Time;
initial_values;
solve_mode;

%% Time loop (mirrors exe.m, with TIME_FOR_DUMP update). Stop early if all checkpoints done.
tic
i_time = 1;
i_time_cnt = 1;
i_wake_time = 1;
time_fluid = 0;
d_t_wake = d_t*dt_wake_per_dt;
time_wake_m = 0;
fluid_compute_flag = 1;
time = 0;
measure_time_struct = 0;
measure_time_fluid = 0;

while (time <= time_m(end) || ~fluid_compute_flag) && ~all(CHECKPOINTS_DONE)
    time = i_time*d_t;
    TIME_FOR_DUMP = time;

    if mod(i_time, max(1, floor(End_Time/d_t/20))) == 0
        fprintf('  Time = %.4f / %.2f  (%d/%d checkpoints done)\n', ...
                time, End_Time, sum(CHECKPOINTS_DONE), length(CHECKPOINT_TIMES));
    end

    measure_time_struct_tmp = toc;
    if flag_fluid_bench
        rigid_structure;
    else
        solve_structure;
    end
    measure_time_struct = measure_time_struct + (toc - measure_time_struct_tmp);

    if mod(i_time, dt_wake_per_dt) == 1
        drawnow

        if fluid_compute_flag
            old_Qf_p_global = Qf_p_global;
            old_Qf_p_mat_global = Qf_p_mat_global;
            old_Qf_p_mat0_global = Qf_p_mat0_global;
            old_Qf_p_lift2_mat_global = Qf_p_lift2_mat_global;

            measure_time_fluid_tmp = toc;
            solve_fluid;
            measure_time_fluid = measure_time_fluid + (toc - measure_time_fluid_tmp);
            i_wake_time = i_wake_time + 1;
        else
            Qf_p_global_a = Qf_p_global;
            Qf_p_mat_global_a = Qf_p_mat_global;
            Qf_p_mat0_global_a = Qf_p_mat0_global;
            Qf_p_lift2_mat_global_a = Qf_p_lift2_mat_global;
            time_fluid = time;
        end
    end

    solve_energy;

    if mod(i_time, dt_wake_per_dt) == 1
        if fluid_compute_flag
            i_time = i_time - i_time_cnt;
            fluid_compute_flag = 0;
        else
            i_time_cnt = 0;
            fluid_compute_flag = 1;
        end
    end

    i_time = i_time + 1;
    i_time_cnt = i_time_cnt + 1;
end

fprintf('\n=== Multi-checkpoint dump done ===\n');
for k = 1:length(CHECKPOINT_TIMES)
    status = 'MISSING';
    if CHECKPOINTS_DONE(k), status = 'OK'; end
    fprintf('  Checkpoint %d (t*=%.4f): %s\n', k, CHECKPOINT_TIMES(k), status);
end
fprintf('Output dir: %s\n', CHECKPOINT_DIR);
