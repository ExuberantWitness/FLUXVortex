%% ========================================================================
%  FSI_by_FEM_and_UVLM - Automated Run Script
%  Runs the single_sheet FSI simulation with configurable parameters
%  and collects results to PNG images and CSV data files
%% ========================================================================

clc; clear all; close all hidden;

%% ========================================================================
%  SECTION 0: Setup paths
%% ========================================================================
script_dir = fileparts(mfilename('fullpath'));
cd(script_dir);

% Add all paths
addpath ./save;
addpath ./cores;
addpath ./cores/functions;
addpath ./cores/functions/structure;
addpath ./cores/functions/fluid;
addpath ./cores/solver;
addpath ./cores/solver/structure;
addpath ./cores/solver/fluid;

% Add toolboxes
addpath ./cores/ToolBoxes/Plate_Mesh;
addpath ./cores/ToolBoxes/mpg_write/src;
addpath ./cores/ToolBoxes/mmwrite;
addpath ./cores/ToolBoxes/mntimes;
addpath ./cores/ToolBoxes/TriStream;
addpath ./cores/ToolBoxes/quiver5;
addpath ./cores/ToolBoxes/FEM_sparse/FEM_sparse;

fprintf('=== FSI Simulation Starting ===\n');
fprintf('Working directory: %s\n', pwd);

%% ========================================================================
%  SECTION 1: Set parameters (override param_setting for quick test)
%% ========================================================================

param_ver = 12.0;

%% Simulation parameters
End_Time = 5;          % Shorter time for quick test (original: 30)
d_t = 2e-3;            % Time step (original: 1.5e-3)
core_num = 4;          % CPU cores
speed_check = 0;
alpha_v = 0.5;         % Implicit solver

%% Flow parameters
Ma = 1.0;              % Mass ratio
Ua = 25;               % Nondimensional flow velocity
coupling_flag = 1;     % Strong coupling

%% Damping
theta_v = 7.8e-4/25;
theta_a = 0*(Ua/Ma^2)*theta_v;
C_theta_a = 0*1e-2;
J_a = 0;
dt_rz_end = 0*3e-1;    % Initial disturbance velocity at trailing edge

%% Initial disturbance
q_in_norm = @(time)( 0.5*sin(pi*time/0.2).*(time < 0.2) );
q_in_vec = [0 0 1].';

%% Mode analysis
mode_num = 5;

%% Plot parameters
i_snapshot = 50;
Snapshot_tmin = 3;
Snapshot_tmax = End_Time;
panel_node_plot = 0;
pressure_interp_plot = 0;
movie_format = 'avi';

%% Sheet geometry
mu_m = 1/Ma;
nu = 0.3;
Length = 1.0;
Width = 1.00*Length;
thick = 1e-3;
Aa = Width*thick;
Ia = Width*thick^3/12;
eta_m = mu_m/Ua^2;
zeta_m = Aa/Ia*eta_m*Length^2;

Nx = 10;    % x elements (original: 15)
Ny = 6;     % y elements (original: 10)

n_LW = 1.0;
N_gauss = 5;

k_gravity = 0.136^4/(1.21^3*4.989e-4);
F_in = 0*[0 1 0].';

x_vec = ((0:Nx)/Nx).^n_LW*Length;
y_vec = (0:Ny)/Ny*Width;
N_element = Nx*Ny;

Dp_mat = 1/(1 - nu^2)*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];

%% Fluid-only benchmark flag
flag_fluid_bench = 0;
flag_output = 0;   % Nonlinear force computation flag (set to 1 inside solve_structure)

%% Pitch parameters
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

%% Flow parameters
U_in = 1.0;
V_in = ones(N_element,1)*U_in*[1 0 0];
r_eps.fine = 1e-6;
r_eps.rough = 10e-2;
Ncore = 2;
eps_v = 1e-9;

dL_vec = diff(x_vec);
dt_wake = dL_vec(end)/U_in;
dt_wake_per_dt = ceil(dt_wake/d_t);

%% Boundary conditions (clamped at leading edge)
node_r_0 = [1:Ny+1];
node_dxr_0 = [1:Ny+1];
node_dyr_0 = [1:Ny+1];
node_dxr_theta_c = [];
node_dyr_theta_c = [];
element_C_theta = 1:Ny;

%% Wake parameters
R_wake_x_threshold = 5.5*Length;
R_wake_x_threshold_no_change = R_wake_x_threshold - 1.5*Length;

%% Global variables
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

%% ========================================================================
%  SECTION 2: Version check
%% ========================================================================
exe_ver = 12.0;
fprintf('Solver version: %.1f\n', exe_ver);
fprintf('Parameter file version: %.1f\n', param_ver);

%% ========================================================================
%  SECTION 3: Run simulation (same as exe.m)
%% ========================================================================
fprintf('\n--- Starting FSI Computation ---\n');
maxNumCompThreads(core_num);

%% Mesh and shape functions
fprintf('[Step 1] Generating shape functions...\n');
generate_shape_function;

fprintf('[Step 2] Generating mesh...\n');
generate_elements;

fprintf('[Step 3] Generating panels...\n');
generate_panel;

fprintf('[Step 4] Generating matrices...\n');
generate_matrices;

fprintf('[Step 5] Generating external forces...\n');
generate_Qf_time_mat;

%% Time loop
time_m = 0:d_t:End_Time;
initial_values;

fprintf('[Step 6] Solving modes...\n');
solve_mode;

fprintf('[Step 7] Starting time integration...\n');
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

while time <= time_m(end) || ~fluid_compute_flag
    time = i_time*d_t;

    if mod(i_time, max(1, floor(End_Time/d_t/20))) == 0
        fprintf('  Time = %.4f / %.1f  (%.0f%%)\n', time, End_Time, time/End_Time*100);
    end

    %% Structure solver
    measure_time_struct_tmp = toc;
    if flag_fluid_bench
        rigid_structure;
    else
        solve_structure;
    end
    measure_time_struct = measure_time_struct + (toc - measure_time_struct_tmp);

    %% Fluid solver
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

        if coupling_flag
            coupling_str = 'Strong coupling';
        else
            coupling_str = 'Weak coupling';
        end
    end

    %% Energy evaluation
    solve_energy;

    %% Auto-save
    if mod(i_time, 500) == 0 && ~fluid_compute_flag
        save ./save/NUM_DATA -v7.3
    end

    %% Fluid compute flag toggle
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
measure_time_all = toc;

fprintf('\n=== Simulation Complete ===\n');
fprintf('Total time: %.2f [s]\n', measure_time_all);
fprintf('Structure solver: %.2f [s]\n', measure_time_struct);
fprintf('Fluid solver: %.2f [s]\n', measure_time_fluid);

%% Save results
save ./save/NUM_DATA -v7.3
fprintf('Results saved to ./save/NUM_DATA.mat\n');

%% ========================================================================
%  SECTION 4: Post-processing and visualization
%% ========================================================================
fprintf('\n--- Post-processing ---\n');

% Extract tip displacement
idx_r = reshape([1:N_qi:N_q_all; 2:N_qi:N_q_all; 3:N_qi:N_q_all], 1, []);
r_vec_final = reshape(h_X_vec(idx_r, end), 3, []);

% Spanwise center displacement
i_center = floor(Ny/2)+1;
tip_nodes = [i_center, Ny+i_center];
tip_x = r_vec_final(1, tip_nodes);
tip_y = r_vec_final(2, tip_nodes);
tip_z = r_vec_final(3, tip_nodes);

fprintf('Tip displacement (center span):\n');
fprintf('  X = %.6f\n', tip_x(1));
fprintf('  Z = %.6f\n', tip_z(1));

% Time history of tip Z-displacement
r_vec_all = reshape(h_X_vec(idx_r, :), 3, []);
z_tip_history = r_vec_all(3, end-Ny:end-Ny+1);
z_tip_time = squeeze(z_tip_history(1, :))';

%% Plot 1: Mesh
fig1 = figure('Visible', 'off');
X_mesh = zeros(4,N_element);
Y_mesh = zeros(4,N_element);
for ii = 1:N_element
    X_mesh(:,ii) = coordinates(nodes(ii,:),1);
    Y_mesh(:,ii) = coordinates(nodes(ii,:),2);
end
patch(X_mesh, Y_mesh, 'w', 'EdgeColor', 'k');
axis equal;
xlabel('x position'); ylabel('y position');
title('Finite Element Mesh');
saveas(gcf, './save/fig/mesh.png');
fprintf('Saved: mesh.png\n');

%% Plot 2: Tip displacement time history
fig2 = figure('Visible', 'off');
% Get center-span trailing edge node displacement
trailing_center_node = Nx*(Ny+1) + floor(Ny/2)+1;
z_trail = reshape(h_X_vec(3*(trailing_center_node-1)+3, :), 1, []);
plot(time_m, z_trail(1:length(time_m)), 'b-', 'LineWidth', 1.5);
xlabel('Nondimensional time'); ylabel('Z^* position');
title('Trailing Edge Displacement (center span)');
grid on;
saveas(gcf, './save/fig/tip_displacement.png');
fprintf('Saved: tip_displacement.png\n');

%% Plot 3: Energy history
fig3 = figure('Visible', 'off');
if exist('h_E_inertia', 'var')
    plot(time_m, h_E_em(1:length(time_m)), 'r-', 'LineWidth', 1.5); hold on;
    plot(time_m, h_E_ek(1:length(time_m)), 'b-', 'LineWidth', 1.5);
    plot(time_m, h_E_inertia(1:length(time_m)), 'g-', 'LineWidth', 1.5);
    legend('Elastic', 'Kinetic', 'Inertia');
    xlabel('Time'); ylabel('Energy');
    title('Energy History');
    grid on;
end
saveas(gcf, './save/fig/energy_history.png');
fprintf('Saved: energy_history.png\n');

%% Plot 4: Work rate
fig4 = figure('Visible', 'off');
if exist('h_W_total_m', 'var')
    plot(time_m(1:end-1), h_W_total_m(1:length(time_m)-1), 'b-', 'LineWidth', 2);
    xlabel('Time'); ylabel('Work rate');
    title('Work Rate');
    grid on;
end
saveas(gcf, './save/fig/work_rate.png');
fprintf('Saved: work_rate.png\n');

%% Plot 5: 3D snapshot of final state
fig5 = figure('Visible', 'off');
X3d = zeros(4,N_element);
Y3d = zeros(4,N_element);
Z3d = zeros(4,N_element);
for ii = 1:N_element
    X3d(:,ii) = r_vec_final(1, nodes(ii,:));
    Y3d(:,ii) = r_vec_final(2, nodes(ii,:));
    Z3d(:,ii) = r_vec_final(3, nodes(ii,:));
end
patch(X3d, Y3d, Z3d, 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.8);
view([1 -2 1]); axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Final Sheet Deformation');
light; lighting gouraud;
saveas(gcf, './save/fig/final_snapshot.png');
fprintf('Saved: final_snapshot.png\n');

%% Plot 6: Mode shapes
if exist('Phi_q_mat_BC', 'var') && exist('omega_a', 'var')
    for i_mode = 1:min(mode_num, length(omega_a))
        fig_mode = figure('Visible', 'off');
        Phi_r_vec = reshape(Phi_q_mat_BC(idx_r, i_mode), 3, []);
        Xm = zeros(4,N_element);
        Ym = zeros(4,N_element);
        Zm = zeros(4,N_element);
        for ii = 1:N_element
            Xm(:,ii) = Phi_r_vec(1, nodes(ii,:));
            Ym(:,ii) = Phi_r_vec(2, nodes(ii,:));
            Zm(:,ii) = Phi_r_vec(3, nodes(ii,:));
        end
        patch(Xm, Ym, Zm, 'r', 'EdgeColor', 'k');
        view([1 -2 1]); axis equal; grid on;
        xlabel('X'); ylabel('Y'); zlabel('Z');
        title(sprintf('Mode %d: \\omega^*_n = %.3f', i_mode, omega_a(i_mode)));
        light; lighting gouraud;
        saveas(gcf, sprintf('./save/fig/modes/mode_%d.png', i_mode));
        fprintf('Saved: mode_%d.png (omega = %.3f)\n', i_mode, omega_a(i_mode));
    end
end

%% Export numerical results to CSV
fprintf('\n--- Exporting CSV data ---\n');

% Natural frequencies
if exist('omega_a', 'var')
    T = table((1:length(omega_a))', omega_a(:), 'VariableNames', {'Mode', 'Omega_n'});
    writetable(T, './save/natural_frequencies.csv');
    fprintf('Saved: natural_frequencies.csv\n');
    fprintf('\nNatural Frequencies:\n');
    disp(T);
end

% Tip displacement history
if exist('z_trail', 'var')
    T2 = table(time_m(:), z_trail(1:length(time_m))', 'VariableNames', {'Time', 'Z_tip'});
    writetable(T2, './save/tip_displacement.csv');
    fprintf('Saved: tip_displacement.csv\n');
end

% Energy history
if exist('h_E_em', 'var')
    T3 = table(time_m(:), h_E_em(1:length(time_m))', h_E_ek(1:length(time_m))', ...
               h_E_inertia(1:length(time_m))', ...
               'VariableNames', {'Time', 'E_elastic', 'E_kinetic', 'E_inertia'});
    writetable(T3, './save/energy_history.csv');
    fprintf('Saved: energy_history.csv\n');
end

fprintf('\n========================================\n');
fprintf('=== All Results Generated ===\n');
fprintf('========================================\n');
fprintf('PNG figures: ./save/fig/\n');
fprintf('CSV data: ./save/\n');
fprintf('MAT data: ./save/NUM_DATA.mat\n');
fprintf('========================================\n');
