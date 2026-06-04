function run_full
% Full FSI simulation with original parameters
% Bypasses GUI, runs exe.m flow directly

cd('E:\DATA\vscode\FSI_by_FEM_and_UVLM\single_sheet');

% Suppress dialog boxes
warning('off', 'all');

add_pathes;
param_setting;

% Override version_check to skip pause and warndlg
exe_ver = 12.0;
fprintf('Solver version: %.1f, Param version: %.1f\n', exe_ver, param_ver);
if exe_ver ~= param_ver
    fprintf('WARNING: Version mismatch!\n');
end

maxNumCompThreads(core_num);

% Run original exe flow
clc;
clear all;
close all hidden;

delete('*.asv');
delete('*.log');

add_pathes;
param_setting;

fprintf('Solver version: %.1f\n', 12.0);

maxNumCompThreads(core_num);

% Shape functions
generate_shape_function;

% Mesh
generate_elements;

% Fluid panels
generate_panel;

% Matrices
generate_matrices;

% External forces
generate_Qf_time_mat;

% Initial values
time_m = 0:d_t:End_Time;
initial_values;

% Mode analysis
% Inline solve_mode without warndlg
q_vec = h_X_vec(1:N_q_all,1);
dt_q_vec = 0*q_vec;

flag_output = 1;
theta_a_tmp = theta_a;
theta_a = 1;
generate_stiff_matrices;
theta_a = theta_a_tmp;

% Build stiffness matrices (from solve_mode)
dq_e_Dp_dq_e_global = sparse(N_q_all,N_q_all);
for ii = 1:N_element
    i_vec = repmat((N_qi*(nodes(ii,:)-1)+1).',[1 N_qi])+repmat(0:N_qi-1,[length(nodes(ii,:)) 1]);
    i_vec = reshape(i_vec.',1,[]);
    dq_e_Dp_dq_e_global(i_vec,i_vec) = dq_e_Dp_dq_e_global(i_vec,i_vec)+squeeze(zeta_m*Qd_eps_mat_i(:,:,ii));
end
K_e_m_mat = dq_e_Dp_dq_e_global;

dq_k_Dp_dq_k_global = sparse(N_q_all,N_q_all);
for ii = 1:N_element
    i_vec = repmat((N_qi*(nodes(ii,:)-1)+1).',[1 N_qi])+repmat(0:N_qi-1,[length(nodes(ii,:)) 1]);
    i_vec = reshape(i_vec.',1,[]);
    dq_k_Dp_dq_k_global(i_vec,i_vec) = dq_k_Dp_dq_k_global(i_vec,i_vec)+squeeze(eta_m*Qd_k_mat_i(:,:,ii));
end
K_e_k_mat = dq_k_Dp_dq_k_global;

% Apply BCs
i_r = reshape((repmat((N_qi*(node_r_0-1)+1).',[1 3])+repmat(0:2,[length(node_r_0) 1])).',1,[]);
i_dx = reshape((repmat((N_qi*(node_dxr_0-1)+4).',[1 3])+repmat(0:2,[length(node_dxr_0) 1])).',1,[]);
i_dy = reshape((repmat((N_qi*(node_dyr_0-1)+7).',[1 3])+repmat(0:2,[length(node_dyr_0) 1])).',1,[]);
i_vec_bc = [i_r i_dx i_dy];
not_i_vec = 1:N_q_all; not_i_vec(i_vec_bc) = [];

M_global_BC = M_global; M_global_BC(i_vec_bc,:)=[]; M_global_BC(:,i_vec_bc)=[];
K_e_m_mat_BC = K_e_m_mat; K_e_m_mat_BC(i_vec_bc,:)=[]; K_e_m_mat_BC(:,i_vec_bc)=[];
K_e_k_mat_BC = K_e_k_mat; K_e_k_mat_BC(i_vec_bc,:)=[]; K_e_k_mat_BC(:,i_vec_bc)=[];

% Eigenvalue solve
[Phi_dq_mat, omega_a2] = eigs((mu_m*M_global_BC)\(K_e_m_mat_BC+K_e_k_mat_BC), mode_num, 'SM');
omega_a = sqrt(diag(omega_a2));

Phi_dq_mat_BC = zeros(N_q_all,mode_num);
Phi_dq_mat_BC(not_i_vec,:) = Phi_dq_mat;
Phi_q_mat_BC = repmat(q_vec,[1 mode_num])+Phi_dq_mat_BC;

zeta_n = theta_a*omega_a/2;

fprintf('Natural frequencies: %s\n', mat2str(omega_a'));
fprintf('Modal damping: %s\n', mat2str(zeta_n'));

% Time integration (same as exe.m)
flag_output = 0;
speed_check = 0;
if speed_check == 1
    time_m = time_m(1:10);
    profile on;
end

tic;
i_time = 1; i_time_cnt = 1; i_wake_time = 1;
time_fluid = 0; d_t_wake = d_t*dt_wake_per_dt;
time_wake_m = 0; fluid_compute_flag = 1; time = 0;
measure_time_struct = 0; measure_time_fluid = 0;

while time <= time_m(end) || ~fluid_compute_flag
    time = i_time*d_t;

    if mod(i_time, max(1, floor(length(time_m)/20))) == 0
        fprintf('Time=%.4f/%.1f (%.0f%%)\n', time, End_Time, time/End_Time*100);
    end

    measure_time_struct_tmp = toc;
    if flag_fluid_bench
        rigid_structure;
    else
        solve_structure;
    end
    measure_time_struct = measure_time_struct+(toc-measure_time_struct_tmp);

    if mod(i_time, dt_wake_per_dt) == 1
        drawnow;
        if fluid_compute_flag
            old_Qf_p_global = Qf_p_global;
            old_Qf_p_mat_global = Qf_p_mat_global;
            old_Qf_p_mat0_global = Qf_p_mat0_global;
            old_Qf_p_lift2_mat_global = Qf_p_lift2_mat_global;
            measure_time_fluid_tmp = toc;
            solve_fluid;
            measure_time_fluid = measure_time_fluid+(toc-measure_time_fluid_tmp);
            i_wake_time = i_wake_time+1;
        else
            Qf_p_global_a = Qf_p_global;
            Qf_p_mat_global_a = Qf_p_mat_global;
            Qf_p_mat0_global_a = Qf_p_mat0_global;
            Qf_p_lift2_mat_global_a = Qf_p_lift2_mat_global;
            time_fluid = time;
        end
    end

    solve_energy;

    if mod(i_time, 500) == 0 && ~fluid_compute_flag
        save ./save/NUM_DATA -v7.3;
    end

    if mod(i_time, dt_wake_per_dt) == 1
        if fluid_compute_flag
            i_time = i_time-i_time_cnt;
            fluid_compute_flag = 0;
        else
            i_time_cnt = 0;
            fluid_compute_flag = 1;
        end
    end

    i_time = i_time+1;
    i_time_cnt = i_time_cnt+1;
end
measure_time_all = toc;

fprintf('\n=== Simulation Complete ===\n');
fprintf('Total: %.1f [s], Structure: %.1f [s], Fluid: %.1f [s]\n', measure_time_all, measure_time_struct, measure_time_fluid);

save ./save/NUM_DATA -v7.3;
fprintf('Results saved to ./save/NUM_DATA.mat\n');

% Generate key plots
fprintf('\nGenerating plots...\n');

idx_r = reshape([1:N_qi:N_q_all; 2:N_qi:N_q_all; 3:N_qi:N_q_all],1,[]);

% Plot 1: Tip displacement
fig1 = figure('Visible','off');
trailing_node = Nx*(Ny+1)+floor(Ny/2)+1;
z_trail = squeeze(h_X_vec(3*(trailing_node-1)+3,:));
plot(time_m, z_trail(1:length(time_m)),'b-','LineWidth',1.5);
xlabel('Time'); ylabel('Z^* tip'); title('Tip Displacement'); grid on;
saveas(gcf,'./save/fig/tip_displacement.png');
fprintf('Saved tip_displacement.png\n');

% Plot 2: Energy
fig2 = figure('Visible','off');
plot(time_m, h_E_em(1:length(time_m)),'r-', time_m, h_E_ek(1:length(time_m)),'b-');
legend('Elastic','Kinetic'); xlabel('Time'); ylabel('Energy'); title('Energy'); grid on;
saveas(gcf,'./save/fig/energy.png');
fprintf('Saved energy.png\n');

% Plot 3: 3D snapshot
fig3 = figure('Visible','off');
r_final = reshape(h_X_vec(idx_r,end),3,[]);
X3=zeros(4,N_element); Y3=zeros(4,N_element); Z3=zeros(4,N_element);
for ii=1:N_element
    X3(:,ii)=r_final(1,nodes(ii,:));
    Y3(:,ii)=r_final(2,nodes(ii,:));
    Z3(:,ii)=r_final(3,nodes(ii,:));
end
patch(X3,Y3,Z3,'r','EdgeColor','k','FaceAlpha',0.8);
view([1 -2 1]); axis equal; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z'); title('Final Deformation');
light; lighting gouraud;
saveas(gcf,'./save/fig/final_snapshot.png');
fprintf('Saved final_snapshot.png\n');

% Export CSV
T1 = table((1:length(omega_a))', omega_a(:), 'VariableNames', {'Mode','Omega'});
writetable(T1,'./save/natural_frequencies.csv');
T2 = table(time_m(:), z_trail(1:length(time_m))', 'VariableNames', {'Time','Z_tip'});
writetable(T2,'./save/tip_displacement.csv');
fprintf('Exported CSV data files\n');

fprintf('\n=== ALL DONE ===\n');
end
