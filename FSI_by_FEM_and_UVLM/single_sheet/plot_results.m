function plot_results
% Post-processing: generate all figures from saved NUM_DATA.mat
% Replaces TriStream/quiver5/mmwrite with MATLAB built-in functions

cd('E:\DATA\vscode\FSI_by_FEM_and_UVLM\single_sheet');

warning('off', 'all');
set(0,'DefaultAxesXGrid','on');
set(0,'DefaultAxesYGrid','on');
set(0, 'DefaultTextFontSize', 20);
set(0, 'DefaultAxesFontSize', 15);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontName', 'Times New Roman');

add_pathes;
param_setting;

maxNumCompThreads(core_num);

% Load simulation data
fprintf('Loading NUM_DATA.mat ...\n');
load ./save/NUM_DATA

fprintf('Data loaded. h_X_vec size: [%s]\n', num2str(size(h_X_vec)));
fprintf('time_m range: [%.4f, %.4f], %d steps\n', time_m(1), time_m(end), length(time_m));

idx_r = reshape([1:N_qi:N_q_all; 2:N_qi:N_q_all; 3:N_qi:N_q_all], 1, []);

% Ensure output dir
if ~exist('./save/fig', 'dir'), mkdir('./save/fig'); end
if ~exist('./save/fig/modes', 'dir'), mkdir('./save/fig/modes'); end

%% ===== Fig 1: FE Mesh =====
fprintf('Fig 1: FE Mesh\n');
h_fig(1) = figure('Visible','off');
X = zeros(4,N_element); Y = zeros(4,N_element);
for ii = 1:N_element
    X(:,ii) = coordinates(nodes(ii,:),1);
    Y(:,ii) = coordinates(nodes(ii,:),2);
end
patch(X, Y, 'w'); axis equal;
xlabel('{\itx} position [-]'); ylabel('{\ity} position [-]');
saveas(gcf, './save/fig/nodes.png');
fprintf('  Saved nodes.png\n');

%% ===== Fig 2: 3D Snapshot (final state) =====
fprintf('Fig 2: 3D Snapshot\n');
h_fig(2) = figure('Visible','off');
r_vec = reshape(h_X_vec(idx_r,end), 3, []);
X = zeros(4,N_element); Y = zeros(4,N_element); Z = zeros(4,N_element);
for ii = 1:N_element
    X(:,ii) = r_vec(1,nodes(ii,:));
    Y(:,ii) = r_vec(2,nodes(ii,:));
    Z(:,ii) = r_vec(3,nodes(ii,:));
end
patch(X, Y, Z, 'r', 'FaceAlpha', 0.8, 'EdgeColor', 'k');
view([1 -2 1]); axis equal; grid on; light; lighting gouraud;
xlabel('{\itX}^* [-]'); ylabel('{\itY}^* [-]'); zlabel('{\itZ}^* [-]');
title('Final Deformation');
saveas(gcf, './save/fig/snapshot.png');
fprintf('  Saved snapshot.png\n');

%% ===== Fig 3: Snapshots over time (like original Fig 3) =====
fprintf('Fig 3: Snapshots over time\n');
h_fig(3) = figure('Visible','off');
Snapshot_tmin = 25; Snapshot_tmax = End_Time;
i_snapshot = 50;
i_time = 1;
for time = time_m
    if mod(i_time, i_snapshot) == 0 && Snapshot_tmin <= time && time <= Snapshot_tmax
        r_vec = reshape(h_X_vec(idx_r,i_time), 3, []);
        X = zeros(4,N_element); Y = zeros(4,N_element); Z = zeros(4,N_element);
        for ii = 1:N_element
            X(:,ii) = r_vec(1,nodes(ii,:));
            Y(:,ii) = r_vec(2,nodes(ii,:));
            Z(:,ii) = r_vec(3,nodes(ii,:));
        end
        patch(X, Y, Z, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        hold on;
    end
    i_time = i_time + 1;
end
view([1 2 1]); axis equal; grid on; light; lighting gouraud;
xlim([-0.5*Length 1.5*Length]); ylim([-1.5*Width 2*Width]); zlim([-0.5*Length 0.5*Length]);
xlabel('{\itX}^* [-]'); ylabel('{\itY}^* [-]'); zlabel('{\itZ}^* [-]');
title('Snapshots of flapping sheet');
saveas(gcf, './save/fig/snapshots_3d.png');
fprintf('  Saved snapshots_3d.png\n');

%% ===== Fig 4: Mid-span displacement =====
fprintf('Fig 4: Mid-span displacement\n');

% Extract centerline displacement
data.N_element = N_element; data.Nx = Nx; data.Ny = Ny;
data.N_qi = N_qi; data.N_q = N_q; data.nodes = nodes; data.h_X_vec = h_X_vec;
[X_center_disp, Z_center_disp] = r_center_disp(data);

h_fig(4) = figure('Visible','off');
trailing_node = Nx*(Ny+1)+floor(Ny/2)+1;
z_trail = squeeze(h_X_vec(3*(trailing_node-1)+3,:));
plot(time_m, z_trail(1:length(time_m)), 'b-', 'LineWidth', 1.5);
xlabel('Nondimensional time'); ylabel('{\itZ}^* trailing edge');
title('Trailing Edge Displacement'); grid on;
saveas(gcf, './save/fig/displacement_mid_span.png');
fprintf('  Saved displacement_mid_span.png\n');

%% ===== Fig 5: Mid-span X-Z phase snapshots =====
fprintf('Fig 5: Mid-span snapshots\n');
h_fig(5) = figure('Visible','off');

[pks, locs] = findpeaks(Z_center_disp(end,:), 'MinPeakDistance', 150);
locs(pks <= 0) = [];
if length(locs) >= 3
    idx_time_Tp = locs(end-2):locs(end-1);
else
    idx_time_Tp = 1:round(size(Z_center_disp,2)*3/4);
end
idx_snapshot = round(linspace(idx_time_Tp(1), idx_time_Tp(end), 50));
plot(X_center_disp(:,idx_snapshot), Z_center_disp(:,idx_snapshot), 'b-', 'LineWidth', 0.2);
axis equal; xlim([0 1]); ylim([-0.5 0.5]);
xlabel('{\itX}^* position'); ylabel('{\itZ}^* position');
title('Mid-span snapshots over one period');
saveas(gcf, './save/fig/snapshot_mid_span.png');
fprintf('  Saved snapshot_mid_span.png\n');

%% ===== Fig 6: Phase plane =====
fprintf('Fig 6: Phase plane\n');
h_fig(6) = figure('Visible','off');
Z_center_vel = [(Z_center_disp(:,2)-Z_center_disp(:,1)), ...
    (Z_center_disp(:,3:end)-Z_center_disp(:,1:end-2))/2, ...
    (Z_center_disp(:,end)-Z_center_disp(:,end-1))] / mean(diff(time_m));
plot(Z_center_disp(end,1:length(time_m)), Z_center_vel(end,1:length(time_m)), 'b-', 'LineWidth', 1);
xlabel('{\itZ}^* position'); ylabel('{\itZ}^* velocity');
title('Phase plane (trailing edge)'); xlim([-0.5 0.5]); ylim([-1.5 1.5]); grid on;
saveas(gcf, './save/fig/disp_vel_mid_span_phase_plane.png');
fprintf('  Saved disp_vel_mid_span_phase_plane.png\n');

%% ===== Fig 7: Energy =====
fprintf('Fig 7: Energy\n');
h_fig(7) = figure('Visible','off');
plot(time_m, h_E_em(1:length(time_m)), 'r-', time_m, h_E_ek(1:length(time_m)), 'b-', 'LineWidth', 1.5);
legend('Elastic', 'Kinetic'); xlabel('Time'); ylabel('Energy');
title('Energy history'); grid on;
saveas(gcf, './save/fig/energy.png');
fprintf('  Saved energy.png\n');

%% ===== Fig 8: Work rate =====
fprintf('Fig 8: Work rate\n');
h_fig(8) = figure('Visible','off');
h_W_inertia = [0 (h_E_inertia(3:end)-h_E_inertia(1:end-2))/(2*d_t) nan];
h_W_em = [0 (h_E_em(3:end)-h_E_em(1:end-2))/(2*d_t) nan];
h_W_ek = [0 (h_E_ek(3:end)-h_E_ek(1:end-2))/(2*d_t) nan];
h_W_Ja = [0 (h_E_Ja(3:end)-h_E_Ja(1:end-2))/(2*d_t) nan];
h_W_total_m = h_W_inertia + h_W_em + h_W_ek + h_W_Ja + h_W_dm + h_W_dk + h_W_d_theta;
h_W_total_m(length(time_m):end) = nan;

plot(time_m(1:end-1), h_W_total_m(1:length(time_m)-1), 'b--', 'LineWidth', 2); hold on;
if exist('time_wake_m','var') && exist('h_W_f_ext','var')
    plot(time_wake_m(1:end-1), h_W_f_ext(1:length(time_wake_m)-1), 'ro-');
end
plot(time_m(1:end-1), h_W_dk(1:length(time_m)-1), 'g');
plot(time_m(1:end-1), h_W_dm(1:length(time_m)-1), 'k');
plot(time_m(1:end-1), h_W_d_theta(1:length(time_m)-1), 'm');
legend('{d_{\itt}\itE_{total}}', '{\itW_{f}}', '{\itW_{dk}}', '{\itW_{dm}}', '{\itW_{d\theta}}');
xlabel('Time'); ylabel('Work rate'); title('Work rate'); ylim([-0.2 0.2]); grid on;
saveas(gcf, './save/fig/work_rate.png');
fprintf('  Saved work_rate.png\n');

%% ===== Fig 9: CL vs alpha (Theodorsen comparison) =====
fprintf('Fig 9: CL vs alpha\n');
h_fig(9) = figure('Visible','off');
if exist('theta_pitch_time','var') && exist('time_wake_m','var') && exist('h_dp_vec','var')
    theta_pitch_time_m = theta_pitch_time(time_wake_m(1:end-1));
    dt_theta_pitch_time_m = dt_theta_pitch_time(time_wake_m(1:end-1));

    A_total = Length*Width;
    dA_vec = dL_vec.*dW_vec;
    dp_sum_vec = h_dp_vec(:,1:length(time_wake_m(1:end-1)));

    if mod(Ny,2) == 0
        idx_cord = floor(Ny/2):Ny:N_element;
        CL_vec = 2*Ny/A_total*(dA_vec(idx_cord)*dp_sum_vec(idx_cord,:) + ...
            dA_vec(idx_cord+1)*dp_sum_vec(idx_cord+1,:))/2.*cos(theta_pitch_time_m);
    else
        idx_cord = floor(Ny/2):Ny:N_element;
        CL_vec = 2*Ny/A_total*(dA_vec(idx_cord+1)*dp_sum_vec(idx_cord+1,:)).*cos(theta_pitch_time_m);
    end

    theta_pitch_theory = Theta_pitch*exp(1i*omega_pitch*time_wake_m(1:end-1));
    CL_theory = pi*(2+1i*k_omega)*C_theodorsen(k_omega)*theta_pitch_theory + 1i*pi*k_omega*theta_pitch_theory;

    plot(theta_pitch_time_m(end/2:end), CL_vec(end/2:end), 'b--', 'LineWidth', 2, 'DisplayName', 'UVLM'); hold on;
    plot(theta_pitch_theory, CL_theory, 'r-', 'LineWidth', 1, 'DisplayName', 'Theodorsen');
    legend; xlabel('{\it\alpha}({\itt}) [rad]'); ylabel('{\itC_L}({\itt})');
    title('CL vs \alpha (UVLM vs Theodorsen)');
    saveas(gcf, './save/fig/alpha_vs_CL.png');
    fprintf('  Saved alpha_vs_CL.png\n');
else
    fprintf('  Skipped (pitch data not available for this configuration)\n');
end

%% ===== Fig 10-12: Velocity distributions =====
fprintf('Fig 10-12: Velocity distributions\n');

if exist('h_r_wake','var') && exist('h_Gamma_wake','var')

    % Use last wake time index
    i_wake_time = length(h_r_wake);
    fprintf('  Using wake time index %d\n', i_wake_time);

    % Velocity grid
    [X_mat, Z_mat] = meshgrid(linspace(-Length, 5*Length, 120), linspace(-1.5*Length, 1.5*Length, 80));
    [N_row, N_col] = size(X_mat);
    X_v = X_mat(:); Z_v = Z_mat(:);
    Y_v = Width/2*ones(N_row*N_col,1);
    r_xyz = [X_v Y_v Z_v];

    % Panel nodes
    r_node = h_r_panel_vec(:,:,i_wake_time);
    N_node = size(r_node,1)/4;
    ii_node = 1:N_node;
    r_node1 = r_node(ii_node,:);
    r_node2 = r_node(N_node+ii_node,:);
    r_node3 = r_node(2*N_node+ii_node,:);
    r_node4 = r_node(3*N_node+ii_node,:);

    % Wake nodes
    r_wake = h_r_wake{i_wake_time};
    N_wake = size(r_wake,1)/4;
    ii_wake = 1:N_wake;
    r_wake1 = r_wake(ii_wake,:);
    r_wake2 = r_wake(N_wake+ii_wake,:);
    r_wake3 = r_wake(2*N_wake+ii_wake,:);
    r_wake4 = r_wake(3*N_wake+ii_wake,:);

    % Velocity computation
    fprintf('  Computing velocity field...\n');
    Gamma_wake = h_Gamma_wake{i_wake_time};
    Gamma_bound = h_Gamma{i_wake_time};

    V_wake = V_wake_func(r_xyz, r_wake1, r_wake2, r_wake3, r_wake4, Gamma_wake, var_param, 0);
    V_gamma = V_wake_func(r_xyz, r_node1, r_node2, r_node3, r_node4, Gamma_bound, var_param, 0);
    V_in = ones(N_row*N_col,1)*U_in*[1 0 0];
    V_xyz = V_wake + V_gamma + V_in;

    Vx_mat = reshape(V_xyz(:,1), N_row, N_col);
    Vz_mat = reshape(V_xyz(:,3), N_row, N_col);
    u_norm_mat = sqrt(Vx_mat.^2 + Vz_mat.^2);
    Vx_mat(u_norm_mat > 2) = nan;
    Vz_mat(u_norm_mat > 2) = nan;
    u_norm_mat(u_norm_mat > 2) = 2;

    rx_mat = reshape(r_xyz(:,1), N_row, N_col);
    rz_mat = reshape(r_xyz(:,3), N_row, N_col);

    % Find time index for sheet shape overlay
    if exist('idx_time_Tp','var')
        i_time_plot = idx_time_Tp(round(end/2));
    else
        i_time_plot = length(time_m);
    end

    % u-velocity
    h_fig(10) = figure('Visible','off');
    contourf(rx_mat, rz_mat, Vx_mat, 40, '-.'); axis equal; grid on; hold on;
    colorbar; ylabel('{\itu}^*'); caxis([0 2]);
    xlim([-Length 5*Length]); ylim([-1.5*Length 1.5*Length]);
    xlabel('{\itX}^* position'); ylabel('{\itZ}^* position');
    title('u-velocity distribution');
    plot(X_center_disp(:,i_time_plot), Z_center_disp(:,i_time_plot), 'b-', 'LineWidth', 3);
    colormap jet;
    saveas(gcf, './save/fig/u_distribution_0.png');
    fprintf('  Saved u_distribution_0.png\n');

    % w-velocity
    h_fig(11) = figure('Visible','off');
    contourf(rx_mat, rz_mat, Vz_mat, 40, '-.'); axis equal; grid on; hold on;
    colorbar; ylabel('{\itw}^*'); caxis([-2 2]);
    xlim([-Length 5*Length]); ylim([-1.5*Length 1.5*Length]);
    xlabel('{\itX}^* position'); ylabel('{\itZ}^* position');
    title('w-velocity distribution');
    plot(X_center_disp(:,i_time_plot), Z_center_disp(:,i_time_plot), 'b-', 'LineWidth', 3);
    colormap jet;
    saveas(gcf, './save/fig/v_distribution_0.png');
    fprintf('  Saved v_distribution_0.png\n');

    % velocity magnitude
    h_fig(12) = figure('Visible','off');
    contourf(rx_mat, rz_mat, u_norm_mat, 40, '-.'); axis equal; grid on; hold on;
    colorbar; ylabel('|{\bf \itu}^*|'); caxis([0 2]);
    xlim([-Length 5*Length]); ylim([-1.5*Length 1.5*Length]);
    xlabel('{\itX}^* position'); ylabel('{\itZ}^* position');
    title('Velocity magnitude distribution');
    plot(X_center_disp(:,i_time_plot), Z_center_disp(:,i_time_plot), 'b-', 'LineWidth', 3);
    colormap jet;
    saveas(gcf, './save/fig/Unorm_distribution_0.png');
    fprintf('  Saved Unorm_distribution_0.png\n');

    %% ===== Fig 13: Velocity field with streamlines =====
    fprintf('Fig 13: Velocity field with streamlines\n');
    h_fig(13) = figure('Visible','off');

    % Use MATLAB built-in stream2 on regular grid (replaces TriStream)
    Vx_plot = reshape(V_xyz(:,1), N_row, N_col);
    Vz_plot = reshape(V_xyz(:,3), N_row, N_col);

    % Starting points for streamlines
    Z0_pos = linspace(-1.5*Length, 1.5*Length, 40);
    X0_pos = (-Length + eps)*ones(size(Z0_pos));

    % Compute streamlines
    streamline_handles = stream2(linspace(-Length, 5*Length, 120), ...
        linspace(-1.5*Length, 1.5*Length, 80), ...
        Vx_plot, Vz_plot, X0_pos, Z0_pos);

    % Plot in 2D (X-Z plane at mid-span)
    hold on;
    for k = 1:length(streamline_handles)
        plot(streamline_handles{k}(:,1), streamline_handles{k}(:,2), 'b-', 'LineWidth', 0.8);
    end

    % Overlay sheet shape
    patch(X, Y, Z, 0, 'CData', h_dp_vec(:,i_wake_time));
    view([0 0 1]); axis equal; grid on;
    xlim([-Length 5*Length]); ylim([-1.5*Length 1.5*Length]);
    xlabel('{\itX}^* [-]'); ylabel('{\itZ}^* [-]');
    title('Velocity field with streamlines');
    colorbar; ylabel('Pressure [-]');
    saveas(gcf, './save/fig/Velocity_field.png');
    fprintf('  Saved Velocity_field.png\n');

else
    fprintf('  Skipped velocity plots (wake data not available)\n');
end

%% ===== Fig 14: Wake visualization =====
fprintf('Fig 14: Wake visualization\n');
if exist('h_r_wake','var')
    h_fig(14) = figure('Visible','off');
    i_wt = length(h_r_wake);
    r_wake = h_r_wake{i_wt};
    N_wake = size(r_wake,1)/4;
    ii_w = 1:N_wake;
    r_w1 = r_wake(ii_w,:);
    r_w2 = r_wake(N_wake+ii_w,:);
    r_w3 = r_wake(2*N_wake+ii_w,:);
    r_w4 = r_wake(3*N_wake+ii_w,:);
    Xw = [r_w1(:,1) r_w2(:,1) r_w3(:,1) r_w4(:,1)].';
    Yw = [r_w1(:,2) r_w2(:,2) r_w3(:,2) r_w4(:,2)].';
    Zw = [r_w1(:,3) r_w2(:,3) r_w3(:,3) r_w4(:,3)].';
    Gamma_w = h_Gamma_wake{i_wt};
    patch(Xw, Yw, Zw, Gamma_w, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    colorbar; caxis([min(Gamma_w) max(Gamma_w)]);
    ylabel('Circulation \Gamma');
    view([1 -2 1]); axis equal; grid on;
    xlabel('{\itX}^* [-]'); ylabel('{\itY}^* [-]'); zlabel('{\itZ}^* [-]');
    title('Wake circulation');
    % Overlay sheet
    r_vec = reshape(h_X_vec(idx_r,end), 3, []);
    X = zeros(4,N_element); Y = zeros(4,N_element); Z = zeros(4,N_element);
    for ii = 1:N_element
        X(:,ii) = r_vec(1,nodes(ii,:));
        Y(:,ii) = r_vec(2,nodes(ii,:));
        Z(:,ii) = r_vec(3,nodes(ii,:));
    end
    hold on; patch(X, Y, Z, 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.9);
    light; lighting gouraud;
    saveas(gcf, './save/fig/wake_snapshot.png');
    fprintf('  Saved wake_snapshot.png\n');
end

%% ===== Fig 15+: Mode shapes =====
fprintf('Generating mode shapes...\n');
if exist('mode_num','var') && exist('Phi_q_mat_BC','var')
    for i_mode = 1:mode_num
        Phi_r_vec = reshape(Phi_q_mat_BC(idx_r,i_mode), 3, []);
        X = zeros(4,N_element); Y = zeros(4,N_element); Z = zeros(4,N_element);
        for ii = 1:N_element
            X(:,ii) = Phi_r_vec(1,nodes(ii,:));
            Y(:,ii) = Phi_r_vec(2,nodes(ii,:));
            Z(:,ii) = Phi_r_vec(3,nodes(ii,:));
        end
        h_fig_mode(i_mode) = figure('Visible','off');
        patch(X, Y, Z, 'r'); light; lighting gouraud;
        view([1 -2 1]); axis equal; grid on;
        xlim([0 Length]); ylim([0 Width]);
        xlabel('{\itX}^* [-]'); ylabel('{\itY}^* [-]'); zlabel('{\itZ}^* [-]');
        title(sprintf('Mode %d: \\omega^*_n = %.3f', i_mode, omega_a(i_mode)));
        saveas(gcf, sprintf('./save/fig/modes/mode_%d.png', i_mode));
    end
    fprintf('  Saved %d mode shapes\n', mode_num);
end

%% ===== Animation =====
fprintf('Generating animation frames...\n');
h_fig_anim = figure('Visible','off');
set(h_fig_anim, 'Position', [100 100 1200 500]);

vidWriter = VideoWriter('./save/fig/animation.mp4', 'MPEG-4');
vidWriter.FrameRate = 30;
open(vidWriter);

anim_snapshot = 50;
i_time = 1;
for time = time_m(time_m <= Snapshot_tmax)
    if mod(i_time, anim_snapshot) == 0
        r_vec = reshape(h_X_vec(idx_r,i_time), 3, []);
        X = zeros(4,N_element); Y = zeros(4,N_element); Z = zeros(4,N_element);
        for ii = 1:N_element
            X(:,ii) = r_vec(1,nodes(ii,:));
            Y(:,ii) = r_vec(2,nodes(ii,:));
            Z(:,ii) = r_vec(3,nodes(ii,:));
        end
        cla;
        patch(X, Y, Z, 'r', 'EdgeColor', 'k', 'FaceAlpha', 0.8);
        view([1 -2 1]); axis equal; grid on; light; lighting gouraud;
        xlim([-Length 5*Length]); ylim([-Width 2*Width]); zlim([-1.5*Length 1.5*Length]);
        xlabel('{\itX}^* [-]'); ylabel('{\itY}^* [-]'); zlabel('{\itZ}^* [-]');
        title(sprintf('Time = %.3f', time));
        drawnow;
        frame = getframe(h_fig_anim);
        writeVideo(vidWriter, frame);
    end
    i_time = i_time + 1;
end
close(vidWriter);
close(h_fig_anim);
fprintf('  Saved animation.mp4\n');

%% ===== CSV exports =====
fprintf('Exporting CSV data...\n');
T1 = table((1:length(omega_a))', omega_a(:), 'VariableNames', {'Mode','Omega'});
writetable(T1, './save/natural_frequencies.csv');

trailing_node = Nx*(Ny+1)+floor(Ny/2)+1;
z_trail = squeeze(h_X_vec(3*(trailing_node-1)+3,:));
T2 = table(time_m(:), z_trail(1:length(time_m))', 'VariableNames', {'Time','Z_tip'});
writetable(T2, './save/tip_displacement.csv');

fprintf('\n=== ALL FIGURES GENERATED ===\n');
fprintf('Output directory: ./save/fig/\n');

close all;
end
