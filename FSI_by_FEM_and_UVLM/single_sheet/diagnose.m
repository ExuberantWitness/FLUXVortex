function diagnose
cd('E:\DATA\vscode\FSI_by_FEM_and_UVLM\single_sheet');
add_pathes;
param_setting;

fprintf('=== FSI Diagnostic ===\n');

generate_shape_function;
generate_elements;
generate_panel;
generate_matrices;

n1 = double(any(isnan(full(M_global))));
fprintf('M_global NaN: %d\n', n1);

generate_Qf_time_mat;
initial_values;

q_vec = h_X_vec(1:N_q_all,1);
dt_q_vec = 0*q_vec;
flag_output = 1;
theta_a_tmp = theta_a;
theta_a = 1;
generate_stiff_matrices;
theta_a = theta_a_tmp;

n2 = double(any(isnan(Qe_global)));
n3 = double(any(isnan(Qk_global)));
n4 = double(any(isnan(full(dq_Qe_global))));
n5 = double(any(isnan(full(Qd_global))));
fprintf('Qe NaN=%d, Qk NaN=%d, dqQe NaN=%d, Qd NaN=%d\n', n2, n3, n4, n5);

dq_e_Dp_dq_e_global = sparse(N_q_all,N_q_all);
for ii = 1:N_element
    iv = repmat((N_qi*(nodes(ii,:)-1)+1).',[1 N_qi])+repmat(0:N_qi-1,[length(nodes(ii,:)) 1]);
    iv = reshape(iv.',1,[]);
    dq_e_Dp_dq_e_global(iv,iv) = dq_e_Dp_dq_e_global(iv,iv)+squeeze(zeta_m*Qd_eps_mat_i(:,:,ii));
end
K_e_m_mat = dq_e_Dp_dq_e_global;

dq_k_Dp_dq_k_global = sparse(N_q_all,N_q_all);
for ii = 1:N_element
    iv = repmat((N_qi*(nodes(ii,:)-1)+1).',[1 N_qi])+repmat(0:N_qi-1,[length(nodes(ii,:)) 1]);
    iv = reshape(iv.',1,[]);
    dq_k_Dp_dq_k_global(iv,iv) = dq_k_Dp_dq_k_global(iv,iv)+squeeze(eta_m*Qd_k_mat_i(:,:,ii));
end
K_e_k_mat = dq_k_Dp_dq_k_global;

n6 = double(any(isnan(full(K_e_m_mat))));
n7 = double(any(isnan(full(K_e_k_mat))));
fprintf('Km NaN=%d, Kk NaN=%d\n', n6, n7);

i_r = reshape((repmat((N_qi*(node_r_0-1)+1).',[1 3])+repmat(0:2,[length(node_r_0) 1])).',1,[]);
i_dx = reshape((repmat((N_qi*(node_dxr_0-1)+4).',[1 3])+repmat(0:2,[length(node_dxr_0) 1])).',1,[]);
i_dy = reshape((repmat((N_qi*(node_dyr_0-1)+7).',[1 3])+repmat(0:2,[length(node_dyr_0) 1])).',1,[]);
ibc = [i_r i_dx i_dy];

M_BC = M_global; M_BC(ibc,:)=[]; M_BC(:,ibc)=[];
Km_BC = K_e_m_mat; Km_BC(ibc,:)=[]; Km_BC(:,ibc)=[];
Kk_BC = K_e_k_mat; Kk_BC(ibc,:)=[]; Kk_BC(:,ibc)=[];

c_M = condest(M_BC);
n8 = double(any(isnan(full(M_BC))));
fprintf('M_BC cond=%.4e NaN=%d size=%dx%d\n', c_M, n8, size(M_BC,1), size(M_BC,2));

K_tot = Km_BC + Kk_BC;
n9 = double(any(isnan(full(K_tot))));
fprintf('K_tot NaN=%d nnz=%d\n', n9, nnz(K_tot));

fprintf('Trying eigs...\n');
try
    A = mu_m*M_BC \ K_tot;
    nA = double(any(isnan(full(A))));
    fprintf('A=M\\K NaN=%d\n', nA);
    [Phi, w2] = eigs(A, mode_num, 'SM');
    w = sqrt(diag(w2));
    fprintf('SUCCESS omega=%s\n', mat2str(w'));
catch ME
    fprintf('FAILED: %s\n', ME.message);
end
fprintf('=== Done ===\n');
end
