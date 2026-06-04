function diagnose2
cd('E:\DATA\vscode\FSI_by_FEM_and_UVLM\single_sheet');
add_pathes;
param_setting;

fprintf('=== NaN Source Diagnosis ===\n');

generate_shape_function;

% Test shape function at a point
dL = 1/15; dW = 1/10;
S_test = Sc_mat(dL/2, dW/2, dL, dW);
fprintf('Sc_mat test: size=[%s], NaN=%d\n', num2str(size(S_test)), double(any(isnan(S_test(:)))));

generate_elements;
fprintf('coordinates(1:5,:):\n');
disp(coordinates(1:5,:));
fprintf('nodes(1:3,:):\n');
disp(nodes(1:3,:));

% Test M_mat_i for element 1
[p_vec, w_vec] = Gauss(N_gauss);
dL1 = dL_vec(1); dW1 = dW_vec(1);
fprintf('dL_vec(1)=%.4f, dW_vec(1)=%.4f\n', dL1, dW1);

int_StS = zeros(N_q);
for ix = 1:length(p_vec)
    for iy = 1:length(p_vec)
        xi = p_vec(ix); eta = p_vec(iy);
        x = dL1*(xi+1)/2; y = dW1*(eta+1)/2;
        S = Sc_mat(x, y, dL1, dW1);
        if any(isnan(S(:)))
            fprintf('NaN in Sc_mat at x=%.4f y=%.4f\n', x, y);
        end
        StS = S'*S;
        int_StS = int_StS + dL1*dW1/4*w_vec(ix)*w_vec(iy)*StS;
    end
end
fprintf('M_mat element 1: NaN=%d, min=%.4e, max=%.4e\n', ...
    double(any(isnan(int_StS(:)))), min(int_StS(:)), max(int_StS(:)));

% Now run full generate_matrices
generate_panel;

% Manually run generate_matrices and check
Sc_mat_v = zeros(3,N_q,length(p_vec),length(p_vec),N_element);
for ii = 1:min(5,N_element)
    dL = dL_vec(ii); dW = dW_vec(ii);
    ix = 1;
    for xi = p_vec
        iy = 1;
        for eta = p_vec
            x = dL*(xi+1)/2; y = dW*(eta+1)/2;
            S = Sc_mat(x, y, dL, dW);
            Sc_mat_v(:,:,ix,iy,ii) = S;
            iy = iy+1;
        end
        ix = ix+1;
    end
    has_nan = double(any(isnan(Sc_mat_v(:,:,:,:,ii))));
    fprintf('Element %d Sc_mat_v NaN=%d\n', ii, has_nan);
end

% Full matrices
generate_matrices;
n1 = double(any(isnan(full(M_global))));
fprintf('\nFull M_global NaN=%d\n', n1);
if n1
    % Find which DOFs have NaN in M_global
    [ri,ci] = find(isnan(M_global));
    fprintf('NaN entries: %d, first rows: %s\n', length(ri), num2str(ri(1:min(10,length(ri)))));
    fprintf('NaN cols: %s\n', num2str(ci(1:min(10,length(ci)))));
    % What nodes do these correspond to?
    node1 = ceil(ri(1)/N_qi);
    fprintf('First NaN at node %d (of %d total), coord: (%.4f, %.4f)\n', ...
        node1, size(coordinates,1), coordinates(node1,1), coordinates(node1,2));
end
fprintf('=== Done ===\n');
end
