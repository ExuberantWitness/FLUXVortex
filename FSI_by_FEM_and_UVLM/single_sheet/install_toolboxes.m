%% Install all required toolboxes via matlab.addons.install
% Run this in MATLAB GUI (not batch mode)
clc; clear;

%% First, remove my stub implementations so real ones get installed
stub_dirs = {
    fullfile('cores', 'ToolBoxes', 'Plate_Mesh')
    fullfile('cores', 'ToolBoxes', 'mntimes')
};

for i = 1:length(stub_dirs)
    d = stub_dirs{i};
    if exist(d, 'dir')
        rmdir(d, 's');
        fprintf('Removed stub: %s\n', d);
    end
end

%% Install from File Exchange using matlab.addons.install
% This will prompt for login if needed

addon_urls = {
    'https://www.mathworks.com/matlabcentral/fileexchange/33731-meshing-a-plate-using-four-noded-elements'
    'https://www.mathworks.com/matlabcentral/fileexchange/47092-vectorized-multi-dimensional-matrix-multiplication'
    'https://www.mathworks.com/matlabcentral/fileexchange/23488-sparse-sub-access'
    'https://www.mathworks.com/matlabcentral/fileexchange/11278-tristream'
    'https://www.mathworks.com/matlabcentral/fileexchange/15881-mmwrite'
    'https://www.mathworks.com/matlabcentral/fileexchange/22351-quiver-5'
};

for i = 1:length(addon_urls)
    fprintf('Installing: %s\n', addon_urls{i});
    try
        matlab.addons.install(addon_urls{i}, true);  % true = accept license
        fprintf('  [OK]\n');
    catch ME
        fprintf('  [FAIL] %s\n', ME.message);
    end
end

fprintf('\nDone! Check installed add-ons with: matlab.addons.installedAddons\n');
