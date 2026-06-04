%% Download all required toolboxes from MATLAB Central File Exchange
% This script opens login prompts as needed.
% Run interactively in MATLAB (not in batch mode).

clc; clear;

toolbox_dir = fullfile(fileparts(mfilename('fullpath')), 'cores', 'ToolBoxes');

%% File Exchange submission IDs
addons = struct( ...
    'Plate_Mesh',     33731, ...
    'mntimes',        47092, ...
    'sparse_sub',     23488, ...
    'TriStream',      11278, ...
    'mmwrite',        15881, ...
    'quiver5',        22351 ...
);

%% Method: download via websave with login
% The download URL for File Exchange submissions:
% https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/{id}/versions/latest/contents.zip

names = fieldnames(addons);
for i = 1:length(names)
    name = names{i};
    id = addons.(name);
    dest_dir = fullfile(toolbox_dir, name);

    % Skip if already populated
    if exist(dest_dir, 'dir') && ~isempty(dir(fullfile(dest_dir, '**', '*.m')))
        fprintf('[SKIP] %s already has .m files\n', name);
        continue;
    end

    zip_file = fullfile(toolbox_dir, [name '.zip']);
    url = sprintf('https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/%d/versions/latest/contents.zip', id);

    fprintf('[DOWN] %s (ID=%d) ...\n', name, id);
    try
        % websave will prompt for login if needed
        websave(zip_file, url);
        fprintf('  Downloaded. Extracting...\n');

        % Extract
        unzip(zip_file, fullfile(toolbox_dir, [name '_tmp']));

        % Move contents to correct directory
        tmp_dir = fullfile(toolbox_dir, [name '_tmp']);
        dirs = dir(tmp_dir);
        dirs = dirs([dirs.isdir] & ~ismember({dirs.name}, {'.', '..'}));

        if ~exist(dest_dir, 'dir')
            mkdir(dest_dir);
        end

        if length(dirs) == 1
            % Single subdirectory - move its contents
            sub = fullfile(tmp_dir, dirs(1).name);
            files = dir(fullfile(sub, '*'));
            for f = 1:length(files)
                if ~ismember(files(f).name, {'.', '..'})
                    movefile(fullfile(sub, files(f).name), fullfile(dest_dir, files(f).name));
                end
            end
        else
            % Multiple items - move all
            files = dir(fullfile(tmp_dir, '*'));
            for f = 1:length(files)
                if ~ismember(files(f).name, {'.', '..'})
                    movefile(fullfile(tmp_dir, files(f).name), fullfile(dest_dir, files(f).name));
                end
            end
        end

        rmdir(tmp_dir, 's');
        delete(zip_file);
        fprintf('  [OK] %s installed\n', name);
    catch ME
        fprintf('  [FAIL] %s: %s\n', name, ME.message);
    end
end

%% Verify
fprintf('\n=== Verification ===\n');
for i = 1:length(names)
    name = names{i};
    mfiles = dir(fullfile(toolbox_dir, name, '**', '*.m'));
    fprintf('%s: %d .m files found\n', name, length(mfiles));
end

fprintf('\nDone! Now run run_sim.m\n');
