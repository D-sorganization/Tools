%LAUNCH_ROTATION_CONVERTER Entry script for MATLAB/Octave rotation converter.
%
% This script prepares paths and runs a short smoke demonstration.

root_dir = fileparts(mfilename("fullpath"));
addpath(root_dir);

is_octave = exist("OCTAVE_VERSION", "builtin") ~= 0;
runtime_name = "MATLAB";
if is_octave
    runtime_name = "GNU Octave";
end

fprintf("Rotation Converter launcher (%s)\n", runtime_name);
fprintf("Package root: %s\n", root_dir);

q = rotation_converter.normalize_quaternion([1, 2, 3, 4]);
r = rotation_converter.quaternion_to_rotation_matrix(q);
fprintf("Demo quaternion (wxyz): [%.6f %.6f %.6f %.6f]\n", q);
fprintf("Demo rotation matrix:\n");
disp(r);

fprintf("\nTip: run test_rotation_converter for full validation.\n");
