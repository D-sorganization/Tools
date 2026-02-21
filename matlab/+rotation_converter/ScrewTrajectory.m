function trajectory = ScrewTrajectory(Xstart, Xend, Tf, N, method)
%ScrewTrajectory Generate a screw-motion trajectory between two SE(3) poses.
%   trajectory = rotation_converter.ScrewTrajectory(XSTART, XEND, TF, N)
%   trajectory = rotation_converter.ScrewTrajectory(XSTART, XEND, TF, N, METHOD)
%
%   XSTART: 4x4 SE(3) start pose.
%   XEND: 4x4 SE(3) end pose.
%   TF: Total time of motion.
%   N: Number of points in the trajectory.
%   METHOD: 3 for cubic, 5 for quintic time scaling (default 3).
%
%   Returns: cell array of N 4x4 SE(3) matrices.
%
%   Reference: Lynch & Park, Modern Robotics, Section 9.4.

    if nargin < 5; method = 3; end

    rotation_converter.internal.require(all(size(Xstart) == [4, 4]), ...
        'Xstart must be 4x4');
    rotation_converter.internal.require(all(size(Xend) == [4, 4]), ...
        'Xend must be 4x4');
    rotation_converter.internal.require(Tf > 0, 'Tf must be positive');
    rotation_converter.internal.require(N >= 2, 'N must be >= 2');
    rotation_converter.internal.require(method == 3 || method == 5, ...
        'method must be 3 or 5', method);

    timegap = Tf / (N - 1);
    se3_diff = rotation_converter.MatrixLog6( ...
        rotation_converter.TransInv(Xstart) * Xend);

    trajectory = cell(1, N);
    for i = 1:N
        t_norm = ((i - 1) * timegap) / Tf;

        if method == 3
            s = 3 * t_norm^2 - 2 * t_norm^3;
        else
            s = 10 * t_norm^3 - 15 * t_norm^4 + 6 * t_norm^5;
        end

        trajectory{i} = Xstart * rotation_converter.MatrixExp6(se3_diff * s);
    end
end
