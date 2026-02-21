function S = VecToso3(omega)
%VecToso3 Convert 3-vector to 3x3 skew-symmetric so(3) matrix.
%   S = rotation_converter.VecToso3(OMEGA)
%   OMEGA: 3-element angular velocity vector.
%   Returns: 3x3 skew-symmetric matrix.
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 3.30.

    omega = omega(:)';
    rotation_converter.internal.require(numel(omega) == 3, ...
        'omega must have 3 elements', numel(omega));
    S = rotation_converter.internal.skew_symmetric(omega);
end
