function se3mat = VecTose3(V)
%VecTose3 Convert 6-vector twist to 4x4 se(3) matrix.
%   se3mat = rotation_converter.VecTose3(V)
%   V: [omega1 omega2 omega3 v1 v2 v3] spatial velocity 6-vector.
%   Returns: 4x4 se(3) matrix.
%
%   Reference: Lynch & Park, Modern Robotics, Eq. 3.63.

    V = V(:)';
    rotation_converter.internal.require(numel(V) == 6, ...
        'twist must have 6 elements', numel(V));

    se3mat = [rotation_converter.VecToso3(V(1:3)), V(4:6)'; ...
              0, 0, 0, 0];
end
