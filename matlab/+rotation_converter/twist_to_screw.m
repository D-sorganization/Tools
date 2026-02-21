function screw = twist_to_screw(xi)
%TWIST_TO_SCREW Decompose a twist into screw axis parameters.
%   screw = rotation_converter.twist_to_screw(XI)
%   XI: 6-element twist vector [omega; v].
%   Returns: struct with fields:
%     .axis  - 3-element direction vector (unit)
%     .point - 3-element point on screw axis
%     .pitch - scalar (0 for pure rotation, inf for pure translation)
%
%   Precondition: xi must be non-zero.

    xi = xi(:)';
    rotation_converter.internal.require(numel(xi) == 6, ...
        'twist must have 6 elements');

    omega = xi(1:3);
    v = xi(4:6);
    omega_norm = norm(omega);

    rotation_converter.internal.require(omega_norm > 1e-12 || norm(v) > 1e-12, ...
        'twist must be non-zero');

    if omega_norm < 1e-12
        % Pure translation
        screw.axis = v / norm(v);
        screw.point = [0, 0, 0];
        screw.pitch = inf;
    else
        rotation_converter.internal.require(abs(omega_norm - 1.0) < 1e-6, ...
            'omega must be unit when non-zero', omega_norm);
        % point = omega x v (closest point on axis to origin)
        screw.axis = omega;
        screw.point = cross(omega, v);
        screw.pitch = dot(omega, v);
    end
end
