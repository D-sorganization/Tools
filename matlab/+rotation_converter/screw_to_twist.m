function xi = screw_to_twist(screw)
%SCREW_TO_TWIST Convert screw axis parameters to twist vector.
%   xi = rotation_converter.screw_to_twist(SCREW)
%   SCREW: struct with fields .axis, .point, .pitch.
%   Returns: 6-element twist vector [omega; v].

    axis = screw.axis(:)';
    point = screw.point(:)';
    pitch = screw.pitch;

    rotation_converter.internal.require(numel(axis) == 3, ...
        'screw axis must have 3 elements');

    if isinf(pitch)
        % Pure translation: normalize axis to unit direction
        axis_norm = norm(axis);
        rotation_converter.internal.require(axis_norm > 1e-12, ...
            'screw axis must be non-zero for pure translation');
        omega = [0, 0, 0];
        v = axis / axis_norm;
    else
        % Rotation/helical: axis must be unit
        rotation_converter.internal.require_unit_vector(axis, 'screw axis');
        omega = axis;
        v = cross(-omega, point) + pitch * omega;
    end

    xi = [omega, v];
    rotation_converter.internal.ensure(numel(xi) == 6, ...
        'result must have 6 elements');
end
