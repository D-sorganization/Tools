function screw_data = extract_screw_axes_from_trajectory(trajectory)
%EXTRACT_SCREW_AXES_FROM_TRAJECTORY Extract screw axis info between consecutive frames.
%   screw_data = rotation_converter.extract_screw_axes_from_trajectory(TRAJECTORY)
%
%   TRAJECTORY: cell array of 4x4 SE(3) matrices.
%
%   Returns: struct array (one per pair of consecutive frames) with fields:
%     .axis      - 3-element screw axis direction
%     .point     - 3-element point on screw axis
%     .pitch     - scalar pitch
%     .theta     - scalar angle
%     .midpoint  - 3-element midpoint between consecutive origins

    n = numel(trajectory);
    screw_data = struct('axis', {}, 'point', {}, 'pitch', {}, ...
                        'theta', {}, 'midpoint', {});

    for i = 1:(n-1)
        T_rel = rotation_converter.TransInv(trajectory{i}) * trajectory{i+1};

        [xi, theta] = rotation_converter.homogeneous_to_twist_angle(T_rel);

        omega_norm = norm(xi(1:3));
        v_norm = norm(xi(4:6));

        if omega_norm < 1e-12 && v_norm < 1e-12
            % No motion
            screw_data(i).axis = [0, 0, 1];
            screw_data(i).point = [0, 0, 0];
            screw_data(i).pitch = 0;
            screw_data(i).theta = 0;
        elseif omega_norm < 1e-12
            % Pure translation
            screw_data(i).axis = xi(4:6) / v_norm;
            screw_data(i).point = [0, 0, 0];
            screw_data(i).pitch = inf;
            screw_data(i).theta = theta;
        else
            s = rotation_converter.twist_to_screw(xi);
            screw_data(i).axis = s.axis;
            screw_data(i).point = s.point;
            screw_data(i).pitch = s.pitch;
            screw_data(i).theta = theta;
        end

        % World-frame midpoint
        p1 = trajectory{i}(1:3, 4)';
        p2 = trajectory{i+1}(1:3, 4)';
        screw_data(i).midpoint = (p1 + p2) / 2;

        % Transform axis to world frame
        R = trajectory{i}(1:3, 1:3);
        screw_data(i).axis = (R * screw_data(i).axis(:))';
        screw_data(i).point = (R * screw_data(i).point(:) + trajectory{i}(1:3, 4))';
    end
end
