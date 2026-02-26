function q_interp =
    slerp(q1, q2, t) % SLERP Spherical Linear Interpolation(slerp)
between two unit quaternions.% % q_interp =
    slerp(q1, q2, t) % % Args : % q1 : Unit quaternion(w, x, y, z)
at t = 0. % q2 : Unit quaternion(w, x, y, z)
at t = 1. % t : Interpolation parameter in range[0, 1].% %
               Returns : %
                         Interpolated unit quaternion(w, x, y, z)
                             .

                         if t <
           0.0 ||
       t > 1.0 error('Interpolation parameter t must be in [0, 1]');
end

    q1 = q1( :)'; q2 = q2( :)';

    dot_val = dot(q1, q2);

% If the dot product is negative,
    slerp takes the long way around.%
            We negate q2 to take the shortest path instead.if dot_val <
        0.0 q2 = -q2;
dot_val = -dot_val;
end

    % If the inputs are too close,
    linearly interpolate and normalize % to avoid division by zero if dot_val >
        0.9995 result = q1 + t * (q2 - q1);
q_interp = rotation_converter.normalize_quaternion(result);
return;
end

    % Standard slerp theta_0 = acos(dot_val);
theta = theta_0 * t;
sin_theta = sin(theta);
sin_theta_0 = sin(theta_0);

s1 = cos(theta) - dot_val * sin_theta / sin_theta_0;
s2 = sin_theta / sin_theta_0;

q_interp = rotation_converter.normalize_quaternion((s1 * q1) + (s2 * q2));
end
