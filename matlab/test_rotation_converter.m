function test_rotation_converter()
%TEST_ROTATION_CONVERTER Comprehensive test suite for the rotation_converter package.
%
%   Run: test_rotation_converter()
%
%   Compatible with MATLAB and GNU Octave.

    fprintf('=== Rotation Converter MATLAB/Octave Test Suite ===\n\n');
    n_pass = 0; n_fail = 0;

    % ========================================
    % 1. Quaternion primitives
    % ========================================
    fprintf('--- Quaternion Primitives ---\n');

    % normalize_quaternion
    q = rotation_converter.normalize_quaternion([0, 0, 0, 2]);
    [n_pass, n_fail] = check(abs(norm(q) - 1) < 1e-12, 'normalize_quaternion: unit norm', n_pass, n_fail);
    [n_pass, n_fail] = check(q(1) >= 0, 'normalize_quaternion: canonical w>=0', n_pass, n_fail);

    % quaternion_conjugate
    q = [1, 2, 3, 4] / norm([1, 2, 3, 4]);
    qc = rotation_converter.quaternion_conjugate(q);
    [n_pass, n_fail] = check(all(abs(qc - [q(1), -q(2), -q(3), -q(4)]) < 1e-12), ...
        'quaternion_conjugate', n_pass, n_fail);

    % quaternion_multiply identity
    q_id = [1, 0, 0, 0];
    q_test = rotation_converter.normalize_quaternion([1, 2, 3, 4]);
    qm = rotation_converter.quaternion_multiply(q_id, q_test);
    [n_pass, n_fail] = check(all(abs(qm - q_test) < 1e-12), ...
        'quaternion_multiply: identity', n_pass, n_fail);

    % quaternion_multiply inverse
    qc = rotation_converter.quaternion_conjugate(q_test);
    qi = rotation_converter.quaternion_multiply(q_test, qc);
    [n_pass, n_fail] = check(abs(qi(1) - 1) < 1e-9 && norm(qi(2:4)) < 1e-9, ...
        'quaternion_multiply: inverse gives identity', n_pass, n_fail);

    % ========================================
    % 2. Quaternion <-> Rotation Matrix
    % ========================================
    fprintf('--- Quaternion <-> Rotation Matrix ---\n');

    % Known 90-degree rotation about Z
    q_z90 = rotation_converter.normalize_quaternion([cos(pi/4), 0, 0, sin(pi/4)]);
    R = rotation_converter.quaternion_to_rotation_matrix(q_z90);
    R_expected = [0 -1 0; 1 0 0; 0 0 1];
    [n_pass, n_fail] = check(max(abs(R(:) - R_expected(:))) < 1e-9, ...
        'quat_to_rotmat: 90deg Z', n_pass, n_fail);

    % Roundtrip
    q_back = rotation_converter.rotation_matrix_to_quaternion(R);
    [n_pass, n_fail] = check(abs(abs(dot(q_z90, q_back)) - 1) < 1e-9, ...
        'quat<->rotmat roundtrip', n_pass, n_fail);

    % Identity
    R_id = rotation_converter.quaternion_to_rotation_matrix([1 0 0 0]);
    I3 = eye(3);
    [n_pass, n_fail] = check(max(abs(R_id(:) - I3(:))) < 1e-12, ...
        'quat_to_rotmat: identity', n_pass, n_fail);

    % Shepperd branches: 180-degree rotations
    axes_180 = {[1 0 0], [0 1 0], [0 0 1], [1 1 1]/sqrt(3)};
    for k = 1:4
        R180 = rotation_converter.axis_angle_to_rotation_matrix(axes_180{k}, pi);
        q180 = rotation_converter.rotation_matrix_to_quaternion(R180);
        R180_back = rotation_converter.quaternion_to_rotation_matrix(q180);
        [n_pass, n_fail] = check(max(abs(R180(:) - R180_back(:))) < 1e-9, ...
            sprintf('Shepperd 180deg axis %d', k), n_pass, n_fail);
    end

    % Random roundtrips
    rng_state = rng(42);
    for trial = 1:20
        q_rand = randn(1, 4);
        q_rand = rotation_converter.normalize_quaternion(q_rand);
        R_rand = rotation_converter.quaternion_to_rotation_matrix(q_rand);
        q_roundtrip = rotation_converter.rotation_matrix_to_quaternion(R_rand);
        [n_pass, n_fail] = check(abs(abs(dot(q_rand, q_roundtrip)) - 1) < 1e-9, ...
            sprintf('quat<->rotmat random %d', trial), n_pass, n_fail);
    end

    % ========================================
    % 3. Axis-Angle <-> Quaternion
    % ========================================
    fprintf('--- Axis-Angle <-> Quaternion ---\n');

    [ax, ang] = rotation_converter.quaternion_to_axis_angle(q_z90);
    [n_pass, n_fail] = check(abs(ang - pi/2) < 1e-9, ...
        'quat_to_axisangle: angle', n_pass, n_fail);
    [n_pass, n_fail] = check(abs(ax(3) - 1) < 1e-9, ...
        'quat_to_axisangle: axis', n_pass, n_fail);

    % Zero rotation
    [ax0, ang0] = rotation_converter.quaternion_to_axis_angle([1 0 0 0]);
    [n_pass, n_fail] = check(ang0 == 0, 'zero rotation angle', n_pass, n_fail);

    % Roundtrip
    for trial = 1:10
        ax_rand = randn(1, 3); ax_rand = ax_rand / norm(ax_rand);
        ang_rand = rand * pi;
        q_aa = rotation_converter.axis_angle_to_quaternion(ax_rand, ang_rand);
        [ax2, ang2] = rotation_converter.quaternion_to_axis_angle(q_aa);
        q_aa2 = rotation_converter.axis_angle_to_quaternion(ax2, ang2);
        [n_pass, n_fail] = check(abs(abs(dot(q_aa, q_aa2)) - 1) < 1e-9, ...
            sprintf('axis-angle roundtrip %d', trial), n_pass, n_fail);
    end

    % ========================================
    % 4. Rodrigues <-> Quaternion
    % ========================================
    fprintf('--- Rodrigues <-> Quaternion ---\n');

    r_vec = rotation_converter.quaternion_to_rodrigues(q_z90);
    [n_pass, n_fail] = check(abs(norm(r_vec) - pi/2) < 1e-9, ...
        'rodrigues magnitude', n_pass, n_fail);

    q_rod_back = rotation_converter.rodrigues_to_quaternion(r_vec);
    [n_pass, n_fail] = check(abs(abs(dot(q_z90, q_rod_back)) - 1) < 1e-9, ...
        'rodrigues roundtrip', n_pass, n_fail);

    % Zero vector
    q_zero = rotation_converter.rodrigues_to_quaternion([0 0 0]);
    [n_pass, n_fail] = check(abs(q_zero(1) - 1) < 1e-12, ...
        'rodrigues zero -> identity', n_pass, n_fail);

    % ========================================
    % 5. Euler Angles (all 12 conventions)
    % ========================================
    fprintf('--- Euler Angles ---\n');

    conventions = {'xyz','xzy','yxz','yzx','zxy','zyx', ...
                   'xyx','xzx','yxy','yzy','zxz','zyz'};

    for ci = 1:12
        conv = conventions{ci};
        % Forward-inverse roundtrip through quaternion
        a = 0.3; b = 0.7; c = -0.5;
        q_euler = rotation_converter.euler_to_quaternion(a, b, c, conv);
        [a2, b2, c2] = rotation_converter.quaternion_to_euler(q_euler, conv);
        q_euler2 = rotation_converter.euler_to_quaternion(a2, b2, c2, conv);
        [n_pass, n_fail] = check(abs(abs(dot(q_euler, q_euler2)) - 1) < 1e-9, ...
            sprintf('euler roundtrip %s', conv), n_pass, n_fail);
    end

    % Euler -> rotation matrix -> Euler roundtrip
    for ci = 1:12
        conv = conventions{ci};
        R_e = rotation_converter.euler_to_rotation_matrix(0.4, 0.8, -0.3, conv);
        [a3, b3, c3] = rotation_converter.rotation_matrix_to_euler(R_e, conv);
        R_e2 = rotation_converter.euler_to_rotation_matrix(a3, b3, c3, conv);
        [n_pass, n_fail] = check(max(abs(R_e(:) - R_e2(:))) < 1e-9, ...
            sprintf('euler<->rotmat %s', conv), n_pass, n_fail);
    end

    % Gimbal lock: Tait-Bryan at +/- pi/2
    tb_convs = {'xyz','xzy','yxz','yzx','zxy','zyx'};
    for ci = 1:6
        conv = tb_convs{ci};
        for b_sign = [1, -1]
            b_val = b_sign * pi / 2;
            R_gl = rotation_converter.euler_to_rotation_matrix(0.3, b_val, 0.0, conv);
            [ag, bg, cg] = rotation_converter.rotation_matrix_to_euler(R_gl, conv);
            R_gl2 = rotation_converter.euler_to_rotation_matrix(ag, bg, cg, conv);
            [n_pass, n_fail] = check(max(abs(R_gl(:) - R_gl2(:))) < 1e-7, ...
                sprintf('gimbal lock TB %s b=%+.2f', conv, b_val), n_pass, n_fail);
        end
    end

    % Gimbal lock: proper Euler at b=0 and b=pi
    pe_convs = {'xyx','xzx','yxy','yzy','zxz','zyz'};
    for ci = 1:6
        conv = pe_convs{ci};
        for b_val = [0, pi]
            R_gl = rotation_converter.euler_to_rotation_matrix(0.7, b_val, 0.0, conv);
            [ag, bg, cg] = rotation_converter.rotation_matrix_to_euler(R_gl, conv);
            R_gl2 = rotation_converter.euler_to_rotation_matrix(ag, bg, cg, conv);
            [n_pass, n_fail] = check(max(abs(R_gl(:) - R_gl2(:))) < 1e-7, ...
                sprintf('gimbal lock PE %s b=%.2f', conv, b_val), n_pass, n_fail);
        end
    end

    % ========================================
    % 6. Twist / Screw conversions
    % ========================================
    fprintf('--- Twist / Screw ---\n');

    % twist_vector <-> se3_matrix roundtrip
    xi_test = [0 0 1 1 0 0];
    M_se3 = rotation_converter.twist_vector_to_se3_matrix(xi_test);
    xi_back = rotation_converter.se3_matrix_to_twist_vector(M_se3);
    [n_pass, n_fail] = check(max(abs(xi_test - xi_back)) < 1e-12, ...
        'twist<->se3 roundtrip', n_pass, n_fail);

    % twist_angle <-> homogeneous roundtrip
    xi_rot = [0 0 1 0.5 -0.3 0.1];
    T_twist = rotation_converter.twist_angle_to_homogeneous(xi_rot, 1.2);
    [xi2, theta2] = rotation_converter.homogeneous_to_twist_angle(T_twist);
    T_twist2 = rotation_converter.twist_angle_to_homogeneous(xi2, theta2);
    [n_pass, n_fail] = check(max(abs(T_twist(:) - T_twist2(:))) < 1e-9, ...
        'twist+angle<->SE3 roundtrip', n_pass, n_fail);

    % Pure translation twist
    xi_trans = [0 0 0 1 0 0];
    T_trans = rotation_converter.twist_angle_to_homogeneous(xi_trans, 3.0);
    [n_pass, n_fail] = check(abs(T_trans(1, 4) - 3.0) < 1e-12, ...
        'pure translation twist', n_pass, n_fail);

    % Identity transform
    [xi_id, th_id] = rotation_converter.homogeneous_to_twist_angle(eye(4));
    [n_pass, n_fail] = check(th_id == 0, 'identity twist angle', n_pass, n_fail);

    % twist <-> screw roundtrip
    xi_screw = [0 0 1 0.5 -0.3 0.1];
    s = rotation_converter.twist_to_screw(xi_screw);
    xi_from_screw = rotation_converter.screw_to_twist(s);
    [n_pass, n_fail] = check(max(abs(xi_screw - xi_from_screw)) < 1e-9, ...
        'twist<->screw roundtrip', n_pass, n_fail);

    % Pure translation screw
    xi_pure_t = [0 0 0 0 0 1];
    s_t = rotation_converter.twist_to_screw(xi_pure_t);
    [n_pass, n_fail] = check(isinf(s_t.pitch), 'pure translation pitch=inf', n_pass, n_fail);

    % adjoint_representation
    T_adj = rotation_converter.twist_angle_to_homogeneous(xi_rot, 0.8);
    Ad = rotation_converter.adjoint_representation(T_adj);
    [n_pass, n_fail] = check(all(size(Ad) == [6 6]), 'adjoint 6x6', n_pass, n_fail);

    % Random twist roundtrips
    for trial = 1:10
        omega_r = randn(1, 3); omega_r = omega_r / norm(omega_r);
        v_r = randn(1, 3);
        xi_r = [omega_r, v_r];
        theta_r = rand * 2;
        T_r = rotation_converter.twist_angle_to_homogeneous(xi_r, theta_r);
        [xi_r2, theta_r2] = rotation_converter.homogeneous_to_twist_angle(T_r);
        T_r2 = rotation_converter.twist_angle_to_homogeneous(xi_r2, theta_r2);
        [n_pass, n_fail] = check(max(abs(T_r(:) - T_r2(:))) < 1e-8, ...
            sprintf('twist SE3 random roundtrip %d', trial), n_pass, n_fail);
    end

    % ========================================
    % 7. Modern Robotics: SO(3)/SE(3)
    % ========================================
    fprintf('--- Modern Robotics: SO(3)/SE(3) ---\n');

    % VecToso3 / so3ToVec roundtrip
    omega_test = [1 2 3];
    S = rotation_converter.VecToso3(omega_test);
    omega_back = rotation_converter.so3ToVec(S);
    [n_pass, n_fail] = check(max(abs(omega_test - omega_back)) < 1e-12, ...
        'VecToso3<->so3ToVec', n_pass, n_fail);

    % MatrixExp3 / MatrixLog3 roundtrip
    omega_unit = [0 0 1];
    so3 = rotation_converter.VecToso3(omega_unit) * (pi/3);
    R_exp = rotation_converter.MatrixExp3(so3);
    so3_back = rotation_converter.MatrixLog3(R_exp);
    R_exp2 = rotation_converter.MatrixExp3(so3_back);
    [n_pass, n_fail] = check(max(abs(R_exp(:) - R_exp2(:))) < 1e-9, ...
        'MatrixExp3<->MatrixLog3', n_pass, n_fail);

    % MatrixLog3 identity
    so3_id = rotation_converter.MatrixLog3(eye(3));
    [n_pass, n_fail] = check(max(abs(so3_id(:))) < 1e-12, ...
        'MatrixLog3 identity', n_pass, n_fail);

    % MatrixLog3 pi-rotation about all axes
    for ax_idx = 1:3
        R_pi = -eye(3); R_pi(ax_idx, ax_idx) = 1;
        so3_pi = rotation_converter.MatrixLog3(R_pi);
        R_pi_back = rotation_converter.MatrixExp3(so3_pi);
        [n_pass, n_fail] = check(max(abs(R_pi(:) - R_pi_back(:))) < 1e-9, ...
            sprintf('MatrixLog3 pi-rot axis %d', ax_idx), n_pass, n_fail);
    end

    % MatrixExp6 / MatrixLog6 roundtrip
    xi6 = [0 0 1 1 0 0];
    se3_test = rotation_converter.VecTose3(xi6) * 0.5;
    T_exp = rotation_converter.MatrixExp6(se3_test);
    se3_back = rotation_converter.MatrixLog6(T_exp);
    T_exp2 = rotation_converter.MatrixExp6(se3_back);
    [n_pass, n_fail] = check(max(abs(T_exp(:) - T_exp2(:))) < 1e-9, ...
        'MatrixExp6<->MatrixLog6', n_pass, n_fail);

    % MatrixLog6 identity
    se3_id = rotation_converter.MatrixLog6(eye(4));
    [n_pass, n_fail] = check(max(abs(se3_id(:))) < 1e-12, ...
        'MatrixLog6 identity', n_pass, n_fail);

    % TransToRp / RpToTrans roundtrip
    R_test = rotation_converter.quaternion_to_rotation_matrix( ...
        rotation_converter.normalize_quaternion([1 1 0 0]));
    p_test = [1; 2; 3];
    T_rp = rotation_converter.RpToTrans(R_test, p_test);
    [R_back, p_back] = rotation_converter.TransToRp(T_rp);
    [n_pass, n_fail] = check(max(abs(R_test(:) - R_back(:))) < 1e-12, ...
        'TransToRp/RpToTrans R', n_pass, n_fail);
    [n_pass, n_fail] = check(max(abs(p_test - p_back)) < 1e-12, ...
        'TransToRp/RpToTrans p', n_pass, n_fail);

    % TransInv roundtrip
    T_inv = rotation_converter.TransInv(T_rp);
    prod = T_rp * T_inv;
    I4 = eye(4);
    [n_pass, n_fail] = check(max(abs(prod(:) - I4(:))) < 1e-9, ...
        'TransInv roundtrip', n_pass, n_fail);

    % Random MatrixExp3/Log3 roundtrips
    for trial = 1:10
        om = randn(1, 3); om = om / norm(om);
        th = rand * 2.5;
        so3_r = rotation_converter.VecToso3(om) * th;
        R_r = rotation_converter.MatrixExp3(so3_r);
        so3_r2 = rotation_converter.MatrixLog3(R_r);
        R_r2 = rotation_converter.MatrixExp3(so3_r2);
        [n_pass, n_fail] = check(max(abs(R_r(:) - R_r2(:))) < 1e-9, ...
            sprintf('Exp3/Log3 random %d', trial), n_pass, n_fail);
    end

    % ========================================
    % 8. Forward / Inverse Kinematics
    % ========================================
    fprintf('--- Forward / Inverse Kinematics ---\n');

    % 3R planar robot
    M_fk = eye(4); M_fk(1, 4) = 3;  % end-effector at (3,0,0)
    Slist = [0 0 1 0 0 0;
             0 0 1 0 -1 0;
             0 0 1 0 -2 0]';
    Blist = [0 0 1 0 3 0;
             0 0 1 0 2 0;
             0 0 1 0 1 0]';

    theta_fk = [0 0 0];
    T_space = rotation_converter.FKinSpace(M_fk, Slist, theta_fk);
    T_body = rotation_converter.FKinBody(M_fk, Blist, theta_fk);
    [n_pass, n_fail] = check(max(abs(T_space(:) - M_fk(:))) < 1e-9, ...
        'FKinSpace zero angles', n_pass, n_fail);
    [n_pass, n_fail] = check(max(abs(T_body(:) - M_fk(:))) < 1e-9, ...
        'FKinBody zero angles', n_pass, n_fail);

    % FK consistency: space and body should give same result
    theta_test = [0.5, -0.3, 0.8];
    T_s = rotation_converter.FKinSpace(M_fk, Slist, theta_test);
    T_b = rotation_converter.FKinBody(M_fk, Blist, theta_test);
    [n_pass, n_fail] = check(max(abs(T_s(:) - T_b(:))) < 1e-9, ...
        'FKinSpace == FKinBody', n_pass, n_fail);

    % Jacobians
    Js = rotation_converter.JacobianSpace(Slist, theta_test);
    Jb = rotation_converter.JacobianBody(Blist, theta_test);
    [n_pass, n_fail] = check(all(size(Js) == [6 3]), 'JacobianSpace 6x3', n_pass, n_fail);
    [n_pass, n_fail] = check(all(size(Jb) == [6 3]), 'JacobianBody 6x3', n_pass, n_fail);

    % IK: use FK result as target
    T_desired_ik = T_b;
    [theta_ik, success_ik] = rotation_converter.IKinBody( ...
        Blist, M_fk, T_desired_ik, [0 0 0]);
    [n_pass, n_fail] = check(success_ik, 'IKinBody converges', n_pass, n_fail);

    if success_ik
        T_ik = rotation_converter.FKinBody(M_fk, Blist, theta_ik);
        [n_pass, n_fail] = check(max(abs(T_ik(:) - T_desired_ik(:))) < 1e-3, ...
            'IKinBody accuracy', n_pass, n_fail);
    end

    % IK non-convergence
    T_far = eye(4); T_far(1:3, 4) = [100; 100; 100];
    Blist_1 = [0 0 1 0 0 0]';
    M_1 = eye(4); M_1(1, 4) = 1;
    [~, suc_fail] = rotation_converter.IKinBody(Blist_1, M_1, T_far, [0], 1e-4, 1e-4, 5);
    [n_pass, n_fail] = check(~suc_fail, 'IKinBody non-convergence', n_pass, n_fail);

    % ========================================
    % 9. Screw Trajectory
    % ========================================
    fprintf('--- Screw Trajectory ---\n');

    X_start = eye(4);
    X_end = eye(4); X_end(1:3, 4) = [1; 0; 0];
    traj = rotation_converter.ScrewTrajectory(X_start, X_end, 1.0, 10, 3);
    [n_pass, n_fail] = check(numel(traj) == 10, 'ScrewTrajectory N=10', n_pass, n_fail);
    [n_pass, n_fail] = check(max(abs(traj{1}(:) - X_start(:))) < 1e-9, ...
        'ScrewTrajectory start', n_pass, n_fail);
    [n_pass, n_fail] = check(max(abs(traj{end}(:) - X_end(:))) < 1e-9, ...
        'ScrewTrajectory end', n_pass, n_fail);

    traj5 = rotation_converter.ScrewTrajectory(X_start, X_end, 1.0, 10, 5);
    [n_pass, n_fail] = check(max(abs(traj5{end}(:) - X_end(:))) < 1e-9, ...
        'ScrewTrajectory quintic end', n_pass, n_fail);

    % ========================================
    % 10. Rotation class
    % ========================================
    fprintf('--- Rotation Class ---\n');

    rot_id = rotation_converter.Rotation.identity();
    q_id_out = rot_id.as_quaternion();
    [n_pass, n_fail] = check(abs(q_id_out(1) - 1) < 1e-12, ...
        'Rotation.identity', n_pass, n_fail);

    rot = rotation_converter.Rotation.from_euler(0.3, 0.7, -0.5, 'xyz');
    R_out = rot.as_rotation_matrix();
    [n_pass, n_fail] = check(abs(det(R_out) - 1) < 1e-9, ...
        'Rotation.from_euler -> SO(3)', n_pass, n_fail);

    [ax_out, ang_out] = rot.as_axis_angle();
    [n_pass, n_fail] = check(abs(norm(ax_out) - 1) < 1e-9, ...
        'Rotation.as_axis_angle unit axis', n_pass, n_fail);

    rod_out = rot.as_rodrigues();
    [n_pass, n_fail] = check(numel(rod_out) == 3, ...
        'Rotation.as_rodrigues 3 elements', n_pass, n_fail);

    % Compose and inverse
    rot2 = rotation_converter.Rotation.from_axis_angle([0 1 0], 0.5);
    rot3 = rot.compose(rot2);
    rot_inv = rot3.inverse();
    rot_id_check = rot3.compose(rot_inv);
    q_id_check = rot_id_check.as_quaternion();
    [n_pass, n_fail] = check(abs(abs(q_id_check(1)) - 1) < 1e-9, ...
        'Rotation compose * inverse = identity', n_pass, n_fail);

    % All factory methods
    rot_q = rotation_converter.Rotation.from_quaternion([1 0 0 0]);
    rot_rm = rotation_converter.Rotation.from_rotation_matrix(eye(3));
    rot_ro = rotation_converter.Rotation.from_rodrigues([0 0 0]);
    [n_pass, n_fail] = check(rot_q == rot_rm, 'Rotation equality', n_pass, n_fail);

    % ========================================
    % 11. RigidTransform class
    % ========================================
    fprintf('--- RigidTransform Class ---\n');

    rt_id = rotation_converter.RigidTransform.identity('world');
    [n_pass, n_fail] = check(rt_id.is_identity(), ...
        'RigidTransform.identity', n_pass, n_fail);

    % from_rotation_translation
    R_rt = rotation_converter.axis_angle_to_rotation_matrix([0 0 1], pi/4);
    rt = rotation_converter.RigidTransform.from_rotation_translation( ...
        R_rt, [1 2 3], 'base', 'tool');
    [n_pass, n_fail] = check(strcmp(rt.source_frame(), 'base'), 'source frame', n_pass, n_fail);
    [n_pass, n_fail] = check(strcmp(rt.target_frame(), 'tool'), 'target frame', n_pass, n_fail);

    % Composition with frame checking
    rt2 = rotation_converter.RigidTransform.from_rotation_translation( ...
        eye(3), [0 0 1], 'tool', 'sensor');
    rt3 = rt.compose(rt2);
    [n_pass, n_fail] = check(strcmp(rt3.source_frame(), 'base') && ...
        strcmp(rt3.target_frame(), 'sensor'), 'compose frames', n_pass, n_fail);

    % Frame mismatch
    rt_bad = rotation_converter.RigidTransform.from_rotation_translation( ...
        eye(3), [0 0 0], 'camera', 'world');
    frame_error_caught = false;
    try
        rt.compose(rt_bad);  % base->tool * camera->world should fail
    catch e
        if ~isempty(strfind(e.identifier, 'FrameError'))
            frame_error_caught = true;
        end
    end
    [n_pass, n_fail] = check(frame_error_caught, 'FrameError on mismatch', n_pass, n_fail);

    % Inverse
    rt_inv = rt.inverse();
    [n_pass, n_fail] = check(strcmp(rt_inv.source_frame(), 'tool'), ...
        'inverse swaps frames', n_pass, n_fail);

    % Compose with inverse = identity
    rt_prod = rt.compose(rt_inv);
    [n_pass, n_fail] = check(rt_prod.is_identity(), ...
        'T * T_inv = identity', n_pass, n_fail);

    % apply_point / apply_vector
    p_out = rt.apply_point([1 0 0]);
    [n_pass, n_fail] = check(numel(p_out) == 3, 'apply_point 3 elements', n_pass, n_fail);

    v_out = rt.apply_vector([1 0 0]);
    [n_pass, n_fail] = check(numel(v_out) == 3, 'apply_vector 3 elements', n_pass, n_fail);

    % apply_points batch
    P_batch = [1 0 0; 0 1 0; 0 0 1];
    P_out = rt.apply_points(P_batch);
    [n_pass, n_fail] = check(all(size(P_out) == [3, 3]), 'apply_points batch', n_pass, n_fail);

    % Predicates
    rt_trans = rotation_converter.RigidTransform.pure_translation([1 2 3], 'a', 'b');
    [n_pass, n_fail] = check(rt_trans.is_pure_translation(), 'is_pure_translation', n_pass, n_fail);
    [n_pass, n_fail] = check(~rt_trans.is_pure_rotation(), 'not pure_rotation', n_pass, n_fail);

    rot_only = rotation_converter.Rotation.from_axis_angle([0 0 1], 0.5);
    rt_rot = rotation_converter.RigidTransform.pure_rotation(rot_only, 'a', 'b');
    [n_pass, n_fail] = check(rt_rot.is_pure_rotation(), 'is_pure_rotation', n_pass, n_fail);

    % All factory methods
    rt_quat = rotation_converter.RigidTransform.from_quaternion_translation( ...
        [1 0 0 0], [1 2 3], 'a', 'b');
    rt_euler = rotation_converter.RigidTransform.from_euler_translation( ...
        0.1, 0.2, 0.3, [1 2 3], 'xyz', 'a', 'b');
    rt_aa = rotation_converter.RigidTransform.from_axis_angle_translation( ...
        [0 0 1], 0.5, [1 2 3], 'a', 'b');
    rt_rod = rotation_converter.RigidTransform.from_rodrigues_translation( ...
        [0 0 0.5], [1 2 3], 'a', 'b');
    rt_twist_f = rotation_converter.RigidTransform.from_twist( ...
        [0 0 1 0.5 -0.3 0.1], 0.5, 'a', 'b');
    [n_pass, n_fail] = check(~isempty(rt_quat.as_matrix()), 'from_quaternion_translation', n_pass, n_fail);
    [n_pass, n_fail] = check(~isempty(rt_euler.as_matrix()), 'from_euler_translation', n_pass, n_fail);
    [n_pass, n_fail] = check(~isempty(rt_aa.as_matrix()), 'from_axis_angle_translation', n_pass, n_fail);
    [n_pass, n_fail] = check(~isempty(rt_rod.as_matrix()), 'from_rodrigues_translation', n_pass, n_fail);
    [n_pass, n_fail] = check(~isempty(rt_twist_f.as_matrix()), 'from_twist', n_pass, n_fail);

    % Body/space twists
    Vb = rt.body_twist();
    Vs = rt.space_twist();
    [n_pass, n_fail] = check(numel(Vb) == 6, 'body_twist 6 elements', n_pass, n_fail);
    [n_pass, n_fail] = check(numel(Vs) == 6, 'space_twist 6 elements', n_pass, n_fail);

    Vs2 = rt.body_to_space_twist(Vb);
    [n_pass, n_fail] = check(max(abs(Vs - Vs2)) < 1e-9, ...
        'body_to_space_twist consistency', n_pass, n_fail);

    Vb2 = rt.space_to_body_twist(Vs);
    [n_pass, n_fail] = check(max(abs(Vb - Vb2)) < 1e-9, ...
        'space_to_body_twist consistency', n_pass, n_fail);

    % Wrench conversions
    Fb = [1 0 0 0 0 0];
    Fs = rt.body_to_space_wrench(Fb);
    Fb2 = rt.space_to_body_wrench(Fs);
    [n_pass, n_fail] = check(max(abs(Fb - Fb2)) < 1e-9, ...
        'wrench roundtrip', n_pass, n_fail);

    % as_twist / as_screw
    [xi_out, theta_out] = rt.as_twist();
    [n_pass, n_fail] = check(numel(xi_out) == 6 && theta_out >= 0, ...
        'as_twist output', n_pass, n_fail);

    screw_out = rt.as_screw();
    [n_pass, n_fail] = check(isfield(screw_out, 'axis') && isfield(screw_out, 'pitch'), ...
        'as_screw fields', n_pass, n_fail);

    % ========================================
    % 12. Motion Examples
    % ========================================
    fprintf('--- Motion Examples ---\n');

    traj_fb = rotation_converter.football_spiral(10);
    [n_pass, n_fail] = check(numel(traj_fb) == 10, 'football_spiral N=10', n_pass, n_fail);
    [n_pass, n_fail] = check(all(size(traj_fb{1}) == [4 4]), ...
        'football_spiral SE(3)', n_pass, n_fail);

    traj_fr = rotation_converter.frisbee_flight(10);
    [n_pass, n_fail] = check(numel(traj_fr) == 10, 'frisbee_flight N=10', n_pass, n_fail);

    % ========================================
    % 13. Visualization data (no actual plotting)
    % ========================================
    fprintf('--- Visualization Data ---\n');

    frames = rotation_converter.build_animation_frames(traj_fb);
    [n_pass, n_fail] = check(numel(frames) == 10, 'build_animation_frames N', n_pass, n_fail);
    [n_pass, n_fail] = check(numel(frames(1).origin) == 3, ...
        'frame origin 3 elements', n_pass, n_fail);

    screw_data = rotation_converter.extract_screw_axes_from_trajectory(traj_fb);
    [n_pass, n_fail] = check(numel(screw_data) == 9, ...
        'extract_screw_axes N-1 pairs', n_pass, n_fail);

    % ========================================
    % 14. Contract enforcement (NaN/Inf rejection)
    % ========================================
    fprintf('--- Contract Enforcement ---\n');

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.normalize_quaternion([0 0 0 0]), ...
        'reject zero quaternion', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.quaternion_to_rotation_matrix([2 0 0 0]), ...
        'reject non-unit quaternion', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.quaternion_to_rotation_matrix([NaN 0 0 0]), ...
        'reject NaN quaternion', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.MatrixExp3([0 Inf 0; 0 0 0; 0 0 0]), ...
        'reject Inf in so3', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.MatrixLog3(eye(3) * Inf), ...
        'reject Inf in MatrixLog3', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.axis_angle_to_quaternion([2 0 0], 1), ...
        'reject non-unit axis', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.euler_to_quaternion(0, 0, 0, 'abc'), ...
        'reject invalid convention', n_pass, n_fail);

    [n_pass, n_fail] = check_error(@() ...
        rotation_converter.screw_to_twist(struct('axis', [0 0 0], ...
            'point', [0 0 0], 'pitch', inf)), ...
        'reject zero axis inf pitch', n_pass, n_fail);

    % ========================================
    % Summary
    % ========================================
    fprintf('\n========================================\n');
    fprintf('RESULTS: %d passed, %d failed, %d total\n', ...
            n_pass, n_fail, n_pass + n_fail);
    fprintf('========================================\n');

    if n_fail > 0
        error('rotation_converter:TestFailure', '%d tests failed', n_fail);
    else
        fprintf('All tests passed!\n');
    end

    rng(rng_state);  % restore RNG state
end

% --- Helper functions ---

function [n_pass, n_fail] = check(condition, name, n_pass, n_fail)
    if condition
        n_pass = n_pass + 1;
    else
        fprintf('  FAIL: %s\n', name);
        n_fail = n_fail + 1;
    end
end

function [n_pass, n_fail] = check_error(fn, name, n_pass, n_fail)
    caught = false;
    try
        fn();
    catch
        caught = true;
    end
    [n_pass, n_fail] = check(caught, name, n_pass, n_fail);
end
