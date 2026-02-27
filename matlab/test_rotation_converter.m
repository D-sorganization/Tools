function test_rotation_converter()
%TEST_ROTATION_CONVERTER Run smoke/regression tests for rotation_converter.
%
% Compatible with MATLAB and GNU Octave.

    root_dir = fileparts(mfilename("fullpath"));
    addpath(root_dir);

    pass_count = 0;
    fail_count = 0;

    [pass_count, fail_count] = test_quaternion_primitives(pass_count, fail_count);
    [pass_count, fail_count] = test_matrix_roundtrip(pass_count, fail_count);
    [pass_count, fail_count] = test_euler_roundtrip(pass_count, fail_count);
    [pass_count, fail_count] = test_twist_roundtrip(pass_count, fail_count);
    [pass_count, fail_count] = test_rigid_transform(pass_count, fail_count);

    fprintf("\nRotation Converter tests: %d passed, %d failed.\n", pass_count, fail_count);
    if fail_count > 0
        error("rotation_converter:TestFailure", "%d tests failed", fail_count);
    end
end

function [pass_count, fail_count] = test_quaternion_primitives(pass_count, fail_count)
    fprintf("Running quaternion primitive tests...\n");

    q = rotation_converter.normalize_quaternion([0 0 0 2]);
    [pass_count, fail_count] = check_true(abs(norm(q) - 1.0) < 1e-12, ...
        "normalize_quaternion unit norm", pass_count, fail_count);

    q_id = [1 0 0 0];
    q_test = rotation_converter.normalize_quaternion([1 2 3 4]);
    product = rotation_converter.quaternion_multiply(q_id, q_test);
    [pass_count, fail_count] = check_true(max(abs(product - q_test)) < 1e-12, ...
        "quaternion identity multiplication", pass_count, fail_count);

    q_inv = rotation_converter.quaternion_conjugate(q_test);
    product_inv = rotation_converter.quaternion_multiply(q_test, q_inv);
    [pass_count, fail_count] = check_true(abs(product_inv(1) - 1.0) < 1e-9 && ...
        norm(product_inv(2:4)) < 1e-9, ...
        "quaternion inverse multiplication", pass_count, fail_count);
end

function [pass_count, fail_count] = test_matrix_roundtrip(pass_count, fail_count)
    fprintf("Running matrix roundtrip tests...\n");

    q_z90 = rotation_converter.normalize_quaternion([cos(pi/4), 0, 0, sin(pi/4)]);
    r = rotation_converter.quaternion_to_rotation_matrix(q_z90);
    expected = [0 -1 0; 1 0 0; 0 0 1];
    [pass_count, fail_count] = check_true(max(abs(r(:) - expected(:))) < 1e-9, ...
        "known quaternion to rotation matrix", pass_count, fail_count);

    q_back = rotation_converter.rotation_matrix_to_quaternion(r);
    [pass_count, fail_count] = check_quaternion_equivalent(q_z90, q_back, ...
        "quaternion-matrix roundtrip", pass_count, fail_count);

    init_rng();
    for idx = 1:10
        q_rand = rotation_converter.normalize_quaternion(randn(1, 4));
        r_rand = rotation_converter.quaternion_to_rotation_matrix(q_rand);
        q_round = rotation_converter.rotation_matrix_to_quaternion(r_rand);
        label = sprintf("random roundtrip %d", idx);
        [pass_count, fail_count] = check_quaternion_equivalent(q_rand, q_round, ...
            label, pass_count, fail_count);
    end
end

function [pass_count, fail_count] = test_euler_roundtrip(pass_count, fail_count)
    fprintf("Running Euler conversion tests...\n");

    conventions = {"xyz", "zyx", "zxz"};
    for idx = 1:numel(conventions)
        convention = conventions{idx};
        q = rotation_converter.euler_to_quaternion(0.3, 0.7, -0.5, convention);
        [a2, b2, c2] = rotation_converter.quaternion_to_euler(q, convention);
        q2 = rotation_converter.euler_to_quaternion(a2, b2, c2, convention);
        label = sprintf("Euler convention %s roundtrip", convention);
        [pass_count, fail_count] = check_quaternion_equivalent(q, q2, label, ...
            pass_count, fail_count);
    end
end

function [pass_count, fail_count] = test_twist_roundtrip(pass_count, fail_count)
    fprintf("Running twist/screw tests...\n");

    xi = [0 0 1 1 0 0];
    se3 = rotation_converter.twist_vector_to_se3_matrix(xi);
    xi_back = rotation_converter.se3_matrix_to_twist_vector(se3);
    [pass_count, fail_count] = check_true(max(abs(xi - xi_back)) < 1e-12, ...
        "twist <-> se3 roundtrip", pass_count, fail_count);

    t = rotation_converter.twist_angle_to_homogeneous(xi, 0.8);
    [xi2, theta2] = rotation_converter.homogeneous_to_twist_angle(t);
    t2 = rotation_converter.twist_angle_to_homogeneous(xi2, theta2);
    [pass_count, fail_count] = check_true(max(abs(t(:) - t2(:))) < 1e-8, ...
        "twist-angle <-> SE3 roundtrip", pass_count, fail_count);
end

function [pass_count, fail_count] = test_rigid_transform(pass_count, fail_count)
    fprintf("Running rigid transform tests...\n");

    rot = rotation_converter.Rotation.from_axis_angle([0 0 1], pi / 6);
    tf = rotation_converter.RigidTransform.from_rotation_translation( ...
        rot.as_rotation_matrix(), [1 2 3], "world", "tool");
    q = tf.as_quaternion_translation();

    [pass_count, fail_count] = check_true(numel(q.quaternion) == 4, ...
        "RigidTransform quaternion output size", pass_count, fail_count);
    [pass_count, fail_count] = check_true(numel(q.translation) == 3, ...
        "RigidTransform translation output size", pass_count, fail_count);
end

function [pass_count, fail_count] = check_quaternion_equivalent(q1, q2, label, pass_count, fail_count)
    value = abs(abs(dot(q1, q2)) - 1.0);
    [pass_count, fail_count] = check_true(value < 1e-9, label, pass_count, fail_count);
end

function [pass_count, fail_count] = check_true(condition, label, pass_count, fail_count)
    if condition
        pass_count = pass_count + 1;
    else
        fail_count = fail_count + 1;
        fprintf("  FAIL: %s\n", label);
    end
end

function init_rng()
    try
        rng(42, "twister");
    catch
        rand("seed", 42); %#ok<RAND>
        randn("seed", 42); %#ok<RAND>
    end
end
