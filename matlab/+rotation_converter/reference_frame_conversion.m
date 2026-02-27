function out = reference_frame_conversion(operation, varargin)
%REFERENCE_FRAME_CONVERSION Educational reference-frame conversion workflow.
%
% out = rotation_converter.reference_frame_conversion(operation, Name, Value, ...)
%
% Supported operations:
%   - "twist_frame_conversion"
%       Required: 'transform' (4x4), 'twist' (1x6 or 6x1)
%   - "homogeneous_transform"
%       Required: 'rotation_matrix' (3x3), 'translation' (1x3 or 3x1)
%   - "so3_so3_maps"
%       One of: 'so3_vector' (1x3/3x1), 'so3_matrix' (3x3), 'rotation_matrix' (3x3)
%
% Returns struct fields:
%   out.operation
%   out.results
%   out.explanation_markdown
%   out.explanation_latex

    params = parse_name_value_pairs(varargin{:});

    switch char(operation)
        case 'twist_frame_conversion'
            out = op_twist_frame_conversion(params);
        case 'homogeneous_transform'
            out = op_homogeneous_transform(params);
        case 'so3_so3_maps'
            out = op_so3_so3_maps(params);
        otherwise
            error('rotation_converter:PreconditionError', ...
                'Unsupported operation: %s', char(operation));
    end
end

function params = parse_name_value_pairs(varargin)
    params = struct();
    if mod(numel(varargin), 2) ~= 0
        error('rotation_converter:PreconditionError', ...
            'Name/value arguments must be provided in pairs');
    end
    for i = 1:2:numel(varargin)
        name = varargin{i};
        value = varargin{i + 1};
        if ~(ischar(name) || isstring(name))
            error('rotation_converter:PreconditionError', ...
                'Parameter names must be strings');
        end
        params.(char(name)) = value;
    end
end

function out = op_twist_frame_conversion(params)
    require_field(params, 'transform');
    require_field(params, 'twist');
    T = params.transform;
    xi = params.twist(:);
    rotation_converter.internal.require(all(size(T) == [4, 4]), ...
        'transform must be 4x4');
    rotation_converter.internal.require(numel(xi) == 6, ...
        'twist must have 6 elements');
    Ad = rotation_converter.adjoint_representation(T);
    xi_out = Ad * xi;

    out = struct();
    out.operation = 'twist_frame_conversion';
    out.results = struct( ...
        'adjoint_matrix', Ad, ...
        'input_twist', xi(:)', ...
        'output_twist', xi_out(:)');
    out.explanation_markdown = [ ...
        'Twists transform with the adjoint matrix of a homogeneous transform: ' ...
        'V_b = Ad_T * V_a, where Ad_T = [[R,0],[skew(p)R,R]].'];
    out.explanation_latex = [ ...
        'V_b = \mathrm{Ad}_T V_a,\quad ' ...
        '\mathrm{Ad}_T=\begin{bmatrix}R&0\\[p]_\times R&R\end{bmatrix}.'];
end

function out = op_homogeneous_transform(params)
    require_field(params, 'rotation_matrix');
    require_field(params, 'translation');
    R = params.rotation_matrix;
    p = params.translation(:);
    rotation_converter.internal.validate_rotation_matrix(R);
    rotation_converter.internal.require(numel(p) == 3, ...
        'translation must have 3 elements');
    T = eye(4);
    T(1:3, 1:3) = R;
    T(1:3, 4) = p;
    T_inv = eye(4);
    T_inv(1:3, 1:3) = R';
    T_inv(1:3, 4) = -R' * p;

    out = struct();
    out.operation = 'homogeneous_transform';
    out.results = struct( ...
        'rotation_matrix', R, ...
        'translation', p(:)', ...
        'homogeneous_transform', T, ...
        'inverse_transform', T_inv);
    out.explanation_markdown = [ ...
        'Build T = [[R,p],[0,1]] where R rotates and p translates frame origin. ' ...
        'Inverse is T^{-1} = [[R^T,-R^Tp],[0,1]].'];
    out.explanation_latex = [ ...
        'T=\begin{bmatrix}R&p\\0&1\end{bmatrix},\quad ' ...
        'T^{-1}=\begin{bmatrix}R^\top&-R^\top p\\0&1\end{bmatrix}.'];
end

function out = op_so3_so3_maps(params)
    has_vec = isfield(params, 'so3_vector');
    has_mat = isfield(params, 'so3_matrix');
    has_rot = isfield(params, 'rotation_matrix');

    if has_mat
        so3_hat = params.so3_matrix;
        rotation_converter.internal.require(all(size(so3_hat) == [3, 3]), ...
            'so3_matrix must be 3x3');
        omega = rotation_converter.so3ToVec(so3_hat);
    elseif has_vec
        omega = params.so3_vector(:)';
        rotation_converter.internal.require(numel(omega) == 3, ...
            'so3_vector must have 3 elements');
        so3_hat = rotation_converter.VecToso3(omega);
    elseif has_rot
        R = params.rotation_matrix;
        rotation_converter.internal.validate_rotation_matrix(R);
        so3_hat = rotation_converter.MatrixLog3(R);
        omega = rotation_converter.so3ToVec(so3_hat);
    else
        error('rotation_converter:PreconditionError', ...
            'so3_so3_maps requires so3_vector, so3_matrix, or rotation_matrix');
    end

    R_exp = rotation_converter.MatrixExp3(so3_hat);
    so3_log = rotation_converter.MatrixLog3(R_exp);
    omega_log = rotation_converter.so3ToVec(so3_log);

    out = struct();
    out.operation = 'so3_so3_maps';
    out.results = struct( ...
        'so3_vector', omega(:)', ...
        'so3_hat_matrix', so3_hat, ...
        'so3_vee_vector', rotation_converter.so3ToVec(so3_hat), ...
        'so3_exponential_SO3', R_exp, ...
        'so3_log_vector', omega_log(:)');
    out.explanation_markdown = [ ...
        'Hat maps omega in R^3 to so(3): omega^; exp maps so(3)->SO(3): R=exp(omega^); ' ...
        'log maps back with omega=vee(log(R)).'];
    out.explanation_latex = [ ...
        '\widehat{\omega}=\begin{bmatrix}0&-\omega_3&\omega_2\\' ...
        '\omega_3&0&-\omega_1\\-\omega_2&\omega_1&0\end{bmatrix},\quad ' ...
        'R=\exp(\widehat{\omega}),\quad \omega=\mathrm{vee}(\log R).'];
end

function require_field(params, name)
    if ~isfield(params, name)
        error('rotation_converter:PreconditionError', ...
            'Missing required parameter: %s', name);
    end
end
