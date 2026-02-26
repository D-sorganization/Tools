function T = dh_to_matrix(theta, d, a, alpha, modified)
% DH_TO_MATRIX Convert Denavit-Hartenberg parameters to a 4x4 matrix.
%
%   T = dh_to_matrix(theta, d, a, alpha, modified)
%
%   Args:
%       theta: Joint rotation about z-axis (rad)
%       d: Link offset along z-axis
%       a: Link length along x-axis
%       alpha: Link twist about x-axis (rad)
%       modified: (Optional) If true, uses modified Craig DH parameters.
%                 If false (default), uses standard Spong DH parameters.
%
%   Returns:
%       T: 4x4 homogeneous transformation matrix SE(3).

    if nargin < 5
        modified = false;
    end

    ct = cos(theta);
    st = sin(theta);
    ca = cos(alpha);
    sa = sin(alpha);

    % Clean up near-zero floats for perfect orthogonality
    if abs(ct) < 1e-12; ct = 0.0; end
    if abs(st) < 1e-12; st = 0.0; end
    if abs(ca) < 1e-12; ca = 0.0; end
    if abs(sa) < 1e-12; sa = 0.0; end

    if modified
        % Modified DH (Craig): i-1 to i
        T = [ ct,       -st,      0,   a;
              st * ca,  ct * ca, -sa, -d * sa;
              st * sa,  ct * sa,  ca,  d * ca;
              0.0,      0.0,      0.0, 1.0 ];
    else
        % Standard DH (Spong): i-1 to i
        T = [ ct,  -st * ca,  st * sa, a * ct;
              st,   ct * ca, -ct * sa, a * st;
              0.0,  sa,       ca,      d;
              0.0,  0.0,      0.0,     1.0 ];
    end
end
