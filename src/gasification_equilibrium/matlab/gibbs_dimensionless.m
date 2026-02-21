function g = gibbs_dimensionless(coeffs, T)
%GIBBS_DIMENSIONLESS Compute G/(RT) from NASA polynomial coefficients.
%
%   g = gibbs_dimensionless(coeffs, T)
%
%   Inputs:
%     coeffs - 7-element NASA polynomial coefficient vector
%     T      - Temperature [K], scalar > 0
%
%   Output:
%     g - Dimensionless Gibbs free energy G/(RT)
%
%   Design by Contract:
%     Precondition:  length(coeffs) == 7, T > 0
%     Postcondition: g is a finite scalar
%
%   Compatible with MATLAB and GNU Octave.

    assert(length(coeffs) == 7, 'coeffs must have 7 elements');
    assert(T > 0, 'Temperature must be positive');

    a = coeffs;

    % H/(RT)
    h_rt = a(1) + a(2)*T/2 + a(3)*T^2/3 + a(4)*T^3/4 + a(5)*T^4/5 + a(6)/T;

    % S/R
    s_r = a(1)*log(T) + a(2)*T + a(3)*T^2/2 + a(4)*T^3/3 + a(5)*T^4/4 + a(7);

    % G/(RT) = H/(RT) - S/R
    g = h_rt - s_r;
end
