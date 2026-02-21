function S = skew_symmetric(v)
%SKEW_SYMMETRIC Build the 3x3 skew-symmetric (cross-product) matrix of a 3-vector.
%   S = skew_symmetric(V) returns [0 -v3 v2; v3 0 -v1; -v2 v1 0].

    S = [  0,    -v(3),  v(2);
          v(3),   0,    -v(1);
         -v(2),  v(1),   0  ];
end
