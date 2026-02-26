function [thetalist, success] = IKinSpace(Slist, M, T, thetalist0, eomg, ev)
% IKINSPACE Computes inverse kinematics in the space frame for an open chain robot
%
%   [thetalist, success] = IKinSpace(Slist, M, T, thetalist0, eomg, ev)
%
%   Args:
%       Slist: 6×n space screw axes.
%       M: 4x4 home configuration.
%       T: 4x4 desired end-effector SE(3) pose.
%       thetalist0: n-vector initial guess for joint angles.
%       eomg: Angular error tolerance (rad).
%       ev: Linear error tolerance.
%
%   Returns:
%       thetalist: Joint angles that achieve T within tolerances.
%       success: Logical true if algorithm found a solution.

    thetalist = thetalist0(:);
i = 0;
maxiterations = 20;
Tsb = rotation_converter.FKinSpace(M, Slist, thetalist);
Vs = rotation_converter.adjoint_representation(Tsb) *
     ...rotation_converter.se3ToVec(
         rotation_converter.MatrixLog6(rotation_converter.TransInv(Tsb) * T));
err = (norm(Vs(1 : 3)) > eomg) || (norm(Vs(4 : 6)) > ev);

while
  err &&i < maxiterations J =
      rotation_converter.JacobianSpace(Slist, thetalist);
thetalist = thetalist + pinv(J) * Vs;
i = i + 1;
Tsb = rotation_converter.FKinSpace(M, Slist, thetalist);
Vs = rotation_converter.adjoint_representation(Tsb) *
     ...rotation_converter.se3ToVec(
         rotation_converter.MatrixLog6(rotation_converter.TransInv(Tsb) * T));
err = (norm(Vs(1 : 3)) > eomg) || (norm(Vs(4 : 6)) > ev);
end success = ~err;
end
