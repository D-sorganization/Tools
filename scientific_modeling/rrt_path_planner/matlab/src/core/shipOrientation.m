function [R, up_vector] = shipOrientation(current_pos, next_pos, current_forward, world_up)
% SHIPORIENTATION Calculates proper ship orientation for nose-forward movement
% Inputs:
%   current_pos - current position [x,y,z]
%   next_pos - next position [x,y,z]
%   current_forward - current forward direction [x,y,z]
%   world_up - world up vector [x,y,z]
% Outputs:
%   R - rotation matrix (3x3)
%   up_vector - calculated up vector

% Calculate desired forward direction
if norm(next_pos - current_pos) < 1e-6
    % If no movement, keep current orientation
    forward = current_forward;
else
    forward = (next_pos - current_pos) / norm(next_pos - current_pos);
end

% Ensure forward is normalized
forward = forward / norm(forward);

% Calculate right vector (cross product of world_up and forward)
right = cross(world_up, forward);
if norm(right) < 1e-6
    % If forward is parallel to world_up, use a different up vector
    right = cross([1, 0, 0], forward);
    if norm(right) < 1e-6
        right = cross([0, 1, 0], forward);
    end
end
right = right / norm(right);

% Calculate up vector (cross product of forward and right)
up_vector = cross(forward, right);
up_vector = up_vector / norm(up_vector);

% Build rotation matrix [forward, right, up]
R = [forward', right', up_vector'];

% Ensure orthogonality
[U, ~, V] = svd(R);
R = U * V';

end 