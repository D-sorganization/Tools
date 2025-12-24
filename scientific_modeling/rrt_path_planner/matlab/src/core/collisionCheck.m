function flag = collisionCheck(point, obstacles)
% collisionCheck.m
% Checks if a point collides with any obstacle (sphere or cube)

flag = false; % Default: no collision

if isempty(point)
    flag = true;
    return;
end

point = point(:)'; % Force row vector

for i = 1:size(obstacles,1)
    type = obstacles(i,1);
    center = obstacles(i,2:4);
    center = center(:)'; % Force row vector

    % --- SAFETY CHECK ---
    if numel(point) ~= 3 || numel(center) ~= 3
        continue; % Skip bad points/centers
    end
    % ---------------------

    size_param = obstacles(i,5);

    if type == 0
        % Sphere collision check
        dist = norm(point - center);
        if dist <= size_param
            flag = true;
            return;
        end
    else
        % Cube collision check
        if all(abs(point - center) <= size_param/2)
            flag = true;
            return;
        end
    end
end

end
