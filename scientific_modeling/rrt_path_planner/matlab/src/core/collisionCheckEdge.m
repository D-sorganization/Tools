function isValid = collisionCheckEdge(path, obstacles, stepSize)
%COLLISIONCHECKEDGE Validate a polyline path against obstacles.
%   isValid = collisionCheckEdge(path, obstacles) samples line segments
%   between waypoints using a default resolution and returns true only if
%   no point collides with any obstacle.
%
%   isValid = collisionCheckEdge(path, obstacles, stepSize) lets callers
%   override the sampling resolution along each segment.

    arguments
        path (:, 3) double
        obstacles (:, 8) double
        stepSize (1, 1) double {mustBePositive} = 0.02
    end

    if isempty(path)
        isValid = false;
        return;
    end

    if size(path, 1) == 1
        isValid = ~collisionCheck(path(1, :), obstacles);
        return;
    end

    isValid = true;
    numPoints = size(path, 1);

    for idx = 1:(numPoints - 1)
        startPoint = path(idx, :);
        endPoint = path(idx + 1, :);

        segment = endPoint - startPoint;
        distance = norm(segment);

        % Always include both endpoints for robustness
        sampleCount = max(2, ceil(distance / stepSize) + 1);
        parameter = linspace(0, 1, sampleCount)';
        samples = startPoint + parameter .* segment;

        for sampleIdx = 1:sampleCount
            samplePoint = samples(sampleIdx, :);
            if collisionCheck(samplePoint, obstacles)
                isValid = false;
                return;
            end
        end
    end
end
