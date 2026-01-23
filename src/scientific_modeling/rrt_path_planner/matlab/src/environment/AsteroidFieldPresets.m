function obstacles = AsteroidFieldPresets(presetName, bounds)
%ASTEROIDFIELDPRESETS Generate standardized obstacle fields.
%   obstacles = AsteroidFieldPresets('light_debris', bounds)
%   returns a matrix with obstacle definitions matching the existing
%   [type x y z size r g b] format used by the RRT planners.

presetName = lower(string(presetName));
switch presetName
    case {"empty", "none"}
        obstacles = zeros(0, 8);

    case {"light_debris", "light"}
        obstacles = generatePreset(bounds, 20, 0.03, 0.08, 'sparse');

    case {"dense_belt", "dense"}
        obstacles = generatePreset(bounds, 75, 0.04, 0.10, 'belt');

    case {"death_star_trench", "trench"}
        obstacles = generateTrench(bounds);

    case {"hoth_field", "hoth"}
        obstacles = generatePreset(bounds, 60, 0.05, 0.09, 'clustered');

    case {"ring_formation", "ring"}
        obstacles = generateRing(bounds, 50, 0.04, 0.08);

    otherwise
        obstacles = generatePreset(bounds, 50, 0.03, 0.08, 'medium');
end
end

function obstacles = generatePreset(bounds, count, minSize, maxSize, mode)
obstacles = zeros(count, 8);
center = [(bounds(1) + bounds(2)) / 2, (bounds(3) + bounds(4)) / 2, (bounds(5) + bounds(6)) / 2];

distribution = mode;
for idx = 1:count
    switch distribution
        case 'belt'
            angle = rand() * 2 * pi;
            radius = 0.4 * (bounds(2) - bounds(1)) * rand() + 0.2 * (bounds(2) - bounds(1));
            x = center(1) + radius * cos(angle);
            y = center(2) + 0.1 * (bounds(4) - bounds(3)) * randn();
            z = center(3) + 0.05 * (bounds(6) - bounds(5)) * randn();
        case 'clustered'
            x = (bounds(2) - bounds(1)) * rand() + bounds(1);
            y = (bounds(4) - bounds(3)) * rand() + bounds(3);
            z = center(3) + 0.1 * (bounds(6) - bounds(5)) * randn();
        case 'sparse'
            x = (bounds(2) - bounds(1)) * rand() + bounds(1);
            y = (bounds(4) - bounds(3)) * rand() + bounds(3);
            z = (bounds(6) - bounds(5)) * rand() + bounds(5);
        otherwise
            x = (bounds(2) - bounds(1)) * rand() + bounds(1);
            y = (bounds(4) - bounds(3)) * rand() + bounds(3);
            z = (bounds(6) - bounds(5)) * rand() + bounds(5);
    end

    sizeVal = minSize + (maxSize - minSize) * rand();
    isCube = round(rand());
    color = [0.5 + 0.5 * rand(), 0.3 + 0.7 * rand(), 0.3 + 0.7 * rand()];
    x = clampToBounds(x, bounds(1), bounds(2), sizeVal);
    y = clampToBounds(y, bounds(3), bounds(4), sizeVal);
    z = clampToBounds(z, bounds(5), bounds(6), sizeVal);

    obstacles(idx, :) = [isCube, x, y, z, sizeVal, color];
end
end

function obstacles = generateRing(bounds, count, minSize, maxSize)
obstacles = zeros(count, 8);
center = [(bounds(1) + bounds(2)) / 2, (bounds(3) + bounds(4)) / 2, (bounds(5) + bounds(6)) / 2];
radius = 0.5 * min(bounds(2) - bounds(1), bounds(4) - bounds(3));
    for idx = 1:count
        angle = rand() * 2 * pi;
        radialJitter = 0.1 * radius * randn();
        x = center(1) + (radius + radialJitter) * cos(angle);
        y = center(2) + (radius + radialJitter) * sin(angle);
        z = center(3) + 0.1 * (bounds(6) - bounds(5)) * randn();
        sizeVal = minSize + (maxSize - minSize) * rand();
        x = clampToBounds(x, bounds(1), bounds(2), sizeVal);
        y = clampToBounds(y, bounds(3), bounds(4), sizeVal);
        z = clampToBounds(z, bounds(5), bounds(6), sizeVal);
        color = [0.6 + 0.4 * rand(), 0.4 + 0.6 * rand(), 0.3 + 0.5 * rand()];
        obstacles(idx, :) = [round(rand()), x, y, z, sizeVal, color];
    end
end

function obstacles = generateTrench(bounds)
% Build a narrow corridor inspired by the Death Star trench run.
count = 80;
obstacles = zeros(count, 8);
centerLineY = (bounds(3) + bounds(4)) / 2;
centerLineZ = (bounds(5) + bounds(6)) / 2;
for idx = 1:count
    x = (bounds(2) - bounds(1)) * rand() + bounds(1);
    y = centerLineY + 0.1 * (bounds(4) - bounds(3)) * randn();
    z = centerLineZ + 0.05 * (bounds(6) - bounds(5)) * randn();
    sizeVal = 0.03 + 0.07 * rand();
    color = [0.5 + 0.5 * rand(), 0.5 + 0.5 * rand(), 0.5 + 0.5 * rand()];
    isCube = mod(idx, 3) == 0;
    obstacles(idx, :) = [isCube, x, y, z, sizeVal, color];
end
end

function value = clampToBounds(value, lower, upper, padding)
%CLAMPTOBOUNDS Keep values inside bounds with padding for obstacle size.
    range = upper - lower;
    maxPadding = max(0, min(padding, range / 2));
    effectiveLower = lower + maxPadding;
    effectiveUpper = upper - maxPadding;
    value = min(max(value, effectiveLower), effectiveUpper);
end
