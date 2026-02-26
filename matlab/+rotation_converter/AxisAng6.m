function [S, theta] = AxisAng6(expc6)
    theta = norm(expc6(1:3));
    if theta < 1e-12
        theta = norm(expc6(4:6));
    end
    S = expc6 / theta;
end