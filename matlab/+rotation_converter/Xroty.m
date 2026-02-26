function X = Xroty(theta)
    c = cos(theta); s = sin(theta);
    R = [c 0 -s; 0 1 0; s 0 c];
    X = [R, zeros(3); zeros(3), R];
end