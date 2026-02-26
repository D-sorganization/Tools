function X = Xrotz(theta)
    c = cos(theta); s = sin(theta);
    R = [c s 0; -s c 0; 0 0 1];
    X = [R, zeros(3); zeros(3), R];
end