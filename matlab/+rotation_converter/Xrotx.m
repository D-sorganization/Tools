function X = Xrotx(theta)
    c = cos(theta); s = sin(theta);
    R = [1 0 0; 0 c s; 0 -s c];
    X = [R, zeros(3); zeros(3), R];
end