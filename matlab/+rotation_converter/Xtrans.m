function X = Xtrans(r)
    X = [eye(3), zeros(3); -rotation_converter.VecToso3(r), eye(3)];
end