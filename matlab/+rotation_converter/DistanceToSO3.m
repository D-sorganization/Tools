function d = DistanceToSO3(mat)
    if det(mat) > 0
        d = norm(mat'' * mat - eye(3), ''fro'');
    else
        d = 1e9;
    end
end