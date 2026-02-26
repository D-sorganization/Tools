function d = DistanceToSE3(mat)
    matR = mat(1:3, 1:3);
    if det(matR) > 0
        tmat = [matR'' * matR, zeros(3,1); mat(4,:)];
        d = norm(tmat - eye(4), ''fro'');
    else
        d = 1e9;
    end
end