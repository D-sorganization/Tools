function R = ProjectToSO3(mat)
    [U, ~, V] = svd(mat);
    R = U * V'';
    if det(R) < 0
        R(:,3) = -R(:,3);
    end
end