function T = ProjectToSE3(mat)
    R = rotation_converter.ProjectToSO3(mat(1:3, 1:3));
    T = rotation_converter.RpToTrans(R, mat(1:3, 4));
end