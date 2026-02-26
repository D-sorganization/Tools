function b = TestIfSO3(mat)
    b = rotation_converter.DistanceToSO3(mat) < 1e-3;
end