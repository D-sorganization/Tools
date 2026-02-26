function b = TestIfSE3(mat)
    b = rotation_converter.DistanceToSE3(mat) < 1e-3;
end