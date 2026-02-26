function AdT = Adjoint(T)
    [R, p] = rotation_converter.TransToRp(T);
    AdT = [R, zeros(3); rotation_converter.VecToso3(p)*R, R];
end