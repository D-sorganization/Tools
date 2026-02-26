function vcross = crm(v)
    vcross = [rotation_converter.VecToso3(v(1:3)), zeros(3); rotation_converter.VecToso3(v(4:6)), rotation_converter.VecToso3(v(1:3))];
end