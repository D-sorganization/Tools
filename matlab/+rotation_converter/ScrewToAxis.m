function S = ScrewToAxis(q, s, h)
    S = [s(:); cross(q(:), s(:)) + h * s(:)];
end