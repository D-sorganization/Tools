function plotRRT_Static(obstacles, path, nodes, start, goal, bounds)
% plotRRT_Static.m
% Static RRT plotter: red tree branches + blue final path

figure('Color','w','Units','normalized','OuterPosition',[0 0 1 1]);
hold on; grid on; axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
xlim([bounds(1), bounds(2)]);
ylim([bounds(3), bounds(4)]);
zlim([bounds(5), bounds(6)]);
view(3);

% Plot obstacles
[theta,phi] = meshgrid(linspace(0,2*pi,20),linspace(0,pi,20));

for i = 1:size(obstacles,1)
    type = obstacles(i,1);
    x = obstacles(i,2);
    y = obstacles(i,3);
    z = obstacles(i,4);
    size_val = obstacles(i,5);
    color = obstacles(i,6:8);

    if type == 0
        r = size_val;
        xc = x + r*sin(phi).*cos(theta);
        yc = y + r*sin(phi).*sin(theta);
        zc = z + r*cos(phi);
        surf(xc, yc, zc, 'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    else
        [X,Y,Z] = ndgrid([-1 1]*size_val/2, [-1 1]*size_val/2, [-1 1]*size_val/2);
        vertices = [X(:), Y(:), Z(:)];
        faces = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
        patch('Vertices',vertices + [x y z], 'Faces',faces, 'FaceColor',color, 'FaceAlpha',0.3, 'EdgeColor','none');
    end
end

% Plot start and goal
plot3(start(1), start(2), start(3), 'go', 'MarkerSize',10, 'MarkerFaceColor','g');
plot3(goal(1), goal(2), goal(3), 'rx', 'MarkerSize',12, 'LineWidth',2);

% --- Plot all explored branches (RED thin lines) ---
for i = 2:size(nodes,1)
    parent_idx = nodes(i,4); % 4th column = parent index
    if parent_idx ~= 0
        parent_node = nodes(parent_idx,1:3);
        this_node = nodes(i,1:3);
        plot3([parent_node(1) this_node(1)], [parent_node(2) this_node(2)], [parent_node(3) this_node(3)], ...
            'r-', 'LineWidth', 0.5); % Red thin lines
    end
end

% --- Plot final path (BLUE thick line) ---
plot3(path(:,1), path(:,2), path(:,3), 'b-', 'LineWidth', 3);

title('RRT Exploration and Final Path');
legend({'Start','Goal','Explored Tree','Final Path'}, 'Location', 'northeastoutside');
hold off;

end
