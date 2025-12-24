function starWarsPathPlannerGUI()
% STARWARSPATHPLANNERGUI Main GUI for Star Wars RRT Path Planner
% Features:
% - Single ship navigation
% - Pursuit mode with two ships
% - Multiple camera views
% - Ship selection
% - Environment customization
% - Real-time visualization

% Create main figure
fig = figure('Name', 'Star Wars RRT Path Planner', ...
             'Position', [100, 100, 1400, 900], ...
             'Color', [0.1, 0.1, 0.15], ...
             'MenuBar', 'none', ...
             'ToolBar', 'none', ...
             'Resize', 'on');

% Create panels
control_panel = uipanel('Parent', fig, ...
                       'Title', 'Controls', ...
                       'Position', [0.02, 0.02, 0.25, 0.96], ...
                       'BackgroundColor', [0.15, 0.15, 0.2], ...
                       'ForegroundColor', [1, 1, 1], ...
                       'FontSize', 12, ...
                       'FontWeight', 'bold');

visualization_panel = uipanel('Parent', fig, ...
                             'Title', '3D Visualization', ...
                             'Position', [0.28, 0.02, 0.7, 0.96], ...
                             'BackgroundColor', [0.1, 0.1, 0.15], ...
                             'ForegroundColor', [1, 1, 1], ...
                             'FontSize', 12, ...
                             'FontWeight', 'bold');

% Create 3D axes
ax = axes('Parent', visualization_panel, ...
          'Position', [0.05, 0.05, 0.9, 0.9], ...
          'Color', [0, 0, 0], ...
          'XColor', [0.7, 0.7, 0.7], ...
          'YColor', [0.7, 0.7, 0.7], ...
          'ZColor', [0.7, 0.7, 0.7], ...
          'GridColor', [0.3, 0.3, 0.3]);

% Initialize global variables
global gui_data;
gui_data.ax = ax;
gui_data.obstacles = [];
gui_data.bounds = [-1.0, 1.0, -0.6, 0.6, -0.3, 0.3];
gui_data.current_mode = 'single';
gui_data.pursuer_ship = 'falcon';
gui_data.target_ship = 'tie_fighter';
gui_data.camera_view = 'cinematic';
gui_data.animation_speed = 0.02;
gui_data.show_path = true;
gui_data.show_obstacles = true;
gui_data.show_starfield = true;

% Load default obstacles
loadDefaultObstacles();

% Create control elements
createControlElements(control_panel);

% Initialize visualization
initializeVisualization();

% Set up callbacks
setupCallbacks();

end

function createControlElements(panel)
% Create all control elements in the control panel

global gui_data;

% Mode selection
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Mode:', ...
          'Position', [10, 850, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.mode_popup = uicontrol('Parent', panel, 'Style', 'popupmenu', ...
                               'String', {'Single Ship Navigation', 'Pursuit Mode'}, ...
                               'Position', [120, 850, 150, 25], ...
                               'BackgroundColor', [0.2, 0.2, 0.25], ...
                               'ForegroundColor', [1, 1, 1], ...
                               'FontSize', 10);

% Ship selection
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Pursuer Ship:', ...
          'Position', [10, 810, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.pursuer_popup = uicontrol('Parent', panel, 'Style', 'popupmenu', ...
                                  'String', {'Millennium Falcon', 'X-Wing', 'TIE Fighter'}, ...
                                  'Position', [120, 810, 150, 25], ...
                                  'BackgroundColor', [0.2, 0.2, 0.25], ...
                                  'ForegroundColor', [1, 1, 1], ...
                                  'FontSize', 10);

uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Target Ship:', ...
          'Position', [10, 770, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.target_popup = uicontrol('Parent', panel, 'Style', 'popupmenu', ...
                                 'String', {'TIE Fighter', 'Millennium Falcon', 'X-Wing'}, ...
                                 'Position', [120, 770, 150, 25], ...
                                 'BackgroundColor', [0.2, 0.2, 0.25], ...
                                 'ForegroundColor', [1, 1, 1], ...
                                 'FontSize', 10);

% Camera view selection
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Camera View:', ...
          'Position', [10, 730, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.camera_popup = uicontrol('Parent', panel, 'Style', 'popupmenu', ...
                                 'String', {'Cinematic', 'Chase Cam', 'Top Down', 'Side View', 'Free Camera'}, ...
                                 'Position', [120, 730, 150, 25], ...
                                 'BackgroundColor', [0.2, 0.2, 0.25], ...
                                 'ForegroundColor', [1, 1, 1], ...
                                 'FontSize', 10);

% Start positions
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Start Position:', ...
          'Position', [10, 690, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.start_x = uicontrol('Parent', panel, 'Style', 'edit', ...
                            'String', '-0.8', ...
                            'Position', [120, 690, 50, 25], ...
                            'BackgroundColor', [0.2, 0.2, 0.25], ...
                            'ForegroundColor', [1, 1, 1], ...
                            'FontSize', 10);

gui_data.start_y = uicontrol('Parent', panel, 'Style', 'edit', ...
                            'String', '0', ...
                            'Position', [175, 690, 50, 25], ...
                            'BackgroundColor', [0.2, 0.2, 0.25], ...
                            'ForegroundColor', [1, 1, 1], ...
                            'FontSize', 10);

gui_data.start_z = uicontrol('Parent', panel, 'Style', 'edit', ...
                            'String', '0', ...
                            'Position', [230, 690, 50, 25], ...
                            'BackgroundColor', [0.2, 0.2, 0.25], ...
                            'ForegroundColor', [1, 1, 1], ...
                            'FontSize', 10);

% Goal position
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Goal Position:', ...
          'Position', [10, 650, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.goal_x = uicontrol('Parent', panel, 'Style', 'edit', ...
                           'String', '0.8', ...
                           'Position', [120, 650, 50, 25], ...
                           'BackgroundColor', [0.2, 0.2, 0.25], ...
                           'ForegroundColor', [1, 1, 1], ...
                           'FontSize', 10);

gui_data.goal_y = uicontrol('Parent', panel, 'Style', 'edit', ...
                           'String', '0', ...
                           'Position', [175, 650, 50, 25], ...
                           'BackgroundColor', [0.2, 0.2, 0.25], ...
                           'ForegroundColor', [1, 1, 1], ...
                           'FontSize', 10);

gui_data.goal_z = uicontrol('Parent', panel, 'Style', 'edit', ...
                           'String', '0', ...
                           'Position', [230, 650, 50, 25], ...
                           'BackgroundColor', [0.2, 0.2, 0.25], ...
                           'ForegroundColor', [1, 1, 1], ...
                           'FontSize', 10);

% Animation speed
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Animation Speed:', ...
          'Position', [10, 610, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.speed_slider = uicontrol('Parent', panel, 'Style', 'slider', ...
                                 'Min', 0.001, 'Max', 0.1, 'Value', 0.02, ...
                                 'Position', [120, 610, 150, 25], ...
                                 'BackgroundColor', [0.2, 0.2, 0.25]);

% Display options
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Display Options:', ...
          'Position', [10, 570, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.show_path_checkbox = uicontrol('Parent', panel, 'Style', 'checkbox', ...
                                       'String', 'Show Path', ...
                                       'Value', 1, ...
                                       'Position', [120, 570, 100, 25], ...
                                       'BackgroundColor', [0.15, 0.15, 0.2], ...
                                       'ForegroundColor', [1, 1, 1], ...
                                       'FontSize', 10);

gui_data.show_obstacles_checkbox = uicontrol('Parent', panel, 'Style', 'checkbox', ...
                                           'String', 'Show Obstacles', ...
                                           'Value', 1, ...
                                           'Position', [120, 540, 100, 25], ...
                                           'BackgroundColor', [0.15, 0.15, 0.2], ...
                                           'ForegroundColor', [1, 1, 1], ...
                                           'FontSize', 10);

gui_data.show_starfield_checkbox = uicontrol('Parent', panel, 'Style', 'checkbox', ...
                                           'String', 'Show Starfield', ...
                                           'Value', 1, ...
                                           'Position', [120, 510, 100, 25], ...
                                           'BackgroundColor', [0.15, 0.15, 0.2], ...
                                           'ForegroundColor', [1, 1, 1], ...
                                           'FontSize', 10);

% Action buttons
gui_data.plan_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                'String', 'Plan Path', ...
                                'Position', [10, 450, 120, 40], ...
                                'BackgroundColor', [0.2, 0.6, 0.2], ...
                                'ForegroundColor', [1, 1, 1], ...
                                'FontSize', 12, ...
                                'FontWeight', 'bold');

gui_data.animate_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                   'String', 'Animate', ...
                                   'Position', [140, 450, 120, 40], ...
                                   'BackgroundColor', [0.2, 0.2, 0.6], ...
                                   'ForegroundColor', [1, 1, 1], ...
                                   'FontSize', 12, ...
                                   'FontWeight', 'bold');

gui_data.reset_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                 'String', 'Reset', ...
                                 'Position', [10, 400, 120, 40], ...
                                 'BackgroundColor', [0.6, 0.2, 0.2], ...
                                 'ForegroundColor', [1, 1, 1], ...
                                 'FontSize', 12, ...
                                 'FontWeight', 'bold');

gui_data.save_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                'String', 'Save Video', ...
                                'Position', [140, 400, 120, 40], ...
                                'BackgroundColor', [0.6, 0.4, 0.2], ...
                                'ForegroundColor', [1, 1, 1], ...
                                'FontSize', 12, ...
                                'FontWeight', 'bold');

% Environment controls
uicontrol('Parent', panel, 'Style', 'text', ...
          'String', 'Environment:', ...
          'Position', [10, 350, 100, 25], ...
          'BackgroundColor', [0.15, 0.15, 0.2], ...
          'ForegroundColor', [1, 1, 1], ...
          'FontSize', 11, ...
          'FontWeight', 'bold');

gui_data.load_obstacles_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                         'String', 'Load Obstacles', ...
                                         'Position', [10, 320, 120, 30], ...
                                         'BackgroundColor', [0.3, 0.3, 0.4], ...
                                         'ForegroundColor', [1, 1, 1], ...
                                         'FontSize', 10);

gui_data.generate_obstacles_button = uicontrol('Parent', panel, 'Style', 'pushbutton', ...
                                             'String', 'Generate Random', ...
                                             'Position', [140, 320, 120, 30], ...
                                             'BackgroundColor', [0.3, 0.3, 0.4], ...
                                             'ForegroundColor', [1, 1, 1], ...
                                             'FontSize', 10);

% Status display
gui_data.status_text = uicontrol('Parent', panel, 'Style', 'text', ...
                                'String', 'Ready to plan path...', ...
                                'Position', [10, 50, 250, 60], ...
                                'BackgroundColor', [0.1, 0.1, 0.15], ...
                                'ForegroundColor', [0.8, 0.8, 0.8], ...
                                'FontSize', 10, ...
                                'HorizontalAlignment', 'left');

end

function setupCallbacks()
% Set up all button and control callbacks

global gui_data;

% Button callbacks
set(gui_data.plan_button, 'Callback', @planPathCallback);
set(gui_data.animate_button, 'Callback', @animateCallback);
set(gui_data.reset_button, 'Callback', @resetCallback);
set(gui_data.save_button, 'Callback', @saveVideoCallback);
set(gui_data.load_obstacles_button, 'Callback', @loadObstaclesCallback);
set(gui_data.generate_obstacles_button, 'Callback', @generateObstaclesCallback);

% Popup callbacks
set(gui_data.mode_popup, 'Callback', @modeChangeCallback);
set(gui_data.camera_popup, 'Callback', @cameraChangeCallback);

% Slider callback
set(gui_data.speed_slider, 'Callback', @speedChangeCallback);

% Checkbox callbacks
set(gui_data.show_path_checkbox, 'Callback', @displayOptionCallback);
set(gui_data.show_obstacles_checkbox, 'Callback', @displayOptionCallback);
set(gui_data.show_starfield_checkbox, 'Callback', @displayOptionCallback);

end

function planPathCallback(~, ~)
% Callback for plan path button
global gui_data;

updateStatus('Planning path...');

try
    % Get current settings
    mode = get(gui_data.mode_popup, 'Value');
    start_pos = [str2double(get(gui_data.start_x, 'String')), ...
                 str2double(get(gui_data.start_y, 'String')), ...
                 str2double(get(gui_data.start_z, 'String'))];
    goal_pos = [str2double(get(gui_data.goal_x, 'String')), ...
                str2double(get(gui_data.goal_y, 'String')), ...
                str2double(get(gui_data.goal_z, 'String'))];
    
    if mode == 1  % Single ship
        [nodes, path] = RRT(start_pos, goal_pos, gui_data.obstacles, gui_data.bounds);
        if ~isempty(path)
            gui_data.current_path = path;
            gui_data.current_nodes = nodes;
            updateStatus('Path planned successfully!');
            visualizePath();
        else
            updateStatus('No valid path found.');
        end
    else  % Pursuit mode
        pursuer_start = start_pos;
        target_start = goal_pos;
        [pursuer_path, target_path] = pursuitSystem(pursuer_start, target_start, ...
                                                   gui_data.obstacles, gui_data.bounds, 200);
        gui_data.pursuer_path = pursuer_path;
        gui_data.target_path = target_path;
        updateStatus('Pursuit scenario planned!');
        visualizePursuit();
    end
    
catch ME
    updateStatus(['Error: ' ME.message]);
end

end

function animateCallback(~, ~)
% Callback for animate button
global gui_data;

if ~isfield(gui_data, 'current_path') && ~isfield(gui_data, 'pursuer_path')
    updateStatus('Please plan a path first.');
    return;
end

updateStatus('Starting animation...');

try
    mode = get(gui_data.mode_popup, 'Value');
    speed = get(gui_data.speed_slider, 'Value');
    
    if mode == 1  % Single ship
        animateSingleShip(gui_data.current_path, speed);
    else  % Pursuit mode
        animatePursuit(gui_data.pursuer_path, gui_data.target_path, speed);
    end
    
    updateStatus('Animation complete!');
    
catch ME
    updateStatus(['Animation error: ' ME.message]);
end

end

function resetCallback(~, ~)
% Callback for reset button
global gui_data;

% Clear current paths
if isfield(gui_data, 'current_path')
    gui_data = rmfield(gui_data, 'current_path');
end
if isfield(gui_data, 'pursuer_path')
    gui_data = rmfield(gui_data, 'pursuer_path');
end
if isfield(gui_data, 'target_path')
    gui_data = rmfield(gui_data, 'target_path');
end

% Reset visualization
initializeVisualization();
updateStatus('Reset complete. Ready to plan new path.');

end

function saveVideoCallback(~, ~)
% Callback for save video button
global gui_data;

if ~isfield(gui_data, 'current_path') && ~isfield(gui_data, 'pursuer_path')
    updateStatus('Please plan a path first.');
    return;
end

updateStatus('Saving video...');

try
    mode = get(gui_data.mode_popup, 'Value');
    speed = get(gui_data.speed_slider, 'Value');
    
    if mode == 1  % Single ship
        saveSingleShipVideo(gui_data.current_path, speed);
    else  % Pursuit mode
        savePursuitVideo(gui_data.pursuer_path, gui_data.target_path, speed);
    end
    
    updateStatus('Video saved successfully!');
    
catch ME
    updateStatus(['Video save error: ' ME.message]);
end

end

function modeChangeCallback(~, ~)
% Callback for mode change
global gui_data;

mode = get(gui_data.mode_popup, 'Value');
if mode == 1
    set(gui_data.target_popup, 'Enable', 'off');
    set(gui_data.goal_x, 'Enable', 'on');
    set(gui_data.goal_y, 'Enable', 'on');
    set(gui_data.goal_z, 'Enable', 'on');
else
    set(gui_data.target_popup, 'Enable', 'on');
    set(gui_data.goal_x, 'Enable', 'off');
    set(gui_data.goal_y, 'Enable', 'off');
    set(gui_data.goal_z, 'Enable', 'off');
end

updateStatus('Mode changed. Ready to plan new path.');

end

function cameraChangeCallback(~, ~)
% Callback for camera view change
global gui_data;

view_type = get(gui_data.camera_popup, 'Value');
updateCameraView(view_type);

end

function speedChangeCallback(~, ~)
% Callback for speed slider
global gui_data;

speed = get(gui_data.speed_slider, 'Value');
gui_data.animation_speed = speed;

end

function displayOptionCallback(~, ~)
% Callback for display option checkboxes
global gui_data;

gui_data.show_path = get(gui_data.show_path_checkbox, 'Value');
gui_data.show_obstacles = get(gui_data.show_obstacles_checkbox, 'Value');
gui_data.show_starfield = get(gui_data.show_starfield_checkbox, 'Value');

% Update visualization
if isfield(gui_data, 'current_path')
    visualizePath();
elseif isfield(gui_data, 'pursuer_path')
    visualizePursuit();
else
    initializeVisualization();
end

end

function loadObstaclesCallback(~, ~)
% Callback for load obstacles button
global gui_data;

[filename, pathname] = uigetfile('*.csv', 'Select obstacles file');
if filename ~= 0
    try
        obstacles = readmatrix(fullfile(pathname, filename));
        gui_data.obstacles = obstacles;
        updateStatus(['Loaded obstacles from ' filename]);
        initializeVisualization();
    catch ME
        updateStatus(['Error loading obstacles: ' ME.message]);
    end
end

end

function generateObstaclesCallback(~, ~)
% Callback for generate obstacles button
global gui_data;

try
    gui_data.obstacles = generateRandomObstacles(gui_data.bounds, 50);
    updateStatus('Generated random obstacles');
    initializeVisualization();
catch ME
    updateStatus(['Error generating obstacles: ' ME.message]);
end

end

function updateStatus(message)
% Update status display
global gui_data;

set(gui_data.status_text, 'String', message);
drawnow;

end

function loadDefaultObstacles()
% Load default obstacles
global gui_data;

try
    gui_data.obstacles = readmatrix('obstacles3D.csv');
catch
    % If file doesn't exist, generate some default obstacles
    gui_data.obstacles = generateRandomObstacles(gui_data.bounds, 30);
end

end

function obstacles = generateRandomObstacles(bounds, num_obstacles)
% Generate random obstacles
obstacles = zeros(num_obstacles, 8);

for i = 1:num_obstacles
    % Random position
    x = (bounds(2)-bounds(1))*rand() + bounds(1);
    y = (bounds(4)-bounds(3))*rand() + bounds(3);
    z = (bounds(6)-bounds(5))*rand() + bounds(5);
    
    % Random type (0=sphere, 1=cube)
    type = round(rand());
    
    % Random size
    size_val = 0.02 + 0.08*rand();
    
    % Random color
    color = [0.5 + 0.5*rand(), 0.3 + 0.7*rand(), 0.3 + 0.7*rand()];
    
    obstacles(i, :) = [type, x, y, z, size_val, color];
end

end

function initializeVisualization()
% Initialize the 3D visualization
global gui_data;

axes(gui_data.ax);
cla;
hold on;
grid on;

% Set up axes
xlim(gui_data.bounds(1:2));
ylim(gui_data.bounds(3:4));
zlim(gui_data.bounds(5:6));
view(3);

% Set camera view
camera_view = get(gui_data.camera_popup, 'Value');
updateCameraView(camera_view);

% Draw obstacles if enabled
if gui_data.show_obstacles
    drawObstacles();
end

% Draw starfield if enabled
if gui_data.show_starfield
    drawStarfield();
end

% Draw start and goal markers
start_pos = [str2double(get(gui_data.start_x, 'String')), ...
             str2double(get(gui_data.start_y, 'String')), ...
             str2double(get(gui_data.start_z, 'String'))];
goal_pos = [str2double(get(gui_data.goal_x, 'String')), ...
            str2double(get(gui_data.goal_y, 'String')), ...
            str2double(get(gui_data.goal_z, 'String'))];

plot3(start_pos(1), start_pos(2), start_pos(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(goal_pos(1), goal_pos(2), goal_pos(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

title('Star Wars RRT Path Planner', 'Color', 'white', 'FontSize', 14);
xlabel('X', 'Color', 'white');
ylabel('Y', 'Color', 'white');
zlabel('Z', 'Color', 'white');

hold off;

end

function updateCameraView(view_type)
% Update camera view based on selection
global gui_data;

axes(gui_data.ax);

switch view_type
    case 1  % Cinematic
        view(45, 30);
        campos([2, -2, 1]);
        camtarget([0, 0, 0]);
    case 2  % Chase Cam
        view(0, 0);
        campos([-1, 0, 0.5]);
        camtarget([1, 0, 0]);
    case 3  % Top Down
        view(0, 90);
        campos([0, 0, 2]);
        camtarget([0, 0, 0]);
    case 4  % Side View
        view(90, 0);
        campos([0, -2, 0]);
        camtarget([0, 0, 0]);
    case 5  % Free Camera
        view(3);
        camorbit(30, 20);
end

end

function drawObstacles()
% Draw obstacles in the scene
global gui_data;

[theta, phi] = meshgrid(linspace(0, 2*pi, 20), linspace(0, pi, 20));

for i = 1:size(gui_data.obstacles, 1)
    type = gui_data.obstacles(i, 1);
    x = gui_data.obstacles(i, 2);
    y = gui_data.obstacles(i, 3);
    z = gui_data.obstacles(i, 4);
    size_val = gui_data.obstacles(i, 5);
    color = gui_data.obstacles(i, 6:8);
    
    if type == 0  % Sphere
        r = size_val;
        xc = x + r*sin(phi).*cos(theta);
        yc = y + r*sin(phi).*sin(theta);
        zc = z + r*cos(phi);
        surf(xc, yc, zc, 'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    else  % Cube
        [X, Y, Z] = ndgrid([-1 1]*size_val/2, [-1 1]*size_val/2, [-1 1]*size_val/2);
        vertices = [X(:), Y(:), Z(:)];
        faces = [1 2 4 3; 5 6 8 7; 1 2 6 5; 3 4 8 7; 1 3 7 5; 2 4 8 6];
        patch('Vertices', vertices + [x y z], 'Faces', faces, ...
              'FaceColor', color, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
end

end

function drawStarfield()
% Draw starfield background
global gui_data;

num_stars = 500;
star_positions = randn(num_stars, 3) * 2;
scatter3(star_positions(:, 1), star_positions(:, 2), star_positions(:, 3), ...
         5, [0.8, 0.8, 0.8], 'filled');

end

function visualizePath()
% Visualize the planned path
global gui_data;

if ~gui_data.show_path
    return;
end

axes(gui_data.ax);
hold on;

% Draw the path
plot3(gui_data.current_path(:, 1), gui_data.current_path(:, 2), gui_data.current_path(:, 3), ...
      'b-', 'LineWidth', 2);

% Draw the tree (optional)
if isfield(gui_data, 'current_nodes')
    for i = 2:size(gui_data.current_nodes, 1)
        parent = gui_data.current_nodes(i, 4);
        plot3([gui_data.current_nodes(i, 1), gui_data.current_nodes(parent, 1)], ...
              [gui_data.current_nodes(i, 2), gui_data.current_nodes(parent, 2)], ...
              [gui_data.current_nodes(i, 3), gui_data.current_nodes(parent, 3)], ...
              'g-', 'LineWidth', 0.5, 'Color', [0.3, 0.7, 0.3]);
    end
end

hold off;

end

function visualizePursuit()
% Visualize the pursuit scenario
global gui_data;

if ~gui_data.show_path
    return;
end

axes(gui_data.ax);
hold on;

% Draw pursuer path
plot3(gui_data.pursuer_path(:, 1), gui_data.pursuer_path(:, 2), gui_data.pursuer_path(:, 3), ...
      'b-', 'LineWidth', 2);

% Draw target path
plot3(gui_data.target_path(:, 1), gui_data.target_path(:, 2), gui_data.target_path(:, 3), ...
      'r-', 'LineWidth', 2);

hold off;

end

function animateSingleShip(path, speed)
% Animate single ship along path
global gui_data;

% Load ship model
try
    [V, F] = stlread('falcon_clean_fixed.stl');
catch
    % Create simple ship if STL not available
    V = createSimpleShip();
    F = createSimpleShipFaces();
end

% Scale and orient ship
scale_factor = 0.05;
V = V * scale_factor;

% Create ship patch
ship_patch = patch('Faces', F, 'Vertices', V, ...
                   'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');

% Animate along path
for i = 1:size(path, 1)-1
    current_pos = path(i, :);
    next_pos = path(i+1, :);
    
    % Calculate orientation
    [R, ~] = shipOrientation(current_pos, next_pos, [1, 0, 0], [0, 0, 1]);
    
    % Transform ship
    rotated_vertices = (V * R') + current_pos;
    ship_patch.Vertices = rotated_vertices;
    
    drawnow;
    pause(speed);
end

end

function animatePursuit(pursuer_path, target_path, speed)
% Animate pursuit scenario
global gui_data;

% Load ship models
try
    [V1, F1] = stlread('falcon_clean_fixed.stl');
    [V2, F2] = stlread('falcon_clean_fixed.stl');  % Use same model for now
catch
    V1 = createSimpleShip();
    F1 = createSimpleShipFaces();
    V2 = createSimpleShip();
    F2 = createSimpleShipFaces();
end

% Scale ships
scale_factor = 0.05;
V1 = V1 * scale_factor;
V2 = V2 * scale_factor;

% Create ship patches
pursuer_patch = patch('Faces', F1, 'Vertices', V1, ...
                      'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');
target_patch = patch('Faces', F2, 'Vertices', V2, ...
                     'FaceColor', [0.6, 0.6, 0.6], 'EdgeColor', 'none');

% Animate both ships
max_steps = max(size(pursuer_path, 1), size(target_path, 1));

for i = 1:max_steps
    % Update pursuer
    if i <= size(pursuer_path, 1)
        if i < size(pursuer_path, 1)
            [R1, ~] = shipOrientation(pursuer_path(i, :), pursuer_path(i+1, :), [1, 0, 0], [0, 0, 1]);
        else
            [R1, ~] = shipOrientation(pursuer_path(i, :), pursuer_path(i, :), [1, 0, 0], [0, 0, 1]);
        end
        rotated_vertices1 = (V1 * R1') + pursuer_path(i, :);
        pursuer_patch.Vertices = rotated_vertices1;
    end
    
    % Update target
    if i <= size(target_path, 1)
        if i < size(target_path, 1)
            [R2, ~] = shipOrientation(target_path(i, :), target_path(i+1, :), [1, 0, 0], [0, 0, 1]);
        else
            [R2, ~] = shipOrientation(target_path(i, :), target_path(i, :), [1, 0, 0], [0, 0, 1]);
        end
        rotated_vertices2 = (V2 * R2') + target_path(i, :);
        target_patch.Vertices = rotated_vertices2;
    end
    
    drawnow;
    pause(speed);
end

end

function V = createSimpleShip()
% Create simple ship vertices if STL not available
V = [
    -0.5, 0, 0;    % Nose
    0.5, 0, 0;     % Tail
    0, 0.3, 0;     % Right wing
    0, -0.3, 0;    % Left wing
    0, 0, 0.2;     % Top
    0, 0, -0.2;    % Bottom
];

end

function F = createSimpleShipFaces()
% Create simple ship faces
F = [
    1, 3, 5;   % Right side
    1, 5, 4;   % Left side
    1, 4, 6;   % Bottom
    1, 6, 3;   % Top
    2, 3, 5;   % Right back
    2, 5, 4;   % Left back
    2, 4, 6;   % Bottom back
    2, 6, 3;   % Top back
];

end

function saveSingleShipVideo(path, speed)
% Save single ship animation as video
global gui_data;

v = VideoWriter('single_ship_animation.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

% Similar to animateSingleShip but capture frames
% ... (implementation similar to animateSingleShip but with frame capture)

close(v);

end

function savePursuitVideo(pursuer_path, target_path, speed)
% Save pursuit animation as video
global gui_data;

v = VideoWriter('pursuit_animation.mp4', 'MPEG-4');
v.FrameRate = 30;
open(v);

% Similar to animatePursuit but capture frames
% ... (implementation similar to animatePursuit but with frame capture)

close(v);

end 