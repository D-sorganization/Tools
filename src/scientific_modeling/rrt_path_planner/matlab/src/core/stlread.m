function [V, F] = stlread(filename)
% STLREAD Reads STL file and returns vertices and faces
% Input: filename - path to STL file
% Output: V - vertices (Nx3 matrix)
%         F - faces (Mx3 matrix)

V = [];
F = [];

fid = fopen(filename, 'r');
if fid == -1
    error('File could not be opened, check name or path: %s', filename);
end

% Check if file is binary or ASCII
tline = fgetl(fid);
if contains(tline, 'solid')
    % ASCII STL file
    while ischar(tline)
        if contains(tline, 'vertex')
            temp = sscanf(tline, '%*s %f %f %f');
            V = [V; temp'];
        elseif contains(tline, 'endfacet')
            F = [F; size(V,1)-2 size(V,1)-1 size(V,1)];
        end
        tline = fgetl(fid);
    end
else
    % Binary STL file
    fclose(fid);
    fid = fopen(filename, 'r');
    
    % Skip header (80 bytes)
    fread(fid, 80, 'uchar');
    
    % Read number of triangles (4 bytes)
    num_triangles = fread(fid, 1, 'uint32');
    
    % Read triangles
    for i = 1:num_triangles
        % Skip normal (12 bytes)
        fread(fid, 3, 'float32');
        
        % Read vertices (36 bytes)
        vertices = fread(fid, 9, 'float32');
        V = [V; reshape(vertices, 3, 3)'];
        
        % Skip attribute (2 bytes)
        fread(fid, 1, 'uint16');
    end
    
    % Create faces
    F = reshape(1:size(V,1), 3, [])';
end

fclose(fid);

% Remove duplicate vertices
[V, ~, indexMap] = unique(V, 'rows');
F = indexMap(F);

end 