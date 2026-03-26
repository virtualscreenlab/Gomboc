% Load STL file
fv = stlread('gomboc-reverse-engineered.stl');

% Display the structure
disp('STL data structure:');
disp(fv);

% Visualize the model
figure;
trisurf(fv.ConnectivityList, fv.Points(:,1), fv.Points(:,2), fv.Points(:,3), ...
    'FaceColor', [0.8 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
axis equal;
lighting gouraud;
camlight;
% title('Gomboc Model');
xlabel('X'); ylabel('Y'); zlabel('Z');
% grid on;
axis off
set(gcf, 'Color', 'white');  % White background for figure
view(3);

% Optional: Display additional information
fprintf('\nModel Information:\n');
fprintf('Number of vertices: %d\n', size(fv.Points, 1));
fprintf('Number of faces: %d\n', size(fv.ConnectivityList, 1));