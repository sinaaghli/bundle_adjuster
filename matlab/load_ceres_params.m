function [cams,points,obs] = load_ceres_params(fname)

f = fopen(fname);

% Read size
row1 = fscanf(f,'%d',2);
ncam = row1(1);
nlbl = row1(2);

% Read cameras
cams = zeros(9,ncam);
cams(:) = fscanf(f,'%f',9*ncam);

% Read points
points = zeros(3,nlbl);
points(:) = fscanf(f,'%f',3*nlbl);

fclose(f);

end