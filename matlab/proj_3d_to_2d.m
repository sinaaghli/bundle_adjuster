function uv = proj_3d_to_2d( Twc, xwp, cmod, trim_outside,smooth)
% Twc  - Transform world-camera
% xwp  - poses in world coordinates
% cmod - camera
% trim_outside - remove points outside camera

if nargin <= 3, trim_outside = true; end
if nargin <= 4, smooth = true; end

if isstruct(cmod)
    K = [cmod.fx cmod.s cmod.cx; 0 cmod.fy cmod.cy; 0 0 1];
else
    K = cmod;
end

xcp = Twc \ [xwp;ones(1,size(xwp,2))];

uvw = K * xcp(1:3,:);

% This moves the asymptote at w = 0 to w = -inf
eps = 0.1;
if smooth && ~trim_outside
    small_ind = find(uvw(3,:) < eps);
    if small_ind
        uvw(3,small_ind) = eps * exp(uvw(3,small_ind) - eps);
    end
end

%uv = uvw(1:2,:) ./ uvw([3,3],:);
uv = bsxfun(@rdivide, uvw(1:2,:), uvw(3,:));% uvw(1:2,:) ./ uvw([3,3],:);

if trim_outside
    % Behind camera
    behind = find(uvw(3,:) <= 0);

    % Outside image
    out1 = find(uv(1,:) < 0);
    out2 = find(uv(1,:) > cmod.imagewidth);
    out3 = find(uv(2,:) < 0);
    out4 = find(uv(2,:) > cmod.imageheight);
    
%    uv(:,[behind, out1, out2, out3, out4]) = [];
    uv(:,[behind, out1, out2, out3, out4]) = nan;
end

end