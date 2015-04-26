function p = triangulate(uvl, uvr, Twl, Twr, lcmod, rcmod)
    d = uvl(1,:) - uvr(1,:); % disparities
    b = norm(Twl(1:3,4) - Twr(1:3,4)); % baseline
    f = lcmod.fx; % focal length
    p(3,:) = b*f ./ d;
    p(1,:) = (uvl(1,:) - lcmod.cx) .* p(3,:) ./ f;
    p(2,:) = (uvl(2,:) - lcmod.cy) .* p(3,:) ./ f;
    p = Twl * [p;ones(1,size(p,2))];
    p = p(1:3,:);
end
