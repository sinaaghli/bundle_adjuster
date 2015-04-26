function [Tab,inliers] = RANSACMotion(z, m, p, sigma, cam)
% z - tracked features 
% m - landmarks
% p - desired prob of finding the inliers
% sigma - inlier standard deviation
    opts = optimset('display','off');
    n    = 3;
    x    = 1:length(m);

    % Create context from indices
    function ctx = res_ctx(ind)
        ctx.cmod   = cam.lcmod;
        ctx.Twc    = cam.Twl;
        ctx.prev3d = m(:,ind);
        ctx.curr2d = z(:,ind);
    end

    % Residual function
    function r = residual2d(x,ctx)
        T = Cart2T(x(:));
        pred2d = proj_3d_to_2d(T, ctx.prev3d, ctx.cmod, false);
        r = ctx.curr2d - pred2d;
    end

    % Fit complete xyz-pqr pose
    function pose = fitfn_3d(ind)
        resfn3d = @(y)residual2d(y,res_ctx(ind));
        pose = lsqnonlin(resfn3d,zeros(6,1),[],[],opts);
    end

    % Dist function for 3d
    function d = distfn_3d(y,ind)
        d = sqrt(sum(residual2d(y,res_ctx(ind)) .^ 2, 1));
    end

    [Tab,inliers] = ransac(x, @fitfn_3d, @distfn_3d, n, p, sigma);

end