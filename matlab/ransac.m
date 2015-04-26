function [M,inliers] = ransac(x, fittingfn, distfn, n, p, t)
[rows,npts] = size(x);
bestM = nan;
trialcount = 0;
bestscore  = -1;
k = 1;
while k > trialcount
    ind = ceil(rand(1,n)*npts);
    M = feval(fittingfn, x(:,ind));
    d = feval(distfn, M , x);
    inliers = find(abs(d) < t);
    ninliers = length(inliers);
    if ninliers > bestscore
        bestscore = ninliers;
        bestinliers = inliers;
        bestM = M;
        
        w = ninliers / npts;
        k = log(1-p) / log(1 - w^n);
%        fprintf('k = %.1f\n',k)
        k = max(5,k);
    end
    trialcount = trialcount + 1;
end
M = feval(fittingfn, x(:,bestinliers));
d = feval(distfn, M , x);
inliers = find(abs(d) < t);
