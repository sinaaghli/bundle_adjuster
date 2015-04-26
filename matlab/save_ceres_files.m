function save_ceres_files(frame_info,cam,file1,file2,min_match,stereo)

% Defaults
if nargin < 5, min_match = 2; end
if nargin < 6, stereo = false; end

% Count
ncam = length(frame_info);
nlbl = max(cellfun(@(f)max(f.labels),frame_info));
nobs = sum(cellfun(@(f)length(f.labels),frame_info));

% Count observations
obs = zeros(nlbl,1);
for i = 1:ncam
    l = frame_info{i}.labels;
    obs(l) = obs(l) + 1;
end

% Remove marks with too few observations
lmap = -ones(nlbl,1);
keep_ind = (obs >= min_match);
nlbl2 = sum(keep_ind);
lmap(keep_ind) = 0:nlbl2-1;
fprintf('discarding %d of %d labels...\n',nlbl-nlbl2,nlbl)

% After filtering
nobs2 = sum(cellfun(@(f)sum(lmap(f.labels) >= 0),frame_info));

% Open files
f1 = fopen(file1,'w'); % Obs
f2 = fopen(file2,'w'); % Params
fprintf(f1,'%d\n',nobs2);
fprintf(f2,'%d %d\n', ncam, nlbl2);

% Write observations (left)
if ~stereo
    format = '%4d %6d %14.9f %14.9f\n';
    for i = 1:ncam
        frame = frame_info{i};
        for j = 1:length(frame.labels)
            m = lmap(frame.labels(j));
            if m >= 0
                xp = frame.fl(1,j) - cam.lcmod.cx;
                yp = frame.fl(2,j) - cam.lcmod.cy;
                fprintf(f1,format,i-1,m,xp,yp);
            end
        end
    end
else
    % Write left and right observations
    format = '%4d %6d %14.9f %14.9f %14.9f %14.9f\n';
    for i = 1:ncam
        frame = frame_info{i};
        for j = 1:length(frame.labels)
            m = lmap(frame.labels(j));
            if m >= 0
                xp = frame.fl(1,j) - cam.lcmod.cx;
                yp = frame.fl(2,j) - cam.lcmod.cy;
                rj = frame.mr(find(frame.ml == j,1));
                xpr = frame.fr(1,rj) - cam.rcmod.cx;
                ypr = frame.fr(2,rj) - cam.rcmod.cy;
                fprintf(f1,format,i-1,m,xp,yp,xpr,ypr);
            end
        end
    end
end

% Write initial camera parameters
for i = 1:ncam
    frame = frame_info{i};
%    p = frame.pose([4:6,1:3]);
    p = T2Cart(Cart2T(frame.pose) * cam.Twl);
    p = p([4:6,1:3]);
    fprintf(f2,'%f ',p);
    fprintf(f2,'%f 0 0\n',cam.lcmod.fx);
end

% Write initial feature parameters
lestimates = nan(3,nlbl2);
for i = 1:ncam
    frame = frame_info{i};
    l = lmap(frame.labels);
    upd = (l >= 0);
    if ~isempty(upd)
        xcp = frame.tri(:,upd);
        xwp = Cart2T(frame.pose) * [xcp; ones(1,size(xcp,2))];
        lestimates(:,1+l(upd)) = xwp(1:3,:);
    end
end
assert(~any(isnan(lestimates(:))))
fprintf(f2,'%f %f %f\n',lestimates);

fclose(f1);
fclose(f2);

end