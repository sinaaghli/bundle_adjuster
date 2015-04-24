% Show matches
show_matches = false;
show_triangulation = false;
show_trajectory = true;
show_temporal = true;
show_point_traj = false;

% Options
discard_ransac_outliers = true;
discard_non_LR_matches = false;
discard_non_LR_temporal = false;
ransac_sigma = 10;
ransac_p = 0.99;
no_labelling = false;

% Limit disparities
min_x_disp = 0.1;
max_x_disp = 20;
min_y_disp = -10;
max_y_disp = 10;

% Images
if ~exist('limages','var') && ~exist('rimages','var')
    limages = dir([CBPATH,'images/left*']);
    rimages = dir([CBPATH,'images/right*']);
    limages = {limages.name};
    rimages = {rimages.name};
end

% Frame cell array
frame_info = cell(length(limages),1);

% Trajectory plot
if show_trajectory
    traj = zeros(3,length(limages));
    figure, traj_ax = axes;
end

% Temporal plot
if show_temporal
    figure
    temp_ax = axes;
    hold(temp_ax, 'on');
    temp_im = imshow(imread([CBPATH,'images/',limages{1}]));
    temp_lines = [];
end

% Feature match loop
for n = 1:length(limages)
    % Read images and extract features
    img_l = imread([CBPATH,'images/',limages{n}]);
    img_r = imread([CBPATH,'images/',rimages{n}]);
    [feats_l,desc_l] = vl_sift(im2single(img_l));
    [feats_r,desc_r] = vl_sift(im2single(img_r));

    % Spatial matches and descriptors
    match = vl_ubcmatch(desc_l,desc_r);
    match_l = match(1,:);
    match_r = match(2,:);

    % Discard non-matches
    if discard_non_LR_matches
        feats_l = feats_l(:,match_l);
        feats_r = feats_r(:,match_r);
        desc_l  = desc_l(:,match_l);
        desc_r  = desc_r(:,match_r);
        match_l = 1:length(match_l);
        match_r = 1:length(match_r);
    end

    % Filter bad disparities
    dx = feats_l(1,match_l) - feats_r(1,match_r);
    dy = feats_l(2,match_l) - feats_r(2,match_r);
    bad = (dx < min_x_disp | dx > max_x_disp | ...
           dy < min_y_disp | dy > max_y_disp);
    match_l(:,bad) = [];
    match_r(:,bad) = [];

    % Triangulate (relative rig)
    tri_m = triangulate(feats_l(1:2,match_l), feats_r(1:2,match_r), ...
                        cam.Twl, cam.Twr, cam.lcmod, cam.rcmod);
    tri = nan(3,size(feats_l,2));
    tri(:,match_l) = tri_m;
    
    % Store feats in curr struct
    curr.fl = feats_l;
    curr.fr = feats_r;
    curr.dl = desc_l;
    curr.dr = desc_r;
    curr.mr = match_r;
    curr.ml = match_l;
    curr.tri = tri;

    % Temporal match
    if n == 1
        curr.pose = [0 0 0 0 0 0]';
        if ~no_labelling
            last_label = length(curr.fl);
            curr.labels = 1:last_label;
        end
    elseif n > 1
        % Filter features not matched in both left and right frames
        if discard_non_LR_temporal
            % Temporal similarities
            tmatch_l = vl_ubcmatch(prev.dl(prev.ml),curr.dl(curr.ml));
            tmatch_r = vl_ubcmatch(prev.dr(prev.mr),curr.dr(curr.mr));

            % Only save points matched in L-R and P-C
            match = intersect(tmatch_l', tmatch_r','rows')';
            match_p = match(1,:);
            match_c = match(2,:);
            
            % Xm in indices of X.dl
            [~,pmap] = sort(prev.ml);
            [~,cmap] = sort(curr.ml);
            match_p = pmap(match_p);
            match_c = cmap(match_c);
        else
            % Temporal similarities (left frame only)
            tmatch_l = vl_ubcmatch(prev.dl,desc_l);
            match_p = tmatch_l(1,:);
            match_c = tmatch_l(2,:);
        end

        % Matches must be matching in all four frames
        [mc,mci] = intersect(match_c,match_l);
        [mp,mpi] = intersect(match_p(mci),prev.ml);
        mc = mc(mpi);

        % Find pose with RANSAC
%        fprintf('RANSAC(n = %d): ',n)
        z = feats_l(1:2,mc); % Features (2D, current frame)
        m = prev.tri(:,mp);  % Landmarks (3D, previous frame)
        m = [m;ones(1,size(m,2))];
        m = cam.Twl \ m;
        [pose,ransac_inl] = RANSACMotion(z,m(1:3,:),ransac_p,ransac_sigma,cam);
%        fprintf('inliers: %d of %d\n', length(ransac_inl), length(mc))

        % Pose relative world frame
        curr.pose = T2Cart(Cart2T(prev.pose) * Cart2T(pose));
        
        % Store matches
        if discard_ransac_outliers
            % Filter out ransac non-matches?
            curr.mc = mc(ransac_inl);
            curr.mp = mp(ransac_inl);
        else
            curr.mc = match_c;
            curr.mp = match_p;
        end

        % Labelling
        if ~no_labelling
            labels = -ones(1,length(feats_l));
            labels(curr.mc) = prev.labels(curr.mp);
            newfeats = (labels == -1);
            nfcnt = sum(newfeats);
            labels(newfeats) = last_label + (1:nfcnt);
            curr.labels = labels;
            last_label = last_label + nfcnt;
        end
    end

    % Store frame info in cell array
    frame_info{n} = curr;

    % Plot temporal
    if show_temporal && n > 1
        % Plot
        if ~isempty(temp_lines)
            delete(temp_lines)
        end
        set(temp_im,'cdata',img_l)
        hold(temp_ax,'on')
        temp_lines = ...
            plot(temp_ax,feats_l(1,match_c), feats_l(2,match_c), 'bx');
        inl = ransac_inl;
        oul = setdiff(1:length(mc),ransac_inl);
        temp_lines = [temp_lines; plot(temp_ax, ...
             [feats_l(1,mc(inl)); prev.fl(1,mp(inl))], ...
             [feats_l(2,mc(inl)); prev.fl(2,mp(inl))], 'g')];
        temp_lines = [temp_lines; plot(temp_ax, ...
             [feats_l(1,mc(oul)); prev.fl(1,mp(oul))], ... 
             [feats_l(2,mc(oul)); prev.fl(2,mp(oul))], 'r')];
        title(temp_ax,sprintf('frame %d of %d\n', n, length(limages)))
        drawnow
    end

    % Plot point traj
    if show_point_traj && n > 1
        % Plot
        figure(3), clf, imshow(img_l), hold on
        plot(feats_l(1,match_c), feats_l(2,match_c), 'bx')
        lset = labels(curr.mc);
        i = n;
        while i > 1 %&& ~isempty(lset)
            c = frame_info{i};
            p = frame_info{i-1};
            [cpset,cui,pri] = intersect(c.labels,p.labels);
            [lset,ind] = intersect(cpset,lset);
            cui = cui(ind); pri = pri(ind);
            plot([c.fl(1,cui); p.fl(1,pri)], ...
                 [c.fl(2,cui); p.fl(2,pri)], 'g.-')
            i = i-1;
        end
        drawnow
    end

    
    % Plot trajectory
    if show_trajectory
        traj(:,n) = curr.pose(1:3);
        plot3(traj_ax,traj(3,1:n),-traj(1,1:n),-traj(2,1:n))
        axis(traj_ax,'equal')
        drawnow
    end
    
    prev = curr;
end
