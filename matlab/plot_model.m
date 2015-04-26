function plot_model(file1, file2,limit)

L = 50; % length
step = 10; % Step between poses

% Before and after
pts  = cell(2,1);
cams = cell(2,1);
[cams{1},pts{1}]=load_ceres_params(file1);
[cams{2},pts{2}]=load_ceres_params(file2);
ttl = {'triangulated','bundle adjusted'};

% Limit
if nargin < 3, limit = inf; end
if length(pts{1}) > limit,  pts{1} = pts{1}(:,1:limit); end
if length(pts{2}) > limit,  pts{2} = pts{2}(:,1:limit); end

% Plot
figure
for i = 1:length(cams)
    subplot(1,length(cams),i)
    scatter3(pts{i}(3,:),pts{i}(1,:),pts{i}(2,:),1,-pts{i}(2,:))
    hold on
    plot3(cams{i}(6,:),cams{i}(4,:),cams{i}(5,:),'.-r')
    set(gca,'yDir','reverse','zDir','reverse')
    axis equal
    for j = 1:step:length(cams{i})
        c = 'rgb';
        Tj = Cart2T(cams{i}([4:6,1:3],j));
        p0 = Tj * [0,0,0,1]';
        for n = 1:3
            p = [0;0;0;1];
            p(n) = 1;
            p = Tj * p;
            p = p0 + (p-p0)*L;
            plot3([p0(3),p(3)],[p0(1),p(1)],[p0(2),p(2)],c(n))
        end
    end
    view([-90,90])
%    view([-90,0])
    S = 1000;
    axis(S*[-.1,.9,-.7,.3,-.5,.5])
%    axis(S*[-.5,.5,-.5,.5,-.5,.5])
    caxis([0,300])
    xlabel('z'),ylabel('x'),zlabel('y')
    title(ttl{i})
end

end
