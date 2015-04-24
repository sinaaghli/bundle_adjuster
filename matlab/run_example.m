% Include VLFeat
run('/home2/niklas/cv/vlfeat-0.9.20/toolbox/vl_setup')

% Read cam variables
load('cityblock_camera')
assert(exist('cam','var') == 1)

% Set path to cityblock
CBPATH = '/home2/niklas/cv/cityblock/';

% Load file names
if ~exist('limages','var') && ~exist('rimages','var')
    limages = dir([CBPATH,'images/left*']);
    rimages = dir([CBPATH,'images/right*']);
    limages = {limages.name};
    rimages = {rimages.name};
end

% Filter images
limages = limages(1:50);
rimages = rimages(1:50);

% Run triangulation
stereo_temporal_triangulation;

% Save files
save_ceres_files(frame_info,cam,'ceres_test_obs.txt','ceres_test_params.txt',4,1);

% Run ceres here...
% simple_sba ceres_test_obs.txt ceres_test_params.txt
pause

% Plot
plot_model('ceres_test_params.txt','sba-out.txt')
