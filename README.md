# bundle_adjuster
This library gives a batch solution of bundle adjustment problem given a set of camera poses, camera interinsics and set of matched features between images

##Dependencies:
- cerese_solver
- Cmake
- Eigen3
- Sophus
- Opencv
- Pangolin
- SceneGraph
- Calibu

##Commands to clone and install
- cd <into the directory you put your code>
- git clone git@github.com:sinaaghli/bundle_adjuster.git
- cd bundle_adjuster
- mkdir build
- cd buid
- cmake ..
- make

##How to run the application
- in order to run the LadyBugDataset with ceres BAL do
  -- ./bundle_adjuster -dataset <path to your project folder>bundle_adjuster/dataset1.txt
  -- ex. ./bundle_adjuster -dataset /Users/sina/code/bundle_adjuster/dataset1.txt
