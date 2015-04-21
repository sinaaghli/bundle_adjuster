# bundle_adjuster
This library gives a batch solution of bundle adjustment problem given a set of camera poses, camera interinsics and set of matched features between images

##Dependencies(WITH_GUI=ON):
- [cerese_solver](https://github.com/ceres-solver/ceres-solver)
- Eigen
- [Sophus](https://github.com/arpg/Sophus)
- Opencv
- [Pangolin](https://github.com/arpg/Pangolin)
- [SceneGraph](https://github.com/arpg/SceneGraph)
- [Calibu](https://github.com/arpg/Calibu)
- [HAL](https://github.com/arpg/HAL)

##Dependencies(WITH_GUI=OFF):
- [cerese_solver](https://github.com/ceres-solver/ceres-solver)

##Commands to clone and install
- cd <into the directory you put your code>
- git clone git@github.com:sinaaghli/bundle_adjuster.git
- cd bundle_adjuster
- mkdir build
- cd buid
- cmake .. -DWITH_GUI=OFF
- make

##How to run the application (WITH_GUI=ON)
- in order to run the LadyBugDataset with ceres BAL do
  * ./bundle_adjuster -dataset <path to your project folder>bundle_adjuster/problem-49-7776-pre.txt
  * ex. ./bundle_adjuster -dataset /Users/sina/code/bundle_adjuster/problem-49-7776-pre.txt

## Other Instructions:
- please add your code between following comments:
* ///  ADD YOUR CODE FROM HERE !!!
* ///  TO HERE !!!

- if you are using the code with the option (WITH_GUI=OFF) you need to change the path of dataset in the code to its path in your system.
