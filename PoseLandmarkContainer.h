#pragma once

#include <sophus/se3.hpp>
#include <vector>


class PoseLandmarkContainer {
 public:
  PoseLandmarkContainer() {}
  ~PoseLandmarkContainer() {}

  void SetPose(Sophus::SE3d& pose) {
    pose_ = pose;
  }

  void SetLandmark(Sophus::Vector4d& landmark) {
    landmarks_.push_back(landmark);
  }

  void SetLandmark(std::vector<Sophus::Vector4d>& landmark) {
    landmarks_.insert(landmarks_.end(), landmark.begin(), landmark.end());
  }

  void GetPose(Sophus::SE3d& pose) {
    pose = pose_;
  }

  void GetLandmarks(std::vector<Sophus::Vector4d>& landmark) {
    landmark = landmarks_;
  }

  unsigned int NumOfLandmarks() {
    return landmarks_.size();
  }

 private:
  Sophus::SE3d pose_;
  std::vector<Sophus::Vector4d> landmarks_;

};
