/*
 * Command Argument Example is : -poses /Users/Sina/rpg/datasets/poses.txt
 */

#include <unistd.h>
#include <deque>
#include <fstream>
#include <vector>
#include "BALProblem.h"
#include "PoseLandmarkContainer.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>
#include "Timer.h"
#include "TimerView.h"
#include "GLPathRel.h"
#include "GLPathAbs.h"
#include <Eigen/Eigen>
#include <sophus/sophus.hpp>
#include <calibu/Calibu.h>
#include <HAL/Utils/GetPot>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>
#include <calibu/cam/camera_crtp.h>

//function p = triangulate(uvl, uvr, Twl, Twr, lcmod, rcmod)
//    d = uvl(1,:) - uvr(1,:); % disparities
//    b = norm(Twl(1:3,4) - Twr(1:3,4)); % baseline
//    f = lcmod.fx; % focal length
//    p(3,:) = b*f ./ d;
//    p(1,:) = (uvl(1,:) - lcmod.cx) .* p(3,:) ./ f;
//    p(2,:) = (uvl(2,:) - lcmod.cy) .* p(3,:) ./ f;
//    p = Twl * [p;ones(1,size(p,2))];
//    p = p(1:3,:);
//end

double g_baseline = 0;

struct DmatchInTimeAndStereo{
  int queryIdx;
  int trainIdx_stereo;
  int trainIdx_temp;
};

Eigen::Vector4d DepthFromDisparity(const cv::Point& uv, const double& disp,
                                   Eigen::Matrix3f& K, double minDisp = 0.0f ) {
  Eigen::Vector4d P;
  if (disp >= minDisp) {
    P[2] = K(0,0)/*fu*/ * g_baseline / disp;
  } else {
    std::cerr << "Error: disp<minDisp" << std::endl;
  }
  // (x,y,1) = kinv * (u,v,1)'
  P[0] = P[2] * (uv.x-K(0,2)) / K(0,0);
  P[1] = P[2] * (uv.y-K(1,2)) / K(1,1);
  P[3] = 1;
    return P;
}

void SolveLocalBA(Eigen::Matrix4d& rel_pose,
                  std::vector<cv::KeyPoint>& keypoints_l_now,
                  std::vector<cv::KeyPoint>& keypoints_r_now,
                  std::vector<cv::KeyPoint>& keypoints_l_prev,
                  std::vector<DmatchInTimeAndStereo> matches,
                  Eigen::Matrix3f& K_left){
  std::vector<cv::Point> residual;
  for (int ii = 0; ii < matches.size(); ++ii) {
    double disp = keypoints_l_now[matches[ii].queryIdx].pt.x - keypoints_r_now[matches[ii].trainIdx_stereo].pt.x;
    Eigen::Vector4d point_in_space = DepthFromDisparity(keypoints_l_now[matches[ii].queryIdx].pt, disp,K_left);
    point_in_space = rel_pose * point_in_space;
    Eigen::Vector3d pp = calibu::Project(point_in_space);
    cv::Point pt;
    pt.x = pp[0];
    pt.y = pp[1];
    residual.push_back(keypoints_l_prev[matches[ii].trainIdx_temp].pt);// - pt);
  }

}

void DrawStereoLandmarkCorrespondance(cv::Mat& left_img,
                                      const cv::Point& left_lm,
                                      cv::Mat& right_img,
                                      const cv::Point& right_lm) {
  const int gap_btw_images = 5;
  cv::Point tmp_p;
  tmp_p.x = right_lm.x + 640 + gap_btw_images;
  tmp_p.y = right_lm.y;
  cv::line(left_img, left_lm, tmp_p, cv::Scalar(0, 255, 0), 1);
  tmp_p.x = left_lm.x - 640;
  tmp_p.y = left_lm.y;
  cv::line(right_img, tmp_p, right_lm, cv::Scalar(0, 255, 0), 1);
}

void DrawTemporalLandmarkCorrespondance(cv::Mat& curr_img,
                                        const cv::Point& curr_lm,
                                        const cv::Point& prev_lm) {
  cv::circle(curr_img, curr_lm, 5, cv::Scalar(0, 0, 255), 1);
  cv::line(curr_img, prev_lm, curr_lm, cv::Scalar(0, 0, 255), 1);
}

bool FindAndDrawMatches(cv::Mat& image_1, cv::Mat& image_2,
                        const bool is_stereo,
                        Eigen::Matrix3f& K_left,
                        std::vector<Sophus::Vector4d>& landmarks) {
  //-- Step 1: Detect the keypoints using SIFT Detector
  int minHessian = 400;
  cv::SiftFeatureDetector detector(minHessian);
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
  detector.detect(image_1, keypoints_1);
  detector.detect(image_2, keypoints_2);

  //-- Step 2: Calculate descriptors (feature vectors)
  cv::SiftDescriptorExtractor extractor;
  cv::Mat descriptors_1, descriptors_2;
  extractor.compute(image_1, keypoints_1, descriptors_1);
  extractor.compute(image_2, keypoints_2, descriptors_2);

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  // TODO(sina) what is flann matcher
  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;
  matcher.match(descriptors_1, descriptors_2, matches);
  double max_dist = 0;
  double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  std::cout << "Max distance is: " << max_dist << std::endl;
  std::cout << "Min distance is: " << min_dist << std::endl;

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= cv::max(5 * min_dist, 0.02)) {
      good_matches.push_back(matches[i]);
    }
  }
  std::cout << "Number of good matches in frame: " << good_matches.size()
            << std::endl;

  //-- Draw only "good" matches
  if (is_stereo) {
    for (auto it : good_matches) {
      DrawStereoLandmarkCorrespondance(image_1, keypoints_1[it.queryIdx].pt,
                                       image_2, keypoints_2[it.trainIdx].pt);
      double disp = keypoints_1[it.queryIdx].pt.x - keypoints_2[it.trainIdx].pt.x;
      landmarks.push_back(DepthFromDisparity(keypoints_1[it.queryIdx].pt, disp,
                          K_left));
    }
  } else {
    for (auto it : good_matches) {
      DrawTemporalLandmarkCorrespondance(image_1, keypoints_1[it.queryIdx].pt,
                                         keypoints_2[it.trainIdx].pt);
      double disp = keypoints_1[it.queryIdx].pt.x - keypoints_2[it.trainIdx].pt.x;
      landmarks.push_back(DepthFromDisparity(keypoints_1[it.queryIdx].pt, disp,
                          K_left));
    }
  }
  return 0;
}

bool FindAndDrawMatches2(cv::Mat& image_1, cv::Mat& image_2, cv::Mat& prev_img,
                        const bool is_stereo,
                        Eigen::Matrix3f& K_left,
                        std::vector<Sophus::Vector4d>& landmarks,
                        Sophus::SE3d& cam_rel_transform) {
  //-- Step 1: Detect the keypoints using SIFT Detector
  int minHessian = 1000;
  cv::SiftFeatureDetector detector(minHessian);
  std::vector<cv::KeyPoint> keypoints_1, keypoints_2, keypoints_3;
  detector.detect(image_1, keypoints_1);
  detector.detect(image_2, keypoints_2);
  detector.detect(prev_img, keypoints_3);

  //-- Step 2: Calculate descriptors (feature vectors)
  cv::SiftDescriptorExtractor extractor;
  cv::Mat descriptors_1, descriptors_2, descriptors_3;
  extractor.compute(image_1, keypoints_1, descriptors_1);
  extractor.compute(image_2, keypoints_2, descriptors_2);
  extractor.compute(prev_img, keypoints_3, descriptors_3);

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  // TODO(sina) what is flann matcher
  cv::FlannBasedMatcher matcher, matcher_temporal;
  std::vector<cv::DMatch> matches,matches_temporal;
  matcher.match(descriptors_1, descriptors_2, matches);
  matcher_temporal.match(descriptors_1,descriptors_3,matches_temporal);

  double max_dist = 0;
  double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = matches[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }
  std::cout << "Max distance is: " << max_dist << std::endl;
  std::cout << "Min distance is: " << min_dist << std::endl;

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches1, good_matches2;

  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= cv::max(4 * min_dist, 0.02)) {
      good_matches1.push_back(matches[i]);
    }
  }
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches_temporal[i].distance <= cv::max(4 * min_dist, 0.02)) {
      good_matches2.push_back(matches_temporal[i]);
    }
  }
  std::vector<DmatchInTimeAndStereo> good_matches_final;
  for (size_t ii = 0; ii < good_matches1.size(); ii++) {
    for (size_t jj=0; jj<good_matches2.size(); jj++) {
      if (good_matches1[ii].queryIdx == good_matches2[jj].queryIdx) {
        DmatchInTimeAndStereo tmp;
        tmp.queryIdx = good_matches1[ii].queryIdx;
        tmp.trainIdx_stereo = good_matches1[ii].trainIdx;
        tmp.trainIdx_temp = good_matches2[jj].trainIdx;
        good_matches_final.push_back(tmp);
      }
    }
  }

  //-- Draw only "good" matches
  if (is_stereo) {
    for (auto& it : good_matches_final) {
      DrawStereoLandmarkCorrespondance(image_1, keypoints_1[it.queryIdx].pt,
                                       image_2, keypoints_2[it.trainIdx_stereo].pt);
      DrawTemporalLandmarkCorrespondance(image_1, keypoints_1[it.queryIdx].pt,
                                         keypoints_3[it.trainIdx_temp].pt);
      double disp = keypoints_1[it.queryIdx].pt.x - keypoints_2[it.trainIdx_stereo].pt.x;
      landmarks.push_back(DepthFromDisparity(keypoints_1[it.queryIdx].pt, disp,
                          K_left));
    }
  }


  // Find Camera Transform
  //SolveLocalBA(Eigen::Matrix4d(cam_rel_transform.matrix()),keypoints_1,keypoints_2,keypoints_3,good_matches_final,K_left);

  return 0;
}


int SolveBaProblem(std::string bal_file) {
  BALProblem bal_problem;
  if (!bal_problem.LoadFile(bal_file.c_str())) {
    std::cerr << "ERROR: unable to open file" << bal_file.c_str() << std::endl;
    return 0;
  }
  const double* observations = bal_problem.observations();

  ceres::Problem problem;
  for (int ii = 0; ii < bal_problem.num_observations(); ++ii) {
    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
        observations[2 * ii + 0], observations[2 * ii + 1]);
    problem.AddResidualBlock(cost_function, NULL,
                             bal_problem.mutable_camera_for_observation(ii),
                             bal_problem.mutable_point_for_observation(ii));
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 100;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  bool use_cerese_ba = true;

#ifdef WITH_GUI
  std::deque<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> > filter;

  GetPot cl_args(argc, argv);
  std::cout << "Starting bundle_adjuster ..." << std::endl;
  int frame_skip = cl_args.follow(0, "-skip");
  if (!cl_args.search("-cam")) {
    std::cerr << "Camera arguments missing!" << std::endl;
    exit(EXIT_FAILURE);
  }
  hal::Camera camera(cl_args.follow("", "-cam"));
  const int image_width = camera.Width();
  const int image_height = camera.Height();
  std::cout << "- Image Dimensions: " << image_width << "x" << image_height
            << std::endl;
  if (camera.NumChannels() != 2) {
    std::cerr << "Two images (stereo pair) are required in order to"
                 " use this program!" << std::endl;
    exit(EXIT_FAILURE);
  }

  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("bundle_adjuster", 1250, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui")
      .SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool> ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool> ui_reset("ui.Reset", true, false);
  pangolin::Var<bool> ui_show_gt_path("ui.Show GT Path", true, true);

  // Set up container.
  pangolin::View& container = pangolin::CreateDisplay();
  container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 1);
  container.SetLayout(pangolin::LayoutEqual);
  pangolin::DisplayBase().AddDisplay(container);

  // Set up 3D view for container.
  SceneGraph::GLSceneGraph gl_graph;
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  // Reset background color to black.
  glClearColor(0, 0, 0, 1);

  // Add path.
  GLPathRel gl_cam_path;
  gl_cam_path.SetPoseDisplay(0);
  gl_cam_path.SetLineColor(0, 0, 1.0);
  gl_graph.AddChild(&gl_cam_path);
  std::vector<PoseLandmarkContainer>& cam_path_vec = gl_cam_path.GetPathRef();
  gl_cam_path.SetDrawLandmarkSize(3);

  // Add grid.
  SceneGraph::GLGrid gl_grid(100, 1);
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;

  pangolin::OpenGlRenderState stacks3d(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 1E-3, 10 * 1000),
      pangolin::ModelViewLookAt(-5, 0, -8, 0, 0, 0, pangolin::AxisNegZ));

  view_3d.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, stacks3d));

  // Add all subviews to container.
  SceneGraph::ImageView left_image_view;
  left_image_view.SetAspect(640.0 / 480.0);
  container.AddDisplay(left_image_view);

  SceneGraph::ImageView right_image_view;
  right_image_view.SetAspect(640.0 / 480.0);
  container.AddDisplay(right_image_view);

  SceneGraph::ImageView prev_left_image_view;
  prev_left_image_view.SetAspect(640.0 / 480.0);
  container.AddDisplay(prev_left_image_view);

  container.AddDisplay(view_3d);

  // GUI aux variables.
  bool capture_flag = false;
  bool paused = true;
  bool step_once = false;

  ///----- Load camera models.
  calibu::CameraRig rig;
  rig = calibu::ReadXmlRig(cl_args.follow("","-cmod"));
  Eigen::Matrix3f K_0 = rig.cameras[0].camera.K().cast<float>();
//  std::cout << "-- K_0 is: \n" << K_0 << std::endl;
  Sophus::SE3d T_wc_0 = rig.cameras[0].T_wc;
//  std::cout << "-- T_wc_0:\n" << T_wc_0.matrix3x4() << std::endl;
  Eigen::Matrix3f K_1 = rig.cameras[1].camera.K().cast<float>();
//  std::cout << "-- K_1 is: " << std::endl << K_1 << std::endl;
  Sophus::SE3d T_wc_1 = rig.cameras[1].T_wc;
//  std::cout << "-- T_wc_11:\n" << T_wc_1.matrix3x4() << std::endl;
  Sophus::SE3d tmp_se3d;
  tmp_se3d = T_wc_0.inverse() * T_wc_1;
  Eigen::Vector3d diff_T_wc(tmp_se3d.translation());
  g_baseline = diff_T_wc.norm();
  std::cout << "- NOTE: Baseline is:" << g_baseline << std::endl;

  ///----- Aux variables.
  cv::Mat current_left_image, current_right_image;
  current_left_image.flags = CV_LOAD_IMAGE_COLOR;
  current_right_image.flags = CV_LOAD_IMAGE_COLOR;

  ///----- Load file of ground truth poses (optional).
  bool cam_path_enabled;
  std::vector<Sophus::SE3d> cam_poses;
  {
    std::string pose_file = cl_args.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr
          << "- NOTE: No poses file given. Not comparing against ground truth!"
          << std::endl;
      cam_path_enabled = false;
      ui_show_gt_path = false;
    } else {
      FILE* fd = fopen(pose_file.c_str(), "r");
      Eigen::Matrix<double, 6, 1> pose;
      float x, y, z, p, q, r;

      std::cout << "- Loading pose file: '" << pose_file << "'" << std::endl;
      if (cl_args.search("-V")) {
        // Vision convention.
        std::cout << "- NOTE: File is being read in VISION frame." << std::endl;
      } else if (cl_args.search("-C")) {
        // Custom convention.
        std::cout << "- NOTE: File is being read in *****CUSTOM***** frame."
                  << std::endl;
      } else if (cl_args.search("-T")) {
        // Tsukuba convention.
        std::cout << "- NOTE: File is being read in TSUKUBA frame."
                  << std::endl;
      } else {
        // Robotics convention (default).
        std::cout << "- NOTE: File is being read in ROBOTICS frame."
                  << std::endl;
      }

      while (fscanf(fd, "%f\t%f\t%f\t%f\t%f\t%f", &x, &y, &z, &p, &q, &r) !=
             EOF) {
        pose(0) = x;
        pose(1) = y;
        pose(2) = z;
        pose(3) = p;
        pose(4) = q;
        pose(5) = r;

        Sophus::SE3d T(SceneGraph::GLCart2T(pose));

        // Vision convention (default).
        cam_poses.push_back(
            calibu::ToCoordinateConvention(T, calibu::RdfVision));
      }
      std::cout << "- NOTE: " << cam_poses.size() << " poses loaded."
                << std::endl;
      fclose(fd);
      cam_path_enabled = true;
    }
  }

  ///----- Register callbacks.
  // Hide/Show panel.
  pangolin::RegisterKeyPressCallback('~', [&]() {
    static bool fullscreen = true;
    fullscreen = !fullscreen;
    if (fullscreen) {
      container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.75);
    } else {
      container.SetBounds(0, 1, 0, 1);
    }
    pangolin::Display("ui").Show(fullscreen);
  });

  // Container view handler.
  const char keyShowHide[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'};
  const char keySave[] = {'!', '@', '#', '$', '%', '^', '&', '*', '(', ')'};
  for (size_t ii = 0; ii < container.NumChildren(); ii++) {
    pangolin::RegisterKeyPressCallback(
        keyShowHide[ii], [&container, ii]() { container[ii].ToggleShow(); });
    pangolin::RegisterKeyPressCallback(keySave[ii], [&container, ii]() {
      container[ii].SaveRenderNow("screenshot", 4);
    });
  }

  pangolin::RegisterKeyPressCallback(' ', [&paused] { paused = !paused; });
  pangolin::RegisterKeyPressCallback(
      pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
      [&step_once] { step_once = !step_once; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r',
                                     [&ui_reset] { ui_reset = true; });

  ///----- Init general variables.
  unsigned int frame_index;
  Sophus::SE3d vo_pose;
  Sophus::SE3d ba_accum_rel_pose;
  Sophus::SE3d ba_global_pose;

  // IMU-Camera transform through robotic to vision conversion.
  Sophus::SE3d Trv;
  Trv.so3() = calibu::RdfVision;
  // Sophus::SE3d Tic = rig.t_wc_[0] * Trv;

  ///----- Init general variables.

  // Image holder.
  std::shared_ptr<pb::ImageArray> images = pb::ImageArray::Create();
  cv::Mat prev_left_image;



  /////////////////////////////////////////////////////////////////////////////
  ///---- MAIN LOOP
  ///
  while (!pangolin::ShouldQuit()) {
    ///----- Init reset ...
    if (pangolin::Pushed(ui_reset)) {
      // Reset GUI path.
      cam_path_vec.clear();

      // Reset frame counter.
      frame_index = 0;

      // Reset map and current pose.
      vo_pose = Sophus::SE3d();
      ba_global_pose = Sophus::SE3d();
      ba_accum_rel_pose = Sophus::SE3d();

      if (cam_path_enabled) {
        // Draw test path
//        PoseLandmarkContainer poselandmark;
//        std::vector<Sophus::Vector4d> temp_landmarks;
//        for (int ii=-2;ii<3;++ii) {
//          for (int jj = -2; jj < 3; ++jj) {
//            Sophus::Vector4d mark(0.2*ii,0.2*jj,0,1);
//            temp_landmarks.push_back(mark);
//          }
//        }

//        for (int jj = 0; jj < (int)cam_poses.size(); jj++) {
//          poselandmark.SetPose(cam_poses[jj]);
//          poselandmark.SetLandmark(temp_landmarks);
//          cam_path_vec.push_back(poselandmark);
//        }


//        Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity(4,4);
//        matrix(0,3) = 1;
//        Sophus::SE3d tmp(matrix);
//        Sophus::Vector4d vec(0,0,0,1);
//        Sophus::Vector4d result;
//        result = tmp.matrix() * vec;

//        std::cout << "matrix is:\n" << result.data() << std::endl;



        std::cout << "cam_path_enabled is true" << std::endl;
      }

      // Capture first image.
      const int first_frames_to_skip = 700;
      for (size_t ii = 0; ii < first_frames_to_skip; ++ii) {
        capture_flag = camera.Capture(*images);
        //        usleep(100);
      }
      capture_flag = camera.Capture(*images);

      // Set images.
      current_left_image = images->at(0)->Mat().clone();
      current_right_image = images->at(1)->Mat().clone();

      cv::cvtColor(current_left_image, current_left_image, CV_GRAY2RGB);
      cv::cvtColor(current_right_image, current_right_image, CV_GRAY2RGB);
      prev_left_image = current_left_image.clone();

      // Init cam path
      PoseLandmarkContainer init_pose_container;
      Eigen::Matrix4d temp_pose;
      // Set Init Pose to origin
      temp_pose.setIdentity();
      Sophus::SE3d init_pose(temp_pose);
      std::vector<Sophus::Vector4d> init_landmarks;
      FindAndDrawMatches(current_left_image, current_right_image, true,
                         K_0, init_landmarks);
      init_pose_container.SetPose(init_pose);
      init_pose_container.SetLandmark(init_landmarks);
      cam_path_vec.push_back(init_pose_container);

      left_image_view.SetImage(current_left_image.data, current_left_image.cols,
                               current_left_image.rows, GL_RGB, GL_RGB,
                               GL_UNSIGNED_BYTE);
      right_image_view.SetImage(
          current_right_image.data, current_right_image.cols,
          current_right_image.rows, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
      prev_left_image_view.SetImage(prev_left_image.data, prev_left_image.cols,
                                    prev_left_image.rows, GL_RGB, GL_RGB,
                                    GL_UNSIGNED_BYTE);

      frame_index++;
      if (!cl_args.search("-poses") && false) {
#endif
        //////////////////////////////////////////////////////////////////////////
        ///  ADD YOUR CODE FROM HERE !!!
        //////////////////////////////////////////////////////////////////////////

        if (use_cerese_ba) {
          std::cout << "ceres ba implementation is being called" << std::endl;
          SolveBaProblem("../test_ceres_250_b.txt");
        } else {
          std::cout << "local ba implementation is being called" << std::endl;
          // return 0;
        }
//////////////////////////////////////////////////////////////////////////
///  TO HERE !!!
//////////////////////////////////////////////////////////////////////////
#ifdef WITH_GUI
      }
    }

    ///----- Step forward ...
    if (!paused || pangolin::Pushed(step_once)) {
      //  Capture the new image.
      for (int ii = 0; ii < frame_skip; ++ii) {
        capture_flag = camera.Capture(*images);
        //        usleep(100);
      }
      capture_flag = camera.Capture(*images);

      if (capture_flag == false) {
        paused = true;
      } else {
        // Set images.
        current_left_image = images->at(0)->Mat().clone();
        current_right_image = images->at(1)->Mat().clone();
        cv::cvtColor(current_left_image, current_left_image, CV_GRAY2RGB);
        cv::cvtColor(current_right_image, current_right_image, CV_GRAY2RGB);

        // Init cam path
        PoseLandmarkContainer init_pose_container;
        Eigen::Matrix4d temp_pose;
        // Set Init Pose to origin
        temp_pose.setIdentity();
        Sophus::SE3d init_pose(temp_pose);
        std::vector<Sophus::Vector4d> init_landmarks;
//        FindAndDrawMatches(current_left_image, current_right_image, true,
//                           K_0, init_landmarks);
        FindAndDrawMatches2(current_left_image, current_right_image,
                            prev_left_image, true, K_0, init_landmarks,init_pose);
//        LocalBA();

        init_pose_container.SetLandmark(init_landmarks);
        init_pose_container.SetPose(init_pose);
        std::cout << "num of landmarks is:" << init_landmarks.size() << std::endl;
        cam_path_vec[0] = init_pose_container;
        //cam_path_vec.push_back(init_pose_container);

        // Clone left image
        prev_left_image = images->at(0)->Mat().clone();
        cv::cvtColor(prev_left_image, prev_left_image, CV_GRAY2RGB);

        left_image_view.SetImage(
            current_left_image.data, current_left_image.cols,
            current_left_image.rows, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
        right_image_view.SetImage(
            current_right_image.data, current_right_image.cols,
            current_right_image.rows, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE);
        prev_left_image_view.SetImage(prev_left_image.data, prev_left_image.cols,
                                      prev_left_image.rows, GL_RGB, GL_RGB,
                                      GL_UNSIGNED_BYTE);

        //        prev_image = current_left_image.clone();

        // Increment frame counter.
        frame_index++;
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    ///---- Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (ui_camera_follow) {
      stacks3d.Follow(ba_accum_rel_pose.matrix());
    }

    gl_cam_path.SetVisible(true);

    // Sleep a bit if paused.
    if (paused) {
      usleep(1e6 / 60.0);
    }
    pangolin::FinishFrame();
  }

#endif

  return 0;
}
