#include <unistd.h>
#include <deque>
#include <fstream>

#include <Eigen/Eigen>
#include <sophus/sophus.hpp>

#include <opencv2/opencv.hpp>

#include <calibu/Calibu.h>
#include <HAL/Utils/GetPot>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/IMU/IMUDevice.h>

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>

#include "AnalyticsView.h"
#include "Timer.h"
#include "TimerView.h"
#include "GLPathRel.h"
#include "GLPathAbs.h"

std::deque<std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> >   filter;

int main(int argc, char** argv)
{
  std::cout << "Starting bundle_adjuster ..." << std::endl;

  GetPot cl_args(argc, argv);
  //int frame_skip  = cl_args.follow(0, "-skip");

  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("bundle_adjuster", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool>           ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool>           ui_reset("ui.Reset", true, false);
  pangolin::Var<bool>           ui_show_vo_path("ui.Show VO Path", true, true);
  pangolin::Var<bool>           ui_show_ba_path("ui.Show BA Path", false, true);
  pangolin::Var<bool>           ui_show_ba_rel_path("ui.Show BA Rel Path", true, true);
  pangolin::Var<bool>           ui_show_ba_win_path("ui.Show BA Win Path", false, true);
  pangolin::Var<bool>           ui_show_gt_path("ui.Show GT Path", true, true);

  // Set up container.
  pangolin::View& container = pangolin::CreateDisplay();
  container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.75);
  container.SetLayout(pangolin::LayoutEqual);
  pangolin::DisplayBase().AddDisplay(container);

  // Set up timer.
  Timer     timer;
  TimerView timer_view;
  timer_view.SetBounds(0.5, 1, 0.75, 1.0);
  pangolin::DisplayBase().AddDisplay(timer_view);
  timer_view.InitReset();

  // Set up 3D view for container.
  SceneGraph::GLSceneGraph gl_graph;
  SceneGraph::GLSceneGraph::ApplyPreferredGlSettings();

  // Reset background color to black.
  glClearColor(0, 0, 0, 1);

  // Add path.
  GLPathAbs gl_path_vo;
  GLPathAbs gl_path_ba;
  GLPathAbs gl_path_ba_rel;
  gl_path_vo.SetPoseDisplay(5);
  gl_path_ba.SetPoseDisplay(5);
  gl_path_ba_rel.SetPoseDisplay(5);
  gl_path_vo.SetLineColor(1.0, 0, 1.0);
  gl_path_ba.SetLineColor(0, 1.0, 0);
  gl_path_ba_rel.SetLineColor(0, 1.0, 1.0);
  gl_graph.AddChild(&gl_path_vo);
  gl_graph.AddChild(&gl_path_ba);
  gl_graph.AddChild(&gl_path_ba_rel);
  std::vector<Sophus::SE3d>& path_vo_vec = gl_path_vo.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_vec = gl_path_ba.GetPathRef();
  std::vector<Sophus::SE3d>& path_ba_rel_vec = gl_path_ba_rel.GetPathRef();

  // Add grid.
  SceneGraph::GLGrid gl_grid(150, 1);
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;

  pangolin::OpenGlRenderState stacks3d(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 1E-3, 10*1000),
        pangolin::ModelViewLookAt(-5, 0, -8, 0, 0, 0, pangolin::AxisNegZ)
        );

  view_3d.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, stacks3d));

  // Add all subviews to container.
  SceneGraph::ImageView image_view;
  image_view.SetAspect(640.0 / 480.0);
  container.AddDisplay(image_view);

  SceneGraph::ImageView depth_view;
  container.AddDisplay(depth_view);

  container.AddDisplay(view_3d);

  // GUI aux variables.
  bool paused       = true;
  bool step_once    = false;

  ///----- Load file of ground truth poses (optional).
  bool have_gt;
  std::vector<Sophus::SE3d> poses;
  {
    std::string pose_file = cl_args.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr << "- NOTE: No poses file given. Not comparing against ground truth!" << std::endl;
      have_gt = false;
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
        std::cout << "- NOTE: File is being read in *****CUSTOM***** frame." << std::endl;
      } else if (cl_args.search("-T")) {
        // Tsukuba convention.
        std::cout << "- NOTE: File is being read in TSUKUBA frame." << std::endl;
      } else {
        // Robotics convention (default).
        std::cout << "- NOTE: File is being read in ROBOTICS frame." << std::endl;
      }

      while (fscanf(fd, "%f\t%f\t%f\t%f\t%f\t%f", &x, &y, &z, &p, &q, &r) != EOF) {
        pose(0) = x;
        pose(1) = y;
        pose(2) = z;
        pose(3) = p;
        pose(4) = q;
        pose(5) = r;

        Sophus::SE3d T(SceneGraph::GLCart2T(pose));

        // Flag to load poses as a particular convention.
        if (cl_args.search("-V")) {
          // Vision convention.
          poses.push_back(T);
        } else if (cl_args.search("-C")) {
          // Custom setting.
          pose(0) *= -1;
          pose(2) *= -1;
          Sophus::SE3d Tt(SceneGraph::GLCart2T(pose));
          poses.push_back(calibu::ToCoordinateConvention(Tt,
                                                         calibu::RdfRobotics));
        } else if (cl_args.search("-T")) {
          // Tsukuba convention.
          Eigen::Matrix3d tsukuba_convention;
          tsukuba_convention << -1,  0,  0,
                                 0, -1,  0,
                                 0,  0, -1;
          Sophus::SO3d tsukuba_convention_sophus(tsukuba_convention);
          poses.push_back(calibu::ToCoordinateConvention(T,
                                          tsukuba_convention_sophus.inverse()));
        } else {
          // Robotics convention (default).
          poses.push_back(calibu::ToCoordinateConvention(T,
                                          calibu::RdfRobotics));
        }
      }
      std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
      fclose(fd);
      have_gt = true;
    }
  }

  ///----- Register callbacks.
  // Hide/Show panel.
  pangolin::RegisterKeyPressCallback('~', [&](){
    static bool fullscreen = true;
    fullscreen = !fullscreen;
    if (fullscreen) {
      container.SetBounds(0, 1, pangolin::Attach::Pix(panel_size), 0.75);
    } else {
      container.SetBounds(0, 1, 0, 1);
    }
    timer_view.Show(fullscreen);
    pangolin::Display("ui").Show(fullscreen);
  });

  // Container view handler.
  const char keyShowHide[] = {'1','2','3','4','5','6','7','8','9','0'};
  const char keySave[]     = {'!','@','#','$','%','^','&','*','(',')'};
  for (size_t ii = 0; ii < container.NumChildren(); ii++) {
    pangolin::RegisterKeyPressCallback(keyShowHide[ii], [&container,ii]() {
      container[ii].ToggleShow(); });
    pangolin::RegisterKeyPressCallback(keySave[ii], [&container,ii]() {
      container[ii].SaveRenderNow("screenshot", 4); });
  }

  pangolin::RegisterKeyPressCallback(' ', [&paused] { paused = !paused; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
                                     [&step_once] {
                                        step_once = !step_once; });
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r',
                                     [&ui_reset] {
                                        ui_reset = true; });

  ///----- Init general variables.
  unsigned int                      frame_index;
  Sophus::SE3d                      vo_pose;
  Sophus::SE3d                      ba_accum_rel_pose;
  Sophus::SE3d                      ba_global_pose;

  // IMU-Camera transform through robotic to vision conversion.
  Sophus::SE3d Trv;
  Trv.so3() = calibu::RdfRobotics;
  //Sophus::SE3d Tic = rig.t_wc_[0] * Trv;

  /////////////////////////////////////////////////////////////////////////////
  ///---- MAIN LOOP
  ///
  while (!pangolin::ShouldQuit()) {

    // Start timer.
    timer.Tic();

    ///----- Init reset ...
    if (pangolin::Pushed(ui_reset)) {
      // Reset timer and analytics.
      timer_view.InitReset();

      // Reset GUI path.
      path_vo_vec.clear();
      path_ba_vec.clear();
      path_ba_rel_vec.clear();

      // Reset frame counter.
      frame_index = 0;

      // Reset map and current pose.
      vo_pose = Sophus::SE3d();
      ba_global_pose = Sophus::SE3d();
      ba_accum_rel_pose = Sophus::SE3d();
      path_vo_vec.push_back(vo_pose);
      path_ba_vec.push_back(ba_global_pose);
      path_ba_rel_vec.push_back(ba_accum_rel_pose);

    }


    /////////////////////////////////////////////////////////////////////////////
    ///---- Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (ui_camera_follow) {
      stacks3d.Follow(ba_accum_rel_pose.matrix());
    }

    gl_path_vo.SetVisible(ui_show_vo_path);
    gl_path_ba.SetVisible(ui_show_ba_path);
    gl_path_ba_rel.SetVisible(ui_show_ba_rel_path);

    // Sleep a bit if paused.
    if (paused) {
      usleep(1e6/60.0);
    }

    // Stop timer and update.
    timer.Toc();
    timer_view.Update(10, timer.GetNames(3), timer.GetTimes(3));

    pangolin::FinishFrame();
  }

  return 0;
}
