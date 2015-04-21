/*
 * Command Argument Example is : -poses /Users/Sina/rpg/datasets/poses.txt
 */

#include <unistd.h>
#include <deque>
#include <fstream>
#include "BALProblem.h"

#ifdef WITH_GUI
  #include <pangolin/pangolin.h>
  #include <SceneGraph/SceneGraph.h>
  #include "Timer.h"
  #include "TimerView.h"
  #include "GLPathRel.h"
  #include "GLPathAbs.h"
  #include <Eigen/Eigen>
  #include <sophus/sophus.hpp>
  #include <opencv2/opencv.hpp>
  #include <calibu/Calibu.h>
  #include <HAL/Utils/GetPot>
  #include <HAL/Camera/CameraDevice.h>
  #include <HAL/IMU/IMUDevice.h>
#endif

int SolveBaProblem(std::string bal_file) {
  BALProblem bal_problem;
  if (!bal_problem.LoadFile(bal_file.c_str())) {
    std::cerr << "ERROR: unable to open file" << bal_file.c_str()
              << std::endl;
    return 0;
  }
  const double* observations = bal_problem.observations();

  ceres::Problem problem;
  for (int ii = 0; ii < bal_problem.num_observations(); ++ii) {
    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
        observations[2 * ii + 0], observations[2 * ii + 1]);
    problem.AddResidualBlock(
        cost_function, NULL,
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

  ///----- Set up GUI.
  pangolin::CreateGlutWindowAndBind("bundle_adjuster", 1600, 800);

  // Set up panel.
  const unsigned int panel_size = 180;
  pangolin::CreatePanel("ui")
      .SetBounds(0, 1, 0, pangolin::Attach::Pix(panel_size));
  pangolin::Var<bool> ui_camera_follow("ui.Camera Follow", false, true);
  pangolin::Var<bool> ui_reset("ui.Reset", true, false);
  pangolin::Var<bool> ui_show_vo_path("ui.Show VO Path", true, true);
  pangolin::Var<bool> ui_show_ba_path("ui.Show BA Path", false, true);
  pangolin::Var<bool> ui_show_ba_rel_path("ui.Show BA Rel Path", true, true);
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
  GLPathAbs gl_path_gt;
  gl_path_gt.SetPoseDisplay(5);
  gl_path_gt.SetLineColor(0, 0, 1.0);
  gl_graph.AddChild(&gl_path_gt);
  std::vector<Sophus::SE3d>& path_gt_vec = gl_path_gt.GetPathRef();

  // Add grid.
  SceneGraph::GLGrid gl_grid(150, 1);
  gl_graph.AddChild(&gl_grid);

  pangolin::View view_3d;

  pangolin::OpenGlRenderState stacks3d(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 1E-3, 10 * 1000),
      pangolin::ModelViewLookAt(-5, 0, -8, 0, 0, 0, pangolin::AxisNegZ));

  view_3d.SetHandler(new SceneGraph::HandlerSceneGraph(gl_graph, stacks3d))
      .SetDrawFunction(SceneGraph::ActivateDrawFunctor(gl_graph, stacks3d));

  container.AddDisplay(view_3d);

  // GUI aux variables.
  bool paused = true;
  bool step_once = false;

  ///----- Load file of ground truth poses (optional).
  bool have_gt;
  std::vector<Sophus::SE3d> poses;
  {
    std::string pose_file = cl_args.follow("", "-poses");
    if (pose_file.empty()) {
      std::cerr
          << "- NOTE: No poses file given. Not comparing against ground truth!"
          << std::endl;
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

        // Robotics convention (default).
        poses.push_back(calibu::ToCoordinateConvention(T, calibu::RdfRobotics));
      }
      std::cout << "- NOTE: " << poses.size() << " poses loaded." << std::endl;
      fclose(fd);
      have_gt = true;
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
  Trv.so3() = calibu::RdfRobotics;
  // Sophus::SE3d Tic = rig.t_wc_[0] * Trv;

  /////////////////////////////////////////////////////////////////////////////
  ///---- MAIN LOOP
  ///
  while (!pangolin::ShouldQuit()) {
    ///----- Init reset ...
    if (pangolin::Pushed(ui_reset)) {
      // Reset GUI path.
      path_gt_vec.clear();

      // Reset frame counter.
      frame_index = 0;

      // Reset map and current pose.
      vo_pose = Sophus::SE3d();
      ba_global_pose = Sophus::SE3d();
      ba_accum_rel_pose = Sophus::SE3d();
      if (have_gt) {
        for (int jj = 0; jj < (int)poses.size(); jj++)
          path_gt_vec.push_back(poses[jj]);
        std::cout << "have_gt is true" << std::endl;
      }
      frame_index++;
#endif
      //////////////////////////////////////////////////////////////////////////
      ///  ADD YOUR CODE HERE !!!
      ///

      if (use_cerese_ba) {
        std::cout << "local ba implementation is being called" << std::endl;
        SolveBaProblem("/Users/Sina/rpg/classproject/bundle_adjuster/test_ceres_250_b.txt");
      } else {
        std::cout << "local ba implementation is being called" << std::endl;
        //return 0;
      }
      //////////////////////////////////////////////////////////////////////////
#ifdef WITH_GUI
    }
    ////////////////////////////////////////////////////////////////////////////
    ///---- Render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (ui_camera_follow) {
      stacks3d.Follow(ba_accum_rel_pose.matrix());
    }

    gl_path_gt.SetVisible(true);

    // Sleep a bit if paused.
    if (paused) {
      usleep(1e6 / 60.0);
    }

    pangolin::FinishFrame();
  }

#endif

  return 0;
}
