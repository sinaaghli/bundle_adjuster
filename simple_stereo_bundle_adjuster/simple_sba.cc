// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class SBALProblem {
 public:
  ~SBALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 6 * num_cameras_; }

  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 6;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  bool LoadFile(const char* filename, const char* filename2) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

	// Read observations
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[4 * num_observations_];

	printf("Reading observations...\n");
    for (int i = 0; i < num_observations_; ++i) {
//      printf("%d\n",i);
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 4; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 4*i + j);
      }
    }

	fclose(fptr);
    fptr = fopen(filename2, "r");
    if (fptr == NULL) {
      return false;
    };

	// Read initial parameters
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    num_parameters_ = 6 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

	printf("Reading cameras...\n");
    for (int i = 0; i < num_cameras_; ++i) {
      for (int j = 0; j < 6; ++j)
      	FscanfOrDie(fptr, "%lf", parameters_ + i*6 + j);
      double dummy;
      for (int j = 0; j < 3; ++j)
      	FscanfOrDie(fptr, "%lf", &dummy);
    }
	printf("Reading points...\n");
    for (int i = 0; i < num_points_ * 3; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + 6*num_cameras_ + i);
    }

	fclose(fptr);
    return true;
  }

  void print_parameters(FILE * f) {
    double * p = parameters_;
    fprintf(f,"%d %d\n", num_cameras_,num_points_);
    for (int i = 0; i < num_cameras_; i++) {
      for (int j = 0; j < 6; j++)
        fprintf(f,"%f ", *(p++));
      fprintf(f,"268.511900 0 0 \n");
    }
    for (int i = 0; i < num_points_; i++) {
      for (int j = 0; j < 3; j++)
        fprintf(f,"%f ", *(p++));
      fprintf(f,"\n");
    }
  }

 private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};



struct StereoReprojectionError {
  StereoReprojectionError(double obs_xl, double obs_yl, double obs_xr, double obs_yr)
      : obs_xl(obs_xl), obs_yl(obs_yl), obs_xr(obs_xr), obs_yr(obs_yr){}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
	// Constants	
    const T& focal = T(268.511900);
    const T& dx = T(0.7);

	// Transfrom 
	T c1 = cos(camera[0]);
	T c2 = cos(camera[1]);
	T c3 = cos(camera[2]);
	T s1 = sin(camera[0]);
	T s2 = sin(camera[1]);
	T s3 = sin(camera[2]);
	T x1 = point[0], x2 = point[1], x3 = point[2];
	T t1 = camera[3], t2 = camera[4], t3 = camera[5];

    // Twl = Cart2T([t;r]);
    // xlp = Twl \ [x;1];
	// Point in left camera's coordinates
	T pr1 = c1*c3*x1 - c1*c3*t1 - c2*s1*t2 + c1*s3*t3 + c2*s1*x2 - c1*s3*x3 - c3*s1*s2*t3 + c3*s1*s2*x3 - s1*s2*s3*t1 + s1*s2*s3*x1;
	T pr2 = c1*c2*x2 - c1*c2*t2 + c3*s1*t1 - c3*s1*x1 - s1*s3*t3 + s1*s3*x3 - c1*c3*s2*t3 + c1*c3*s2*x3 - c1*s2*s3*t1 + c1*s2*s3*x1;
	T pr3 =                                                               s2*t2 - s2*x2 - c2*c3*t3 + c2*c3*x3 - c2*s3*t1 + c2*s3*x1;
    T pred_xl = focal * pr1 / pr3;
    T pred_yl = focal * pr2 / pr3;

	// Right camera
	pr1 -= dx;
    T pred_xr = focal * pr1 / pr3;
    T pred_yr = focal * pr2 / pr3;

    residuals[0] = pred_xl - T(obs_xl);
    residuals[1] = pred_yl - T(obs_yl);
    residuals[2] = pred_xr - T(obs_xr);
    residuals[3] = pred_yr - T(obs_yr);

    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double obs_xl, double obs_yl, double obs_xr, double obs_yr) {
    return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 4, 6, 3>(
                new StereoReprojectionError(obs_xl, obs_yl, obs_xr, obs_yr)));
  }
  static ceres::CostFunction* Create(const double * obs) {
    return (new ceres::AutoDiffCostFunction<StereoReprojectionError, 4, 6, 3>(
                new StereoReprojectionError(obs[0], obs[1], obs[2], obs[3])));
  }

  double obs_xl;
  double obs_xr;
  double obs_yl;
  double obs_yr;
};



int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    std::cerr << "usage: simple_sba <bal_observations> <bal_initial>\n";
    return 1;
  }

  SBALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1], argv[2])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << " or file " << argv[2] << "\n";
    return 1;
  }

  const double* observations = bal_problem.observations();

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    ceres::CostFunction* cost_function =
        StereoReprojectionError::Create(&observations[4 * i]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
/*
  	double res[2];
	MyReprojectionError f(observations[2 * i + 0],observations[2 * i + 1]);	
    f(bal_problem.mutable_camera_for_observation(i),bal_problem.mutable_point_for_observation(i),res);
	printf("residual[%d]: % 12f,% 12f\n",i,res[0], res[1]);
*/
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  //options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 200;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.num_threads = 4;
  options.num_linear_solver_threads = 4;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  FILE * f = fopen("sba-out.txt","w");
  bal_problem.print_parameters(f);
  fclose(f);
  return 0;
}
