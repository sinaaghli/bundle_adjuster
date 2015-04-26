#pragma once

#include <Eigen/Eigen>
#include <sophus/se3.hpp>
#include "PoseLandmarkContainer.h"
#include <SceneGraph/GLObject.h>

#define MAT4_COL_MAJOR_DATA(m) \
  (Eigen::Matrix<float, 4, 4, Eigen::ColMajor>(m).data())

/////////////////////////////////////////////////////////////////////////////
// Code to render the vehicle path

class GLPathAbs : public SceneGraph::GLObject {
 public:
  GLPathAbs() {
    m_bInitGLComplete = false;
    m_fLineColor(0) = 1.0;
    m_fLineColor(1) = 1.0;
    m_fLineColor(2) = 0.0;
    m_fLineColor(3) = 1.0;
    m_nPoseDisplay = 0;
    m_bDrawAxis = true;
    m_bDrawLines = true;
    m_dDrawLandmarkSize = 1;
    m_bDrawLandmarks = true;
  }

  ~GLPathAbs() {}

  // just draw the path
  void DrawCanonicalObject() {
    pangolin::GlState state;

    glLineWidth(1.0f);

    if (!m_vPosesAndLandmarks.empty()) {
      Eigen::Matrix4f fPose;

      if (m_bDrawAxis) {
        int start = 0;
        if (m_nPoseDisplay != 0) {
          if (m_vPosesAndLandmarks.size() > m_nPoseDisplay) {
            start = m_vPosesAndLandmarks.size() - m_nPoseDisplay;
          }
        }
        for (size_t ii = 0; ii < m_vPosesAndLandmarks.size(); ++ii) {
          glPushMatrix();
          Sophus::SE3d Pose;
          m_vPosesAndLandmarks[ii].GetPose(Pose);
          fPose = Pose.matrix().cast<float>();
          glMultMatrixf(MAT4_COL_MAJOR_DATA(fPose));
          if (static_cast<int>(ii) >= start) {
            glColor3f(1.0, 0.0, 0.0);
            pangolin::glDrawLine(0, 0, 0, 1, 0, 0);
            glColor3f(0.0, 1.0, 0.0);
            pangolin::glDrawLine(0, 0, 0, 0, 1, 0);
            glColor3f(0.0, 0.0, 1.0);
            pangolin::glDrawLine(0, 0, 0, 0, 0, 1);
          }
          glPopMatrix();
        }
      }

      if (m_bDrawLines) {
        glPushMatrix();
        glEnable(GL_LINE_SMOOTH);
        state.glLineWidth(3.0f);
        glColor4f(m_fLineColor(0), m_fLineColor(1), m_fLineColor(2),
                  m_fLineColor(3));

        glBegin(GL_LINE_STRIP);
        for (unsigned int ii = 0; ii < m_vPosesAndLandmarks.size(); ++ii) {
          Sophus::SE3d Pose;
          m_vPosesAndLandmarks[ii].GetPose(Pose);
          fPose = Pose.matrix().cast<float>();
          glVertex3f(fPose(0, 3), fPose(1, 3), fPose(2, 3));
        }
        glEnd();
        glPopMatrix();
      }

      if (m_bDrawLandmarks) {
        glPushMatrix();
        glPointSize(m_dDrawLandmarkSize);
        glBegin(GL_POINTS);
        glColor3f(1.0, 1.0, 0.2);
        for (unsigned int ii = 0; ii < m_vPosesAndLandmarks.size(); ++ii) {
          Sophus::SE3d Pose;
          std::vector<Sophus::Vector3d> landmarks;
          m_vPosesAndLandmarks[ii].GetPose(Pose);
          m_vPosesAndLandmarks[ii].GetLandmarks(landmarks);
          for (size_t jj = 0; jj < landmarks.size(); ++jj) {
              glVertex3f(landmarks[jj].data()[0],
                         landmarks[jj].data()[1],
                         landmarks[jj].data()[2]);
          }
        }
        glEnd();
        glPopMatrix();
      }
    }
  }

  std::vector<PoseLandmarkContainer>& GetPathRef() {
    return m_vPosesAndLandmarks;
  }

  void SetLineColor(float R, float G, float B, float A = 1.0) {
    m_fLineColor(0) = R;
    m_fLineColor(1) = G;
    m_fLineColor(2) = B;
    m_fLineColor(3) = A;
  }

  void SetPoseDisplay(unsigned int Num) { m_nPoseDisplay = Num; }

  void DrawLines(bool Val) { m_bDrawLines = Val; }

  void DrawAxis(bool Val) { m_bDrawAxis = Val; }

  void DrawLandmarks(bool Val) { m_bDrawLandmarks = Val; }

  void SetDrawLandmarkSize(double Size) { m_dDrawLandmarkSize = Size; }

 private:
  bool m_bDrawLines;
  bool m_bDrawLandmarks;
  bool m_bDrawAxis;
  unsigned int m_nPoseDisplay;
  double m_dDrawLandmarkSize;
  bool m_bInitGLComplete;
  Eigen::Vector4f m_fLineColor;
  std::vector<PoseLandmarkContainer> m_vPosesAndLandmarks;
};
