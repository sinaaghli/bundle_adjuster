/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */
#include <string>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <time.h>
using namespace cv;

/**
 * @function main
 * @brief Main function
 */
int sift_match(const std::string& match_image1,const std::string& match_image2)
{


  //string image1 = *match_image1;
  //string image2 = *match_image2;     // C++ STL string
  //printf("%s\n", image1.c_str());
  //image1 = "../left_1.png";
  //image2= "../right_1.png";


  Mat img_1 = imread( match_image1, CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( match_image2, CV_LOAD_IMAGE_GRAYSCALE );



  //Handle errors if not able to read images
  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SIFT Detector
  int minHessian = 400;

  SiftFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SiftDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );
  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(5*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  imshow( "Good Matches", img_matches );
  //for( int i = 0; i < (int)good_matches.size(); i++ )
  //{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

  waitKey(0);

  return 0;
}

int main()
{
    string left_images[10];
    left_images[0] = "../left_1.png";

    left_images[1] = "../left_2.png";
    left_images[2] = "../left_3.png";
    left_images[3] = "../left_4.png";
    left_images[4] = "../left_5.png";
    left_images[5] = "../left_6.png";
    left_images[6] = "../left_7.png";
    left_images[7] = "../left_8.png";
    left_images[8] = "../left_9.png";
    left_images[9] = "../left_10.png";


    string right_images[10];
    right_images[0] = "../right_1.png";

    right_images[1] = "../right_2.png";
    right_images[2] = "../right_3.png";
    right_images[3] = "../right_4.png";
    right_images[4] = "../right_5.png";
    right_images[5] = "../right_6.png";
    right_images[6] = "../right_7.png";
    right_images[7] = "../right_8.png";
    right_images[8] = "../right_9.png";
    right_images[9] = "../right_10.png";

    for(int i=0; i<10; i++)
    {
        printf("%s\n", left_images[i].c_str());
        sift_match(left_images[i].c_str(), right_images[i].c_str());
    }
    return 0;
}
