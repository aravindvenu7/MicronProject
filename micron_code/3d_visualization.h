// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_3D_RECONSTRUCTION_3D_VISUALIZATION_H_
#define SRC_3D_RECONSTRUCTION_3D_VISUALIZATION_H_

#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
void showCloud(cv::Mat M);
void showCloud(std::list<cv::Point3d> list_of_points);
void showCloudAndTip(std::list<cv::Point3d>& list_of_points, const cv::Point3d& micron_tip);
void showCloudandcloud(std::list<cv::Point3d> list_of_points1, std::list<cv::Point3d> list_of_points2);
// void showCloudAllBranches(std::list<Branch> all_branches, cv::Mat
// warp_matrix);
// void showSurface(std::list<cv::Point3d> points);

#endif  // SRC_3D_RECONSTRUCTION_3D_VISUALIZATION_H_
