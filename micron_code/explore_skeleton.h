// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_VESSEL_GRAPH_EXPLORE_SKELETON_H_
#define SRC_VESSEL_GRAPH_EXPLORE_SKELETON_H_

#include <list>
#include <opencv2/opencv.hpp>
#include "branch.h"

std::list<Branch> exploreSkeleton(cv::Mat skel, cv::Mat vesselsR);
std::list<Branch> vesselsTree(int r, int c, int n, int* nNew, cv::Mat sk,
	cv::Mat R);

#endif  // SRC_VESSEL_GRAPH_EXPLORE_SKELETON_H_
