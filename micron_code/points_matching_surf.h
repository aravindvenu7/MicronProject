// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_POINTS_MATCHING_POINTS_MATCHING_SURF_H_
#define SRC_POINTS_MATCHING_POINTS_MATCHING_SURF_H_

#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

cv::Mat warpImageL2R(cv::Mat img_1, cv::Mat img_2, cv::Mat img_1seg);
cv::Mat warpMatrix(cv::Mat im_src, cv::Mat im_dst);
void matchedPoints(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2i>* L,
	std::vector<cv::Point2i>* R);
void drawKeyPoints(cv::Mat i1, cv::Mat i2, std::vector<cv::Point2i> k1,
	std::vector<cv::Point2i> k2);
void showOverlappingImgs(cv::Mat img, cv::Mat img_over, int window);
void testingPair(cv::Mat i1, cv::Mat i2);

#endif  // SRC_POINTS_MATCHING_POINTS_MATCHING_SURF_H_