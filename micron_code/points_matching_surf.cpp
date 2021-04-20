// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include <algorithm>
#include <iostream>
#include "points_matching_surf.h"

cv::Mat warpImageL2R(cv::Mat im_src, cv::Mat im_dst, cv::Mat im_to_warp) {
	cv::Mat im_warped;
	// Warp source image to destination based on homography
	cv::warpPerspective(im_to_warp, im_warped, warpMatrix(im_src, im_dst),
		im_warped.size(), cv::INTER_NEAREST);
	// showOverlappingImgs(im_dst, im_warped, 1);
	return im_warped;
}

cv::Mat warpMatrix(cv::Mat im_src, cv::Mat im_dst) {
	std::vector<cv::Point2i> a;
	std::vector<cv::Point2i> b;

	matchedPoints(im_src, im_dst, &a, &b);
	return findHomography(a, b, cv::RANSAC);
}

void matchedPoints(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2i>* L,
	std::vector<cv::Point2i>* R) {
	const int minHessian = 50;  // default 400
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	auto detector = cv::xfeatures2d::SURF::create(minHessian);
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);
	
	// Matching
	cv::Mat descriptors1, descriptors2;
	detector->compute(img1, keypoints1, descriptors1);
	detector->compute(img2, keypoints2, descriptors2);
	// Matching descriptor vectors using FLANN matcher
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	/*
	* Calculation of max and min distances between keypoints = score of
	* similarity between their descriptors (for real valued -> Euclidean dist)
	* Good matches (i.e. whose distance is less than 2*min_dist, or a small
	* arbitary value ( 0.02 ) in the event that min_dist is very small)
	*/
	std::vector<cv::DMatch> good_matches;
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++) {
		double dist = matches[i].distance;
		if (dist < min_dist) {
			min_dist = dist;
		}
		if (dist > max_dist) {
			max_dist = dist;
		}
	}
	for (int i = 0; i < descriptors1.rows; i++) {
		if (matches[i].distance <= std::max(5 * min_dist, 0.02)) {  // default 2
			good_matches.push_back(matches[i]);
		}
	}

	// Find transformation between two images
	for (int i = 0; i < static_cast<int>(good_matches.size()); i++) {
		// Get the keypoints from the good matches
		L->push_back(keypoints1[good_matches[i].queryIdx].pt);
		R->push_back(keypoints2[good_matches[i].trainIdx].pt);
	}
}

void drawKeyPoints(cv::Mat i1, cv::Mat i2, std::vector<cv::KeyPoint> k1,
	std::vector<cv::KeyPoint> k2) {
	cv::Mat img_k1, img_k2;
	cv::drawKeypoints(i1, k1, img_k1, cv::Scalar::all(-1),
		cv::DrawMatchesFlags::DEFAULT);
	cv::drawKeypoints(i2, k2, img_k2, cv::Scalar::all(-1),
		cv::DrawMatchesFlags::DEFAULT);
	cv::imshow("Keypoints 1", img_k1);
	cv::imshow("Keypoints 2", img_k2);
	cv::waitKey(0);
}

void showOverlappingImgs(cv::Mat img, cv::Mat img_over, int window) {
	cv::Mat dst;
	addWeighted(img, 0.5, img_over, 0.5, 0.0, dst);
	std::stringstream msg;
	msg << "overlapping " << window;
	cv::imshow(msg.str(), dst);
	cv::waitKey(0);
}

void testingPair(cv::Mat i1, cv::Mat i2) {
	bool found1 = false;
	bool found2 = false;
	assert(i1.rows == i2.rows && i1.cols == i2.cols);
	for (int i = 0; i < i1.rows; i++) {
		for (int j = 0; j < i1.cols; j++) {
			if (i1.at<uchar>(i, j) == 1 && !found1) {
				std::cout << "Prima img: riga (y) = " << i << "; colonna (x) = " << j
					<< std::endl;
				found1 = true;
			}
			if (i2.at<uchar>(i, j) == 3 && !found2) {
				std::cout << "Seconda img: riga (y) = " << i << "; colonna (x) = " << j
					<< std::endl;
				found2 = true;
			}
		}
	}
}
