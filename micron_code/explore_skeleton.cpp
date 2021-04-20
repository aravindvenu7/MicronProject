// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include "explore_skeleton.h"

std::list<Branch> exploreSkeleton(cv::Mat skel, cv::Mat vesselsRad) {
	std::list<Branch> branchesAllTrees;
	int x = 0;
	int n = 1;  // branch number
	for (int r = 0; r < skel.rows; r++) {
		for (int c = 0; c < skel.cols; c++) {
			if (skel.at<uchar>(r, c) == 255) {
				x = x + 1;
				//cv::imshow("calls", skel);
				//cv::waitKey(0);
				branchesAllTrees.splice(branchesAllTrees.end(),
					vesselsTree(r, c, n, &n, skel, vesselsRad));
			}
		}
	}
	//std::cout << "explore_skeleton -> number of calls to vesselsTree : " << x << std::endl;
	 //std::cout << "explore_skeleton -> number of total branches: " << n - 1 << std::endl;
	return branchesAllTrees;
}

std::list<Branch> vesselsTree(int r, int c, int n, int* nNew, cv::Mat sk,
	cv::Mat R) {
	std::list<Branch> explored;
	std::list<cv::Point2i> unexplored;
	unexplored.push_back(cv::Point(c, r));
	
	while (!unexplored.empty()) {
		
		Branch newBranch(unexplored.front().y, unexplored.front().x, n, sk, R);
		explored.push_back(newBranch);
		unexplored.pop_front();
		unexplored.splice(unexplored.end(), newBranch.getNewCentersCoord());
		n++;
	}
	
	*nNew = n;
	return explored;
}
