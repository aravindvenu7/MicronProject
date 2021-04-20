// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>
/*
* Branch.cpp
*
*  Created on: Jul 7, 2017
*      Author: simocmu
*/

#include "branch.h"
// #include <vector>
// #include <list>
//int counter;
namespace {
	const uchar kWHITE = 255;
}  // namespace

Branch::Branch(int row_start, int col_start, int n, cv::Mat skeleton,
	cv::Mat d_tr_img) {
	skeleton_colored = skeleton;
	vessel_number = n;
	distances_transformated_img = d_tr_img;
	if (skeleton_colored.at<uchar>(row_start, col_start) == kWHITE) {
		exploreBranch(row_start, col_start);
		findAllRadiiDirection();
		//std::cout << "number of recursions" << counter << std::endl;
	}
}

void Branch::exploreBranch(int row, int col) {
	//counter = counter + 1;
	skeleton_colored.at<uchar>(row, col) =
		vessel_number;  
	centers_coord.push_back(cv::Point(col, row));
	radii.push_back(distances_transformated_img.at<float>(row, col));
	std::list<cv::Point2i> unexplored_pixel;
	for (int r = row - 1; r <= row + 1; r++) {
		for (int c = col - 1; c <= col + 1; c++) {
			if (r >= 0 && c >= 0 && r < skeleton_colored.rows && c < skeleton_colored.cols 
				&& skeleton_colored.at<uchar>(r, c) == kWHITE) {
				unexplored_pixel.push_back(cv::Point(c, r));
				int elem = elSameBranch(r, c, unexplored_pixel);
				if (elem >= 0) {
					if (r - row == 0 || c - col == 0) {  // main connections
						std::list<cv::Point2i>::const_iterator it =
							unexplored_pixel.begin();
						std::advance(it, elem);
						it = unexplored_pixel.erase(it);
					}
					else {
						unexplored_pixel.pop_back();
					}
				}
			}
		}
	}
	// decisions about next step
	if (static_cast<int>(unexplored_pixel.size()) == 1) {  
		exploreBranch(unexplored_pixel.front().y, unexplored_pixel.front().x);
	}
	if (static_cast<int>(unexplored_pixel.size()) > 1) {  
		if (centers_coord.size() == 1) {
			new_centers_coord = unexplored_pixel;
			new_centers_coord.pop_front();
			exploreBranch(unexplored_pixel.front().y, unexplored_pixel.front().x);
		} else {
			new_centers_coord = unexplored_pixel;
		}
	}
	else if (static_cast<int>(unexplored_pixel.size()) == 0 && centers_coord.size() <= 1) {  
		skeleton_colored.at<uchar>(row, col) = 0;  
		centers_coord.pop_back();  
		radii.pop_back();
	}
}

bool Branch::isSameBranch(int r1, int c1, std::list<cv::Point2i> pixels) {
	bool answer = false;
	for (const auto& element : pixels) {
		if (std::abs(r1 + c1 - element.x - element.y) == 1) {
			answer = true;
		}
	}
	return answer;
}

int Branch::elSameBranch(int r1, int c1, std::list<cv::Point2i> pixels) {
	// return index of pixel to delate (part of same branch -> diagonal adjacent to orthogonal)
	int answ = -1; // no adjacent pixels yet
	std::list<cv::Point2i>::iterator i;
	for (i = pixels.begin(); i != pixels.end(); ++i) {
		int check = std::abs(r1 + c1 - i->x - i->y);
		if (check == 1 && (r1 == i->y || c1 == i->x)) {
			answ = std::distance(pixels.begin(), i);
		}
	}
	return answ;
}

cv::Point2d Branch::normalToCurve(cv::Point2d pre_center,
	cv::Point2d post_center) {
	cv::Point2d versor;
	versor.x = -(post_center.y - pre_center.y);
	versor.y = post_center.x - pre_center.x;
	return versor / cv::norm(versor);
}

void Branch::findAllRadiiDirection() {
	for (auto iterator = centers_coord.begin(); iterator != centers_coord.end();
		++iterator) {
		if (iterator == centers_coord.begin()) {
			radii_direction.push_back(normalToCurve(*iterator, *std::next(iterator)));
		}
		else if (iterator == std::prev(centers_coord.end())) {
			radii_direction.push_back(normalToCurve(*std::prev(iterator), *iterator));
		}
		else {
			radii_direction.push_back(
				normalToCurve(*std::prev(iterator), *std::next(iterator)));
		}
	}
}

std::vector<cv::Point2i> Branch::getCentersCoord() { return centers_coord; }
std::list<cv::Point2i> Branch::getNewCentersCoord() {
	return new_centers_coord;
}
std::vector<double> Branch::getRadii() { return radii; }
std::vector<cv::Point2d> Branch::getRadiiDirection() { return radii_direction; }
cv::Point2i Branch::getStartingPoint() { return centers_coord.front(); }
cv::Point2i Branch::getEndingingPoint() { return centers_coord.back(); }
cv::Mat Branch::getSkeletonColored() { return skeleton_colored; }
int Branch::getBranchNumber() { return vessel_number; }
