// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include <cmath>
#include "estimate_distance_tip_structure.h"

void estimateTipDistanceAndNormal(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points,
	double* distance_tip_cloud,
	cv::Point3d* versor_point_to_tip) {
	double distance;
	cv::Point3d closest_point(0, 0, 0);
	double min_distance = pow(10, 9);  // number bigger than feasible distances
	int central_index = findClosestCenterIndex(tip_position, all_centers);
	// 36 points for each crf, search in range [-36n, +36n]
	// n at least = 3
	std::list<cv::Point3d>::iterator i;
	std::list<cv::Point3d>::iterator initial_iterator = all_points.begin();
	std::list<cv::Point3d>::iterator final_iterator = all_points.begin();
	std::advance(initial_iterator, std::max(central_index - 3, 0) * 36 / 2);
	std::advance(final_iterator, std::min((central_index + 3) * 36 / 2, static_cast<int>(all_points.size())));
	for (i = initial_iterator; i != final_iterator; ++i) {
		distance =
			sqrt(pow(tip_position.x - i->x, 2) + pow(tip_position.y - i->y, 2) +
				pow(tip_position.z - i->z, 2));
		if (distance < min_distance) {
			min_distance = distance;
			closest_point = *i;
		}
	}
	*distance_tip_cloud = min_distance;  // distance in um
	cv::Point3d vector_point_to_tip = tip_position - closest_point;
	*versor_point_to_tip = vector_point_to_tip / cv::norm(vector_point_to_tip);
}

void estimateTipDistanceNormalAndClosest(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points,	double* distance_tip_cloud,
	cv::Point3d* versor_point_to_tip, cv::Point3d* closest, cv::Point3d* center) {
	double distance;
	cv::Point3d closest_point(0, 0, 0);
	double min_distance = pow(10, 9);  // number bigger than feasible distances
	int central_index = findClosestCenterIndex(tip_position, all_centers);
	// 36 points for each crf, search in range [-36n, +36n]
	// n at least = 3
	std::list<cv::Point3d>::iterator i;
	std::list<cv::Point3d>::iterator initial_iterator = all_points.begin();
	std::list<cv::Point3d>::iterator final_iterator = all_points.begin();
	std::advance(initial_iterator, std::max(central_index - 3, 0) * 36 / 2);
	std::advance(final_iterator, std::min((central_index + 3) * 36 / 2, static_cast<int>(all_points.size())));
	
	for (i = initial_iterator; i != final_iterator; ++i) {
		
		distance =
			sqrt(pow(tip_position.x - i->x, 2) + pow(tip_position.y - i->y, 2) +
				pow(tip_position.z - i->z, 2));
		if (distance < min_distance) {
			min_distance = distance;
			closest_point = *i;
		}
	}
	*distance_tip_cloud = min_distance;  // distance in um
	cv::Point3d vector_point_to_tip = tip_position - closest_point;
	*versor_point_to_tip = vector_point_to_tip / cv::norm(vector_point_to_tip);
	*closest = closest_point;



 	cv::Point3d c1(all_centers.at<double>(central_index-3, 0), all_centers.at<double>(central_index-3, 1),
		all_centers.at<double>(central_index-3, 2));
	cv::Point3d c2(all_centers.at<double>(central_index, 0), all_centers.at<double>(central_index, 1),
		all_centers.at<double>(central_index, 2));
	cv::Point3d c3(all_centers.at<double>(central_index+3, 0), all_centers.at<double>(central_index+3, 1),
		all_centers.at<double>(central_index+3, 2));
	double distancec1 =
		sqrt(pow(closest_point.x - c1.x, 2) + pow(closest_point.y - c1.y, 2) +
			pow(closest_point.z - c1.z, 2));
	double distancec2 =
		sqrt(pow(closest_point.x - c2.x, 2) + pow(closest_point.y - c2.y, 2) +
			pow(closest_point.z - c2.z, 2));
	double distancec3 =
		sqrt(pow(closest_point.x - c3.x, 2) + pow(closest_point.y - c3.y, 2) +
			pow(closest_point.z - c3.z, 2));
	double mindist = std::min(std::min(distancec1, distancec2), distancec3);
	if (mindist == distancec1)
	{
		*center = c1;
	}
	else if (mindist == distancec2)
	{
		*center = c2;
	}
	else
	{
		*center = c3;
	}

	
	
}


void estimateVersorDistanceVessel(cv::Point3d tip_position,cv::Point3d deformable_point, double* distance_tip_cloud, cv::Point3d* versor_point_to_tip)
{
	*distance_tip_cloud = sqrt(pow(tip_position.x - deformable_point.x, 2) + pow(tip_position.y - deformable_point.y, 2) + pow(tip_position.z - deformable_point.z, 2));
	cv::Point3d vector_point_to_tip = deformable_point - tip_position;
	*versor_point_to_tip = vector_point_to_tip / cv::norm(vector_point_to_tip);
}

double estimateTipDistance(cv::Point3d tip_position, cv::Mat all_centers,
	std::list<cv::Point3d> all_points) {
	const int n_centers = all_points.size() / 36;
	double distance;
	double min_distance = pow(10, 9);
	int central_index = findClosestCenterIndex(tip_position, all_centers);
	// 36 points for each crf, search in range [-36n, +36n]
	// n at least = 3
	std::list<cv::Point3d>::iterator i;
	std::list<cv::Point3d>::iterator initial_iterator = all_points.begin();
	std::list<cv::Point3d>::iterator final_iterator = all_points.begin();
	std::advance(initial_iterator, std::max(central_index - 3, 0) * 36 / 2);
	std::advance(final_iterator, std::min((central_index + 3) * 36 / 2, static_cast<int>(all_points.size())));
	for (i = initial_iterator; i != final_iterator; std::next(i)) {
		distance =
			sqrt(pow(tip_position.x - i->x, 2) + pow(tip_position.y - i->y, 2) +
				pow(tip_position.z - i->z, 2));
		if (distance < min_distance) {
			min_distance = distance;
		}
	}
	return min_distance / 1000;
}

int findClosestCenterIndex(cv::Point3d tip_position, cv::Mat all_centers) {
	double distance;
	double min_distance = pow(10, 9);
	int index = 0;
	// nb only half of the centers are used to estimate the circumferences =>
	// increment i of multiples of 2
	for (int i = 0; i < all_centers.rows; i += 2) {
		distance = sqrt(pow(tip_position.x - all_centers.at<double>(i, 0), 2) +
			pow(tip_position.y - all_centers.at<double>(i, 1), 2) +
			pow(tip_position.z - all_centers.at<double>(i, 2), 2));
		if (distance < min_distance) {
			min_distance = distance;
			index = i;
		}
	}
	return index;
}
