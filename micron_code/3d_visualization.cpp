// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include "3d_visualization.h"

#include <vector>

#include <opencv2/viz.hpp>

cv::viz::Viz3d viz_elem("show_cloud");

void showCloud(cv::Mat M) {
	cv::viz::Viz3d viz_elem("show_cloud");
	viz_elem.setBackgroundColor(cv::viz::Color::black());
	//viz_elem.showWidget("coosys", cv::viz::WCoordinateSystem());
	viz_elem.showWidget("cloud", cv::viz::WCloud(M, cv::viz::Color::white()));
	viz_elem.spin();
}

void showCloud(std::list<cv::Point3d> list_of_points) {
	std::vector<cv::Point3d> vec_pts(list_of_points.begin(),
		list_of_points.end());
	cv::viz::Viz3d viz_elem("show_cloud");
	
	viz_elem.setBackgroundColor(cv::viz::Color::black());
	//viz_elem.showWidget("coosys", cv::viz::WCoordinateSystem());
	
	viz_elem.showWidget(
		"cloud", cv::viz::WCloud(cv::Mat(vec_pts), cv::viz::Color::white()));
	//viz_elem.showImage(cv::Mat(vec_pts));
	viz_elem.spin();
	std::cout << "here inside show cloud" << std::endl;
}

void showCloudandcloud(std::list<cv::Point3d> list_of_points1, std::list<cv::Point3d> list_of_points2) {

	


	list_of_points1.splice(list_of_points1.end(), list_of_points2);
	std::vector<cv::Point3d> vec_pts(list_of_points1.begin(),
		list_of_points1.end());
	cv::viz::Viz3d viz_elem("show_cloud");

	viz_elem.setBackgroundColor(cv::viz::Color::black());
	//viz_elem.showWidget("coosys", cv::viz::WCoordinateSystem());

	viz_elem.showWidget(
		"cloud", cv::viz::WCloud(cv::Mat(vec_pts), cv::viz::Color::white()));
	//viz_elem.showImage(cv::Mat(vec_pts));
	viz_elem.spinOnce(1, true);
	
}



void showCloudAndTip(std::list<cv::Point3d>& list_of_points, const cv::Point3d& micron_tip) {
	std::vector<cv::Point3d> vec_pts(list_of_points.begin(),
		                             list_of_points.end());
	viz_elem.setBackgroundColor(cv::viz::Color::black());
	cv::viz::WCloud* cloud = new cv::viz::WCloud(cv::Mat(vec_pts), cv::viz::Color::white());
	cv::viz::WSphere* tip = new cv::viz::WSphere(micron_tip, 1000, 10, cv::viz::Color::red());
	viz_elem.showWidget("cloud", *cloud);
	viz_elem.showWidget("point", *tip);
	viz_elem.spinOnce(1, true);
}

/*
void showCloudAllBranches(std::list<Branch> all_branches, cv::Mat warp_matrix) {
std::list<cv::Point3d> all_points;
for (const auto& it : all_branches) {
Branch3d currentBranch(it, warp_matrix);
all_points.splice(all_points.end(),
findAllCrfsOfBranchMean3(currentBranch));
}
showCloud(all_points);
}
*/
