// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#ifndef SRC_TIP_POSITION_AND_CONTROL_VISION_H_
#define SRC_TIP_POSITION_AND_CONTROL_VISION_H_

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

void cloudComputation(std::list<cv::Point3d>& all_points_old, cv::Mat& all_centers_old, bool& written_cloud,double& minimumradius,cv::Point3d tip);
void initialization(cv::Mat* warp_matrix);
void initialization1(cv::Mat* warp_matrix, cv::Mat image_left, cv::Mat image_right);
void reconstruct(cv::Mat& warp_matrix1, cv::Mat& warp_matrix2, std::list<cv::Point3d>* cloud_of_points,
	cv::Mat* all_centers, double* minimumradius, cv::Point3d tippos);// , cv::Mat image_left);
void structure3dReconstruction(cv::Mat& imgseglthinned, cv::Mat dtr, cv::Mat warp_m1, cv::Mat warp_m2,
	std::list<cv::Point3d>* all_points,
	cv::Mat* all_centers,double* minrad);
void openingAndClosingInterface();
/*void openingAndClosingInterface_thin();
void changethin(int, void*);
cv::Mat thin_operations(cv::Mat& pre);*/
void changeOpeningAndClosing(int, void*);
cv::Mat morphologicalOperations(cv::Mat& pre);
void printTipTriangulation();
void testReconstruction();
void testReconstructionSURF();

#endif  // SRC_TIP_POSITION_AND_CONTROL_VISION_H_
