// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include <stdio.h>
#include <cassert>
#include <ctime>
#include <iostream>
#include "seg.h"
#include "vision.h"
#include "camera.h"


#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/surface_matching/icp.hpp"

//#include "opencv2/cudastereo.hpp"
//#include "unet_segmentation.h"
#include "points_matching_surf.h"
#include "thinning_zs.h"
#include "explore_skeleton.h"
#include "branch.h"
#include "branch3d.h"
#include "compute_circumferences_points.h"
#include "3d_visualization.h"
//cv::
// #include "surf_comparison.h"

// Variables for sliders 
int picturecounter = 0;
cv::Mat original;
cv::Mat morphological_image;
cv::Mat thin_image;
const int switch_slider_max = 1;
const int opening_slider_max = 50;
const int closing_slider_max = 50;
int switch_slider;
int opening_slider;
int closing_slider;
int switch_value;
int opening_size;
int closing_size;

bool apply_morphological_operations = false;
cv::Mat element_o;
cv::Mat element_c;
cv::Mat leftintrinsic =
(cv::Mat_<double>(3, 3) << 751.706129, 0.000000, 338.665467, 0.000000,
	766.315580, 257.986032, 0.000000, 0.000000, 1.000000);


cv::Mat leftdistortion =
(cv::Mat_<double>(1, 4) << -0.328424, 0.856059, 0.003430, 0.000248);

cv::Mat rightcam =
(cv::Mat_<double>(4, 3) <<
	-3.6727985e-02, -2.2311393e-02, -9.1255752e-03, -4.5127481e+02,
	-2.4047634e-02, 3.6737934e-02, 4.9335274e-03, 1.6571433e+02,
	-9.3507415e-07, -1.6451460e-06, 7.8380448e-06, 2.3031225e-02





	);

std::ofstream evalcloud("evalcloud7.txt");
std::ofstream recontimeold("recontimeold.txt");
cv::Mat leftcam =
(cv::Mat_<double>(4, 3) <<
	-3.6088562e-02, -2.1858681e-02, -1.4944750e-02, -9.0687099e+02,
	-2.4750440e-02, 3.7163560e-02, 5.2922145e-03, 1.8992193e+02,
	-1.5707503e-06, -1.8483953e-06, 7.3806462e-06, -7.3772440e-03);

cv::Mat matrix =
(cv::Mat_<double>(4, 3) <<
	5.16279567e-01, 1.08354334e-01, 4.95403607e+02,
	2.21513862e-02, 9.77019003e-01, -3.32705413e+01,
	-3.35909344e-05, 5.39491180e-07, 1.02123235e+00);

void cloudComputation(std::list<cv::Point3d>& all_points_old, cv::Mat& all_centers_old, bool& written_cloud,double& minimumradius, cv::Point3d tip) 
{

	cv::Mat all_centers_current;
	std::list<cv::Point3d> all_points_current;
	cv::Mat warp_matrix;
	printTipTriangulation();
	initialization(&warp_matrix);
	std::cout << "initialization complete" << std::endl;
	double minimumradiustemp;
	int gate = 0;
	while (true) {
		int t1 = clock();
		
		reconstruct(warp_matrix,warp_matrix, &all_points_current, &all_centers_current,&minimumradiustemp,tip);
		int t2 = clock();
		int i = 0;
		
		//std::cout << "cloud number" << " " << counter << std::endl;
		all_centers_old = all_centers_current;
		all_points_old = all_points_current;
		std::cout << all_points_old.size();
		written_cloud = true;
		minimumradius = minimumradiustemp;


		std::cout << "total time" << (t2 - t1) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
	}
}

void initialization(cv::Mat* warp_matrix) {

	initializeNeuralNetwork();
	std::cout << "neural network initialized" << std::endl;

	openCameras();
	cv::Mat image_left, image_right, image_new;
	int start = clock();
	takeBothPictures(&image_left, &image_right);
	int end = clock();
	//std::cout << end - start << std::endl;
	//image_left = cv::imread("D:/stereo_dataset/left6.jpg");
	//image_right = cv::imread("D:/stereo_dataset/right6.jpg");
	if (!image_left.empty() && !image_right.empty()) {
		*warp_matrix = warpMatrix(image_left, image_right);
		segment(image_left);
		openingAndClosingInterface();
		if (opening_slider == 0, closing_slider == 0) {
			apply_morphological_operations = false;
		}
	}
}


void reconstruct(cv::Mat& warp_matrix1, cv::Mat& warp_matrix2, std::list<cv::Point3d>* cloud_of_points,
	             cv::Mat* all_centers,double* minimumradius,cv::Point3d tippos){

	int tq = clock();
	cv::Mat image_left = takeLeftPicture(); 
	int tw = clock();
	if (!image_left.empty()) {
		int tc = clock();
		cv::Mat image_left_seg = cv::imread("D:/sfoti/left7s.jpg", cv::IMREAD_GRAYSCALE); //
		image_left_seg = morphologicalOperations(image_left_seg);
		int td = clock(); 
		//cv::imshow("left image segmented", image_left_seg);		
		cv::Rect roi;   
		roi.x = 0; //60
		roi.y = 0;  //300
		roi.width = 800; //100
		roi.height = 600; //100		
		int kernel_size = 9; //3
		int scale = 1;
		int delta = 0;
		int ddepth = CV_32FC1;
		//std::cout << "segmentation time" << (td - tc) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
		cv::Mat temp, temp1,tempx;
		int tb = clock();
		cv::threshold(image_left_seg, image_left_seg, 1, 255, cv::THRESH_BINARY); //CV_THRESH_BINARY
		cv::distanceTransform(image_left_seg, temp1, cv::DIST_L2, 3);// CV_DIST_L2

		/*double minVal;
		double maxVal;
		cv::Point minLoc;
		cv::Point maxLoc;
		cv::minMaxLoc(temp1, &minVal, &maxVal, &minLoc, &maxLoc);
		double av = (minVal + maxVal) / 2;*/
		//cv::threshold(temp1, tempx,11, 255, cv::THRESH_BINARY);   //maxVal - av//CV_THRESH_BINARY		
		//tempx.convertTo(tempx, CV_8UC1);
		//cv::Mat tempy;
		cv::Laplacian(temp1, temp, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
		//temp.convertTo(tempy, CV_8UC1);
		cv::threshold(temp, temp, -6200, 1.0, cv::THRESH_BINARY_INV); //2.75 //3.7 // 35  // 425 //6200 //CV_THRESH_BINARY_INV		
		cv::threshold(temp, temp, 0.5, 255, cv::THRESH_BINARY); //CV_THRESH_BINARY
		temp.convertTo(temp, CV_8UC1);
		int tn = clock();

		////////////////////////////////////////////////////////////////////////////
		/*cv::distanceTransform(temp, temp, CV_DIST_L2, 3);
		cv::Laplacian(temp, temp, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
		cv::threshold(temp, temp, -21, 1.0, CV_THRESH_BINARY_INV);
		cv::threshold(temp, temp, 0.5, 255, CV_THRESH_BINARY);
		temp.convertTo(temp, CV_8UC1);*/

		////////////////////////////////////////
		cv::Mat imgseglthinned = temp(roi);
		//cv::Mat imgseglx = tempx(roi);
		//cv::Mat imgsegl = image_left_seg(roi);
		//cv::threshold(imgseglthinned, imgseglthinned, 0.5, 255, CV_THRESH_BINARY);
		//image_left_seg = morphologicalOperations(image_left_seg);		
		cv::Mat dtr = temp1(roi);		
		std::cout << "processing time median blur" << (tn - tb) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		

		std::list<cv::Point3d> points;
		cv::Mat centers;                                                                               ///**Aravind - Tip position is required here for finer search in 60x60 region.**
		double minrad;
		int te = clock();
		structure3dReconstruction(imgseglthinned,dtr,warp_matrix1, warp_matrix2, &points, &centers,&minrad); //imgseglthinned //imgseglx //image_left_seg
		int tf = clock();		
		*cloud_of_points = points;
		*all_centers = centers;
		*minimumradius = minrad;
		showCloud(*cloud_of_points);
		// std::cout << "number of points: " << cloud_of_points.size() << std::endl;
		else
		{
			std::cout << "empty image" << std::endl;
		}
	}
}

void structure3dReconstruction(cv::Mat& imgseglthinned, cv::Mat dtr,  cv::Mat warp_m1, cv::Mat warp_m2,
	                           std::list<cv::Point3d>* all_points,
	                           cv::Mat* all_centers,double* minrad) {
	cv::Mat vessels_centers_L, vessels_centers_L_mod, vessels_radii_L;
	int to = clock();
	vessels_centers_L = imgseglthinned;
	thinning(imgseglthinned, vessels_centers_L);
	int tp = clock();
	cv::Rect myROI(1, 1, vessels_centers_L.cols - 2, vessels_centers_L.rows - 2);
	vessels_centers_L(myROI).copyTo(vessels_centers_L_mod);
	cv::copyMakeBorder(vessels_centers_L_mod, vessels_centers_L_mod, 1, 1, 1, 1, cv::BORDER_CONSTANT);	
	std::list<Branch> branchesOfAllTrees =
		exploreSkeleton(vessels_centers_L_mod, dtr);
	int num = 0;
	cv::Mat fake_center = (cv::Mat_<double>(1, 3) << 0, 0, 0);
	cv::Mat all_centers_plus_one = (cv::Mat_<double>(1, 3) << 0, 0, 0);
	std::vector<double> branch3dradii;
	double minradius = 99999;
	for (const auto& element : branchesOfAllTrees)
	{
		Branch3d currentBranch(element, warp_m1,warp_m2);
		branch3dradii = currentBranch.getRadii();
		for (auto& element : branch3dradii)
		{
			if (element < minradius)
			{
				minradius = element;
			}
		}

	
		if (currentBranch.getCentersCoord3d().rows > 5) 
		{ 
			all_points->splice(all_points->end(),
				findAllCrfsOfBranchMean3(currentBranch));
			cv::Mat ctrs = currentBranch.getCentersCoord3d();
			cv::vconcat(all_centers_plus_one, ctrs.rowRange(1, ctrs.rows - 1),
				all_centers_plus_one);
			if (ctrs.rows % 2 != 0) {
				cv::vconcat(all_centers_plus_one, fake_center, all_centers_plus_one);
			}
		}
	}
	
	*minrad = minradius;
	*all_centers = all_centers_plus_one.rowRange(1, all_centers_plus_one.rows);
	int t8 = clock();
	recontimeold << (t8 - to) / static_cast<double>(CLOCKS_PER_SEC) << ",";
	std::cout << "reconstruction time" << (t8 - to) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
}

void openingAndClosingInterface() {
	switch_slider = 0;
	opening_slider = 0;
	closing_slider = 0;
	apply_morphological_operations = true;
	original.copyTo(morphological_image);
	cv::namedWindow("Morphological Operations", cv::WINDOW_AUTOSIZE);
	cv::createTrackbar("Opening", "Morphological Operations",
		&opening_slider, opening_slider_max, changeOpeningAndClosing);
	cv::createTrackbar("Closing", "Morphological Operations",
		&closing_slider, closing_slider_max, changeOpeningAndClosing);
	cv::createTrackbar("Invert order", "Morphological Operations",
		&switch_slider, switch_slider_max, changeOpeningAndClosing);
	changeOpeningAndClosing(opening_slider, 0);
	cv::waitKey(0);
	cv::destroyWindow("Morphological Operations");
}


void changeOpeningAndClosing(int, void*) {
	switch_value = switch_slider;
	opening_size = opening_slider;
	closing_size = closing_slider;
	element_o = getStructuringElement(cv::MORPH_ELLIPSE,
		                              cv::Size(opening_size + 1, opening_size + 1));
	element_c = getStructuringElement(cv::MORPH_ELLIPSE,
		                              cv::Size(closing_size + 1, closing_size + 1));
	morphological_image = morphologicalOperations(original);
	cv::imshow("Morphological Operations", morphological_image);
}


cv::Mat morphologicalOperations(cv::Mat& pre) {
	cv::Mat post;
	if (apply_morphological_operations) {
		if (switch_value = 0) {
			cv::morphologyEx(pre, post, cv::MORPH_OPEN, element_o);
			cv::morphologyEx(post, post, cv::MORPH_CLOSE, element_c);
		} else {
			cv::morphologyEx(pre, post, cv::MORPH_CLOSE, element_c);
			cv::morphologyEx(post, post, cv::MORPH_OPEN, element_o);
		}
	} else {
		post = pre;
	}
	return post;
}

void printTipTriangulation() {
	cv::Mat LeftPos = (cv::Mat_<double>(4, 3) << -0.001214151033252 ,0.019751914026335, 0.000000279242639,
	0.017568309880265, 0.003261235549883, 0.000000147268971,
	0.009111116835320 ,- 0.003655449098305, - 0.000002790949687,
	788.888413019074050, 222.568734000944740, 0.427738916661315);

	cv::Mat RightPos = (cv::Mat_<double>(4, 3) << -0.001774173442456, 0.019609425867429, - 0.000000147520791,
	0.018673284492501, 0.003437157752487, 0.000000512987018,
	0.006347580232329, - 0.003534691824025, - 0.000003624449195,
	728.051829428587780, 201.923487452608840, 0.358325816404055);


	cv::Point2d Lc(222, 424);
	cv::Point2d Rc(295, 380);
	cv::Mat absolute_4d_coord;

	// triangulation
	cv::triangulatePoints(
		LeftPos, RightPos, cv::Mat(Lc),
		cv::Mat(Rc), absolute_4d_coord);
	cv::Mat homog_coord = absolute_4d_coord.t();
	cv::Mat euclid_coord;
	cv::convertPointsFromHomogeneous(homog_coord.reshape(4), euclid_coord);
	std::cout << "vision -> triangulation of the tip: " << euclid_coord.at<double>(0, 0)
		<< ", " << euclid_coord.at<double>(0, 1) << ", " << euclid_coord.at<double>(0, 2) << std::endl;
}

