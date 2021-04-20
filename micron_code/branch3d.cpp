// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include "branch3d.h"

#include <cmath>
// #include <opencv2/imgproc.hpp>
// #include "src/3d_reconstruction/3d_visualization.h"


cv::Mat LeftPos = (cv::Mat_<double>(4, 3) << -0.001214151033252 ,0.019751914026335, 0.000000279242639,
0.017568309880265, 0.003261235549883, 0.000000147268971,
0.009111116835320 ,- 0.003655449098305, - 0.000002790949687,
788.888413019074050, 222.568734000944740, 0.427738916661315);

cv::Mat RightPos = (cv::Mat_<double>(4, 3) << -0.001774173442456, 0.019609425867429, - 0.000000147520791,
0.018673284492501, 0.003437157752487, 0.000000512987018,
0.006347580232329, - 0.003534691824025, - 0.000003624449195,
728.051829428587780, 201.923487452608840, 0.358325816404055);

Branch3d::Branch3d(Branch currentBr, cv::Mat warp_mat1, cv::Mat warp_mat2) {
	centers_coord_left = currentBr.getCentersCoord();
	if (centers_coord_left.size() > 0) {
		centers_coord_right = findWarpedCenters(warp_mat1, centers_coord_left);
		// radii = currentBr.getRadii();
		vessel_number = currentBr.getBranchNumber();
		centers_coord_3d = find3dPoints(centers_coord_left, centers_coord_right);
		radii =
			computeRealRadii(currentBr.getRadii(), currentBr.getRadiiDirection(),
				centers_coord_left, centers_coord_3d, warp_mat2);
	}
}

std::vector<cv::Point2d> Branch3d::findWarpedCenters(
	cv::Mat warp_mat, std::vector<cv::Point2i> original_centers) {
	std::vector<cv::Point2d> warped_centers;
	for (const auto& current_center : original_centers) {
		warped_centers.push_back(findWarpedCoordinates(warp_mat, current_center));
	}
	return warped_centers;
}

cv::Point2d Branch3d::findWarpedCoordinates(cv::Mat warp_mat, cv::Point2i in) {
	cv::Point2d out;
	double M_11 = warp_mat.at<double>(0, 0);
	double M_12 = warp_mat.at<double>(0, 1);
	double M_13 = warp_mat.at<double>(0, 2);
	double M_21 = warp_mat.at<double>(1, 0);
	double M_22 = warp_mat.at<double>(1, 1);
	double M_23 = warp_mat.at<double>(1, 2);
	double M_31 = warp_mat.at<double>(2, 0);
	double M_32 = warp_mat.at<double>(2, 1);
	double M_33 = warp_mat.at<double>(2, 2);
	out.x =
		(in.x * M_11 + in.y * M_12 + M_13) / (in.x * M_31 + in.y * M_32 + M_33);
	out.y =
		(in.x * M_21 + in.y * M_22 + M_23) / (in.x * M_31 + in.y * M_32 + M_33);
	// std::cout << "point in:" << in << ";     point out:" << out << std::endl;
	return out;
}

cv::Point2d Branch3d::findWarpedCoordinates(cv::Mat warp_mat, cv::Point2d in) {
	cv::Point2d out;
	double M_11 = warp_mat.at<double>(0, 0);
	double M_12 = warp_mat.at<double>(0, 1);
	double M_13 = warp_mat.at<double>(0, 2);
	double M_21 = warp_mat.at<double>(1, 0);
	double M_22 = warp_mat.at<double>(1, 1);
	double M_23 = warp_mat.at<double>(1, 2);
	double M_31 = warp_mat.at<double>(2, 0);
	double M_32 = warp_mat.at<double>(2, 1);
	double M_33 = warp_mat.at<double>(2, 2);
	out.x =
		(in.x * M_11 + in.y * M_12 + M_13) / (in.x * M_31 + in.y * M_32 + M_33);
	out.y =
		(in.x * M_21 + in.y * M_22 + M_23) / (in.x * M_31 + in.y * M_32 + M_33);
	// std::cout << "point in:" << in << ";     point out:" << out << std::endl;
	return out;
}

cv::Mat Branch3d::find3dPointsDistortionCorrection() {
	cv::Mat LeftIntrinsic =
		(cv::Mat_<double>(3, 3) << 751.706129, 0.000000, 338.665467, 0.000000,
			766.315580, 257.986032, 0.000000, 0.000000, 1.000000);
	cv::Mat LeftDist =
		(cv::Mat_<double>(1, 4) << -0.328424, 0.856059, 0.003430, 0.000248);
	cv::Mat RightIntrinsic =
		(cv::Mat_<double>(3, 3) << 752.737796, 0.000000, 263.657845, 0.000000,
			765.331797, 245.883296, 0.000000, 0.000000, 1.000000);
	cv::Mat RightDist =
		(cv::Mat_<double>(1, 4) << -0.332222, 0.708196, 0.003904, 0.008043);
	cv::Mat LeftPos =
		(cv::Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	cv::Mat RightPos =
		(cv::Mat_<double>(4, 3) << 0.9999, 0.0069, -0.0121, -0.0068, 1.0000,
			0.0055, 0.0121, -0.0054, 0.9999, 5.3754, 0.0716, 0.1877);

	cv::Mat left_undistorted, right_undistorted;
	cv::Mat centers_coord_left_mat = cv::Mat(centers_coord_left);
	centers_coord_left_mat.convertTo(centers_coord_left_mat, CV_64F);
	cv::undistortPoints(centers_coord_left_mat, left_undistorted, LeftIntrinsic,
		LeftDist);
	cv::undistortPoints(cv::Mat(centers_coord_right), right_undistorted,
		RightIntrinsic, RightDist);

	cv::Mat centersHomogeneous, centersEuclidean;
	cv::triangulatePoints(LeftPos, RightPos.t(), left_undistorted,
		right_undistorted,
		centersHomogeneous);  // transpose .t() invert .inv()
	cv::Mat cH = centersHomogeneous.t();
	cv::convertPointsFromHomogeneous(cH.reshape(4), centersEuclidean);
	// showCloud(centersEuclidean);

	return centersEuclidean.reshape(1);
}

cv::Mat Branch3d::find3dPoints(std::vector<cv::Point2i> coord_left,
	std::vector<cv::Point2d> coord_right) {
	cv::Mat coord_left_mat = cv::Mat(coord_left);
	coord_left_mat.convertTo(coord_left_mat, CV_64F);
	cv::Mat centersHomogeneous, centersEuclidean;
	cv::triangulatePoints(LeftPos.t(), RightPos.t(), coord_left_mat,
		cv::Mat(coord_right),
		centersHomogeneous);  // transpose .t() invert .inv()
	cv::Mat cH = centersHomogeneous.t();
	cv::convertPointsFromHomogeneous(cH.reshape(4), centersEuclidean);
	return centersEuclidean.reshape(1);
}

cv::Mat Branch3d::find3dPoints(std::vector<cv::Point2d> coord_left,
	std::vector<cv::Point2d> coord_right) {
	cv::Mat coord_left_mat = cv::Mat(coord_left);
	coord_left_mat.convertTo(coord_left_mat, CV_64F);
	cv::Mat centersHomogeneous, centersEuclidean;
	cv::triangulatePoints(LeftPos.t(), RightPos.t(), coord_left_mat,
		cv::Mat(coord_right),
		centersHomogeneous);  // transpose .t() invert .inv()
	cv::Mat cH = centersHomogeneous.t();
	cv::convertPointsFromHomogeneous(cH.reshape(4), centersEuclidean);
	return centersEuclidean.reshape(1);
}

std::vector<double> Branch3d::computeRealRadii(
	std::vector<double> radii_2d, std::vector<cv::Point2d> radii_direction_2d,
	std::vector<cv::Point2i> centers_left, cv::Mat centers_3d, cv::Mat warp_m) {
	std::vector<double> results;
	// cv::Mat img_sx =
	//     imread("/home/simo/Immagini/imgTesi/sx1.png", cv::IMREAD_COLOR);
	assert(radii_2d.size() == radii_direction_2d.size() &&
		radii_2d.size() == centers_left.size());
	cv::Mat radii_2d_mat = cv::Mat(radii_2d);
	cv::Mat radii_direction_2d_mat = cv::Mat(radii_direction_2d);
	cv::Mat centers_left_mat = cv::Mat(centers_left);
	for (int i = 0; i < radii_2d_mat.rows; i++) {
		// compute vectors of 2d radii multiplying versor for radius
		cv::Point2d vect_radius_2d;
		vect_radius_2d.x =
			radii_direction_2d_mat.at<double>(i, 0) * radii_2d_mat.at<double>(i, 0);
		vect_radius_2d.y =
			radii_direction_2d_mat.at<double>(i, 1) * radii_2d_mat.at<double>(i, 1); //was 0 here before

		// finds diametral point in image coordinates summing center position with
		// radius vector
		cv::Point2d vect_absolute_radius_2d;
		vect_absolute_radius_2d.x =
			vect_radius_2d.x + centers_left_mat.at<int>(i, 0);
		vect_absolute_radius_2d.y =
			vect_radius_2d.y + centers_left_mat.at<int>(i, 1);


		// finds corresponding point in other image (warping)
		cv::Point2d vect_absolute_radius_2d_right =
			findWarpedCoordinates(warp_m, vect_absolute_radius_2d);
		cv::Mat absolute_4d_radius_coord;

		// triangulation
		cv::triangulatePoints(
			LeftPos.t(), RightPos.t(), cv::Mat(vect_absolute_radius_2d),
			cv::Mat(vect_absolute_radius_2d_right), absolute_4d_radius_coord);
		cv::Mat homog_coord = absolute_4d_radius_coord.t();
		cv::Mat euclid_coord;
		cv::convertPointsFromHomogeneous(homog_coord.reshape(4), euclid_coord);

		// euclidean distance from the 3d center
		results.push_back(sqrt(
			pow(euclid_coord.at<double>(0, 0) - centers_3d.at<double>(i, 0), 2) +
			pow(euclid_coord.at<double>(0, 1) - centers_3d.at<double>(i, 1), 2) +
			pow(euclid_coord.at<double>(0, 2) - centers_3d.at<double>(i, 2), 2)));
	}

	return results;
}

cv::Mat Branch3d::getCentersCoord3d() { return centers_coord_3d; }

std::vector<double> Branch3d::getRadii() { return radii; }

int Branch3d::getVesselNumber() { return vessel_number; }