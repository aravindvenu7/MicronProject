// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include "compute_circumferences_points.h"

#include <cmath>

const double PI = 3.141592653589793;

std::list<cv::Point3d> findAllCrfsOfBranchMean3(Branch3d currentBr) {
	std::list<cv::Point3d> circumferences_points;
	cv::Mat centers = currentBr.getCentersCoord3d();
	std::vector<double> radii = currentBr.getRadii();
	cv::Point3d versor_1;
	cv::Point3d versor_2;
	cv::Point3d versor_3;
	if (centers.rows > 3) {
		for (int i = 1; i < centers.rows - 1; i += 1) {                 //(int i = 1; i < centers.rows - 1; i += 2)
			versor_1 = versor_2;
			versor_2 = versor_3;
			versor_3 = tangentToCurve(cv::Point3_<double>(centers.row(i - 1)),
				cv::Point3_<double>(centers.row(i + 1)));       
			//versor_3 = tangentToCurve(cv::Point3_<double>(centers.row(i - 1)),
				//cv::Point3_<double>(centers.row(i + 1)));
			cv::Point3d center = cv::Point3_<double>(centers.row(i));
			cv::Point3d versor = meanTangent(versor_1, versor_2, versor_3, i);
			double rad = radii[i];
			cv::Point3d centerfinalx = cv::Point3_<double>(centers.row(i));
			//circumferences_points.push_back(centerfinalx);
			
			//if (rad > 400.0)
			//{
				circumferences_points.splice(circumferences_points.end(),
					computeCrfPts(center, versor, rad));
			//}
		}
	}
	return circumferences_points;
}

std::list<cv::Point3d> findAllCrfsOfBranchMean2(Branch3d currentBr) {
	std::list<cv::Point3d> circumferences_points;
	cv::Mat centers = currentBr.getCentersCoord3d();
	std::vector<double> radii = currentBr.getRadii();
	cv::Point3d versor_1;
	cv::Point3d versor_2;
	if (centers.rows > 3) {
		for (int i = 1; i < centers.rows - 1; i += 1) {
			versor_1 = versor_2;
			versor_2 = tangentToCurve(cv::Point3_<double>(centers.row(i - 1)),
				cv::Point3_<double>(centers.row(i + 1)));
			cv::Point3d center = cv::Point3_<double>(centers.row(i));
			cv::Point3d versor = meanTangent(versor_1, versor_2, i);
			double rad = radii[i];
			circumferences_points.splice(circumferences_points.end(),
				computeCrfPts(center, versor, rad));
		}
	}
	return circumferences_points;
}

std::list<cv::Point3d> findAllCrfsOfBranchNoMean(Branch3d currentBr) {
	std::list<cv::Point3d> circumferences_points;
	cv::Mat centers = currentBr.getCentersCoord3d();
	std::vector<double> radii = currentBr.getRadii();
	if (centers.rows > 3) {
		for (int i = 1; i < centers.rows - 1; i++) {
			cv::Point3d center = cv::Point3_<double>(centers.row(i));
			cv::Point3d versor =
				tangentToCurve(cv::Point3_<double>(centers.row(i - 1)),
					cv::Point3_<double>(centers.row(i + 1)));
			double rad = radii[i];

			circumferences_points.splice(circumferences_points.end(),
				computeCrfPts(center, versor, rad));
		}
	}
	return circumferences_points;
}

std::list<cv::Point3d> computeCrfPts(cv::Point3d center,
	cv::Point3d normal_versor, double radius) {


		std::list<cv::Point3d> crf;
		int n_points_crf = 36;//36;
		cv::Point3d i, j, p;
		i.x = -normal_versor.y;
		i.y = normal_versor.x;
		i.z = normal_versor.z;
		j = normal_versor.cross(i);
		for (int deg = 0; deg < 360; deg += 360 / n_points_crf) {
			double rad = deg * PI / 180;
			p.x = center.x + radius * cos(rad) * i.x + radius * sin(rad) * j.x;
			p.y = center.y + radius * cos(rad) * i.y + radius * sin(rad) * j.y;
			p.z = center.z + radius * cos(rad) * i.z + radius * sin(rad) * j.z;
			crf.push_back(p);
		}
		return crf;

		
	

}

cv::Point3d tangentToCurve(cv::Point3d pre_center, cv::Point3d post_center) {
	cv::Point3d versor;
	versor.x = post_center.x - pre_center.x;
	versor.y = post_center.y - pre_center.y;
	versor.z = post_center.z - pre_center.z;
	return versor / cv::norm(versor);
}

cv::Point3d meanTangent(cv::Point3d first_versor, cv::Point3d second_versor,
	cv::Point3d current_versor, int index) {
	cv::Point3d mean_versor;
	if (index == 1) {
		mean_versor = current_versor;
	}
	if (index == 2) {
		mean_versor.x = (second_versor.x + current_versor.x) / 2;
		mean_versor.y = (second_versor.y + current_versor.y) / 2;
		mean_versor.z = (second_versor.z + current_versor.z) / 2;
	}
	else {
		mean_versor.x = (first_versor.x + second_versor.x + current_versor.x) / 3;
		mean_versor.y = (first_versor.y + second_versor.y + current_versor.y) / 3;
		mean_versor.z = (first_versor.z + second_versor.z + current_versor.z) / 3;
	}
	return mean_versor;
}

cv::Point3d meanTangent(cv::Point3d prev_versor, cv::Point3d current_versor,
	int index) {
	cv::Point3d mean_versor;
	if (index == 1) {
		mean_versor = current_versor;
	}
	else {
		mean_versor.x = (prev_versor.x + current_versor.x) / 2;
		mean_versor.y = (prev_versor.y + current_versor.y) / 2;
		mean_versor.z = (prev_versor.z + current_versor.z) / 2;
	}
	return mean_versor;
}