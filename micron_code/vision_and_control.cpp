// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>

#include <windows.h>
#include <thread>
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>


#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/surface_matching/icp.hpp"



#include "vision_and_control.h"
#include "estimate_distance_tip_structure.h"
#include "micronrecv.h"
#include "micronsend.h"
#include "vision.h"
#include "3d_visualization.h"
#include  "camera.h"

const double SAFETY_DISTANCE = 400;  // in um should be 700

const bool RECORD_CLOUD =true;
const bool RECORD_TRAJ =true;
std::string id("41");

// Communication with Micron (receiving and sending)
MicronRecvPacket packetIn;
MicronRecvSocket micronIn;
MicronSendPacket packetOut;
MicronSendSocket micronOut;

cv::Mat all_centers_old;
std::list<cv::Point3d> all_points_old;
cv::Point3d oldtip;
bool written_cloud = false;
bool first_cloud_created = false;
bool firstcloud = true;
double minimumradius;

void controlMicronWithVision() 
{
	cv::Mat all_centers_control;                                         ///Write function in this class to directly calculate tip x and y in left image from the tip 3d coordinates in packetIn
	std::list<cv::Point3d> all_points_control;
	double tsize;
	cv::Mat transformed_centers;
	std::list<cv::Point3d> transformed_cloud;
	double minimumradiusfirst;

	if(micronIn.recv(packetIn))
	{
		oldtip.x = packetIn.goal_tip[0];
		oldtip.y = packetIn.goal_tip[1];
		oldtip.z = packetIn.goal_tip[2];
	}
	//std::cout << "Tip x" << oldtip << std::endl;
	std::thread vision_thread(cloudComputation, std::ref(all_points_old), 
		                      std::ref(all_centers_old), std::ref(written_cloud),std::ref(minimumradius), oldtip);
	
	//std::cout << "thread returns here" << std::endl;
	//std::thread picture_thread()
	// Start listening on default port and connect to Micron realtime target
	micronIn.start();
	micronOut.start("192.168.1.102");
	//std::cout << "we are here before trajectory" << std::endl;
	std::ofstream out_normal("trajectory_without_correction" + id + ".txt");
	std::ofstream out_correc("trajectory_with_correction" + id + ".txt");
	std::ofstream goaltipfile("goaltip" + id + ".txt");
	std::ofstream goaltipdistancefile("goaltipdistance" + id + ".txt");
	std::ofstream out_closest("closest_point_of_cloud1st" + id + ".txt");
	std::ofstream corrected_distance("corrected_distance" + id + ".txt");
	std::ofstream uncorrected_distance("uncorrected_distance" + id + ".txt");
	std::ofstream out("cloud1" + id + ".txt");
	std::ofstream cradi("closestradii" + id + ".txt");
	std::ofstream outlink("linkerror1" + id + ".txt");
	std::ofstream initial_cloud("initial_cloud1" + id + ".txt");
	
	while (true) 
	{
		int t1 = clock();
		if (firstcloud && written_cloud)
		{
			transformed_centers = all_centers_old;
			transformed_cloud = all_points_old;
			tsize = transformed_cloud.size();
			//transformed_cloud_size = all_points_old.size();
			firstcloud = false;

			std::cout << "entered ere" << std::endl;
			minimumradiusfirst = minimumradius;
			if (RECORD_CLOUD)
			{
				cv::Mat matcloud2(transformed_cloud.size(), 1, CV_64FC3);
				int i = 0;
				for (auto &p : transformed_cloud) {
					matcloud2.at<cv::Vec3d>(i++) = p;
				}
				matcloud2 = matcloud2.reshape(1);


				matcloud2.convertTo(matcloud2, CV_32F);
				cv::ppf_match_3d::writePLY(matcloud2, "D:/stereo_dataset/cloud5th.ply");
				// Save cloud in txt
				//std::cout << "size of pointcloud" << all_points_control.size() << std::endl;
				/*for (const auto& point : transformed_cloud)
				{//transformed_cloud
					initial_cloud << point.x << ",";
					initial_cloud << point.y << ",";
					initial_cloud << point.z << ",";
					initial_cloud << " ";
				}*/

			}
		}
		//int start = clock();
		if (written_cloud) {  //written_cloud
			
			all_centers_control = all_centers_old;
			all_points_control = all_points_old;
			written_cloud = false;
			first_cloud_created = true;
			/*if ( minimumradius < minimumradiusfirst) //all_points_control.size() < tsize ||
					{
						  all_centers_control = transformed_centers;
						  all_points_control = transformed_cloud;

					}*/

			/*if (RECORD_CLOUD) {
				// Save cloud in txt
				//std::cout << "size of pointcloud" << all_points_control.size() << std::endl;
				for (const auto& point : all_points_control) {
					out << point.x << ",";
					out << point.y << ",";
					out << point.z << ",";
					out << " ";
				}
				out << "newcloud";
				//break;
				// out.close();
				if (RECORD_TRAJ) {
					// tells when a new cloud was created
					/*out_traj << "new";
					out_no_cont << "new";
					out_closest << "new";
				}
				//std::cout << "cloud saved";
			}*/
		}
		
		if (micronIn.recv(packetIn) && first_cloud_created) {
			while (micronIn.recv(packetIn));


			cv::Point3d tip_position, tip_position_temp;
			tip_position.x = packetIn.null_pos_tip[0]; //goal_tip
			tip_position.y = packetIn.null_pos_tip[1];
			tip_position.z = packetIn.null_pos_tip[2];
			tip_position_temp.x = packetIn.goal_tip[0]; //goal_tip
			tip_position_temp.y = packetIn.goal_tip[1];
			tip_position_temp.z = packetIn.goal_tip[2];
			
			//std::cout << "we are inside the control thread" << std::endl;
			cv::Point3d tip_position_for_visualization;
			tip_position_for_visualization.x = packetIn.position_tip[0];
			tip_position_for_visualization.y = packetIn.position_tip[1];
			tip_position_for_visualization.z = packetIn.position_tip[2];//- 11800;
			
			showCloudAndTip(all_points_control, tip_position_for_visualization);

			double distance_tip_cloud;
			double  distance_tip_center, closest_radius;
			
			cv::Point3d versor_point_to_tip,versor_point_to_center, versor_closest_to_center;
			
			cv::Point3d closest_point, closest_center;
			
			estimateTipDistanceNormalAndClosest(tip_position, all_centers_control,
				all_points_control,	&distance_tip_cloud, &versor_point_to_tip, &closest_point, &closest_center);
			versor_point_to_center = calcversor(closest_center, tip_position);
			distance_tip_center = calcdistance(closest_center, tip_position);
			closest_radius = calcdistance(closest_center, closest_point);
			
			//showCloudAndTipandclosest(all_points_control, tip_position_for_visualization, closest_point);
			//showCloudAndTipandclosestcenter(all_points_control, tip_position_for_visualization,closest_point,closest_center);
			// std::cout << "received pos: " << tip_position << " \t | ";
			std::cout << "distance what: " << distance_tip_cloud << " um" << " \t | ";
			//uncorrected_distance << distance_tip_cloud << ",";
			std::cout << "distance from center: " << distance_tip_center << " um" << " \t | ";
			
			if (RECORD_TRAJ) {
				// Save trajectories in txt (open file outside the while)
				out_normal << tip_position;
				//out_no_cont << tip_position;
				//uncorrected_distance << calcdistance(closest_point, tip_position) << ",";
				uncorrected_distance << calcdistance(closest_center, tip_position) << ",";
				//corrected_distance << calcdistance(closest_point, tip_position_for_visualization) << ",";
				corrected_distance << calcdistance(closest_center, tip_position_for_visualization) << ",";
				//goaltipdistancefile << calcdistance(closest_point, tip_position_temp);
				//out_closest << tip_position - distance_tip_cloud*versor_point_to_tip;
				out_correc << tip_position_for_visualization;
				cradi << closest_radius << ",";
				//outlink << (double)packetIn.link_error_0[0] << "," << (double)packetIn.link_error_0[1] << (double)packetIn.link_error_0[2] << "," << (double)packetIn.link_error_1[0] << (double)packetIn.link_error_1[1] << "," << (double)packetIn.link_error_1[2] << ";";
			}
			
			if (distance_tip_center<= SAFETY_DISTANCE ) // if (distance_tip_center <= SAFETY_DISTANCE + closest_radius) 
			{ 
				//std::cout << "transmitting tip correction" << std::endl;
				cv::Point3d tip_goal = computeGoalPosition_INSIDE(closest_radius,distance_tip_center, versor_point_to_tip, tip_position);
				std::cout << "!!!" << tip_goal << " \t | ";
				std::cout << "new distance:" << distance_tip_center + cv::norm(tip_goal - tip_position) << " um" << " \t | ";
				//corrected_distance << distance_tip_cloud + cv::norm(tip_goal - tip_position) << ",";
				//packetOut.useGoalPos = 1;
				packetOut.goalPos[0] = tip_goal.x;
				packetOut.goalPos[1] = tip_goal.y;
				packetOut.goalPos[2] = tip_goal.z;

				/*tip_position_for_visualization.x = tip_goal.x;
				tip_position_for_visualization.y = tip_goal.y;
				tip_position_for_visualization.z = tip_goal.z;
				*/
				// If this flag is not on, Micron will default to normal behavior and will disregard our goal positions
				packetOut.useGoalPos = 1;
				packetOut.extraInfo[0] = 0; // Turn laser off 
				micronOut.send(packetOut);
				std::cout << "goal tip" << tip_goal<< std::endl;
				goaltipfile << tip_goal;
			}
			else {				
				cv::Point3d goal(packetIn.goal_tip[0], packetIn.goal_tip[1], packetIn.goal_tip[2]);
				for (int i = 0; i < 3; i++) {
					packetOut.goalPos[i] = packetIn.goal_tip[i];
				}				
				packetOut.useGoalPos = 0;
				//std::cout << "distance: " << distance_tip_cloud << " um" << " \t | ";
				std::cout << "closest point" << closest_point << " \t | ";
				
				std::cout << "normal operation" << goal << std::endl;
				
				micronOut.send(packetOut);

			}


		}
		else {
			waitConnection();
		}
		int end = clock();
		//Sleep(15);
		//std::cout << "total time" << (end - t1) / static_cast<double>(CLOCKS_PER_SEC) << std::endl;
		//vision_thread.join();
	}
	vision_thread.join();
}

cv::Point3d computeGoalPosition(double dist, cv::Point3d versor,
	cv::Point3d tip) {
	return (((SAFETY_DISTANCE - dist) * versor) + tip);  // in um  //SAFETY_DISTANCE
}

cv::Point3d computeGoalPosition_INSIDE(double closest_radius, double distance_tip_center, cv::Point3d versor,
	cv::Point3d tip) {
	return (((( SAFETY_DISTANCE - distance_tip_center  ) * versor)) + tip);  // in um  //+SAFETY_DISTANCE //closest_radius -  at the beginning
}

void waitConnection() {
	//std::cout << "Waiting for connection" << std::flush;
	for (int i = 0; i < 3; i++) {
		std::cout << ".";
		Sleep(50);
	}
	std::cout << "\b\b\b" << "   ";
	std::cout << "\r";
	Sleep(50); 
}

cv::Point3d calcversor(cv::Point3d center, cv::Point3d tip)
{
	cv::Point3d vector_point_to_tip = tip - center;
	return  vector_point_to_tip / cv::norm(vector_point_to_tip);
}

double calcdistance(cv::Point3d p1, cv::Point3d p2)
{
	return 	sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
		pow(p1.z - p2.z, 2));
}

