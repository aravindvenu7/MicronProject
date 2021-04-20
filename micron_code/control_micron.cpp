// Copyright (c) 2017 Simone Foti <simofoti@gmail.com>
/*
#include <windows.h>
#include <cmath>

#include "control_micron.h"
#include "estimate_distance_tip_structure.h"
#include "micronrecv.h"
#include "micronsend.h"
#include "3d_visualization.h"

const double SAFETY_DISTANCE = 1000;  // in um

// Communication with Micron (receiving and sending)
MicronRecvPacket packetIn;
MicronRecvSocket micronIn;
MicronSendPacket packetOut;
MicronSendSocket micronOut;

void controlMicron(cv::Mat all_centers, std::list<cv::Point3d> all_points) {
	// Start listening on default port and connect to Micron realtime target
	micronIn.start();
	micronOut.start("192.168.1.101");
	while (true) {
		if (micronIn.recv(packetIn)) {
			std::cout << "Connected with Micron!" << std::endl;
			while (micronIn.recv(packetIn));
			cv::Point3d tip_position;			
			tip_position.x = packetIn.position_tip[0];
			tip_position.y = packetIn.position_tip[1];
			tip_position.z = packetIn.position_tip[2];

			// showCloudAndTip(all_points, tip_position);

			double distance_tip_cloud;
			cv::Point3d versor_point_to_tip;
			estimateTipDistanceAndNormal(tip_position, all_centers, all_points,
				&distance_tip_cloud, &versor_point_to_tip);
			// std::cout << "received pos: " << tip_position << " \t | ";
			std::cout << "distance: " << distance_tip_cloud << " um" << std::endl;
			if (distance_tip_cloud <= SAFETY_DISTANCE) {
				std::cout << "attention" << std::endl;
				cv::Point3d tip_goal = computeGoalPosition(distance_tip_cloud, versor_point_to_tip, tip_position);
				packetOut.goalPos[0] = tip_goal.x;
				packetOut.goalPos[1] = tip_goal.y;
				packetOut.goalPos[2] = tip_goal.z;
				// If this flag is not on, Micron will default to normal behavior and will disregard our goal positions
				packetOut.useGoalPos = 1;
				packetOut.extraInfo[0] = 0; // Turn laser off 
				micronOut.send(packetOut);
			} else {
				for (int i = 0; i < 3; i++) {
					packetOut.goalPos[i] = packetIn.goal_filter_tip[i];
				}
				packetOut.useGoalPos = 1;
				micronOut.send(packetOut);
			}
		} else {
			waitConnection();
		} 
	}
}

cv::Point3d computeGoalPosition(double dist, cv::Point3d versor,
	cv::Point3d tip) {
	return (((SAFETY_DISTANCE - dist) * versor) + tip);  // in um
}

void waitConnection() {
	std::cout << "Waiting for connection" << std::flush;
	for (int i = 0; i < 3; i++) {
		std::cout << ".";
		Sleep(200);
	}
	std::cout << "\b\b\b" << "   ";
	std::cout << "\r";
	Sleep(200);
}
*/
