/**
* Code for thinning a binary image using Zhang-Suen algorithm.
*
* Author:  Nash (nash [at] opencv-code [dot] com)
* Website: https://github.com/bsdnoobz/zhang-suen-thinning
*/

#ifndef SRC_THIRD_PARTY_THINNING_ZS_H_
#define SRC_THIRD_PARTY_THINNING_ZS_H_

#include <opencv2/opencv.hpp>
void thinningIteration(cv::Mat& img, int iter);
void thinning(const cv::Mat& src, cv::Mat& dst);

#endif /* SRC_THIRD_PARTY_THINNING_ZS_H_ */
