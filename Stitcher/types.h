#pragma once

#include <opencv2/opencv.hpp>

struct OverlapTransform {
	std::vector<cv::Point2f> pts1, pts2;
	int parent, child, nInlers;
	cv::Mat relT;
};