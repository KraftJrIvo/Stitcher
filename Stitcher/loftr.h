#pragma once

#include <opencv2/opencv.hpp>

class LOFTR
{
public:
	LOFTR();
	void match(cv::Mat img1, cv::Mat img2, cv::Mat msk1, cv::Mat msk2, std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);
private:
	void _request_matches(cv::Mat img1, cv::Mat img2, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& matches);
};
