#pragma once
#include "Salient.h"

class Implement
{
public:
	Implement();
	~Implement();

	cv::Mat findPeakRectArea(cv::Mat& im);
	std::vector<cv::Rect> parseCandidatePatches(cv::Mat& im);
	void nms(std::vector<cv::Rect>& boxes);
	void merge(std::vector<cv::Rect>& boxes);

private:
	const int block_w = 5;
	const int block_h = 5;
	const int window_times = 3;
	Salient salient;
	cv::Mat kernel;
	cv::Mat patchKernel;
	cv::Mat getPatch(cv::Mat& im, cv::Rect& rect);
};

