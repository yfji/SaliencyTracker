#pragma once
#include "Salient.h"

class Implement
{
public:
	Implement();
	~Implement();

	cv::Mat findPeakRectArea(cv::Mat& im);
	std::vector<cv::Rect> parseCandidatePatches(const cv::Mat& im);
	void nms(std::vector<cv::Rect>& boxes, float thres=0.25);
	void merge(std::vector<cv::Rect>& boxes);

private:
	const int block_w = 5;
	const int block_h = 5;
	const int window_times = 3;
	Salient salient;
	cv::Mat kernel;
	cv::Mat patchKernel;
	cv::Mat getPatch(const cv::Mat& im, cv::Rect& rect);
};

