#pragma once
#include <opencv2\opencv.hpp>
#include "MyTracker.h"
#include "target.h"
#include "Implement.h"
#include <vector>
#include <thread>

#define MAX_N	20
class Tracker
{
public:
	Tracker();
	~Tracker();

private:
	void initTrackers(cv::Mat& image);
	void trackThreadRef(MyTracker& t, cv::Mat& im);
	void trackThreadPtr(MyTracker* t, cv::Mat* im);
public:
	int addTarget(int x, int y, int w, int h);
	void updateTarget(std::shared_ptr<target>& t, int x, int y, int w, int h, int directx, int directy, int dist, float score = 0.0, target* prev = nullptr);
	void removeTarget(int uuid);

	void detect_filter(cv::Mat& curFrame);
	void track(cv::Mat& curFrame);
	void drawBoundingBox(cv::Mat& curFrame, int scale=2);

	void nms();
	void detectNms();

private:
	int filter_len;
	int cur_index;
	int tracker_num;
	int tracker_id;
	int target_num;
	const int max_diff = 30;
	float psr_thres{ 0.7f };
	const int nForceUpdate = 6;
	const int nRecycle = 11;
	float max_psr;
	Implement impl;
	std::vector<std::shared_ptr<target>> targets;
	MyTracker trackers[MAX_N];
	std::thread mThreads[MAX_N];
	const cv::Scalar colors[MAX_N] ={
		cv::Scalar(0,0,255),
		cv::Scalar(0,255,255),
		cv::Scalar(0,255,0),
		cv::Scalar(255,128,128),
		cv::Scalar(192,128,255),
		cv::Scalar(255,255,128),
		cv::Scalar(128,128,0),
		cv::Scalar(0,128,128),
		cv::Scalar(64,128,255),
		cv::Scalar(192,192,192),
		cv::Scalar(255,128,0),
		cv::Scalar(64,128,0),
		cv::Scalar(64,0,64),
		cv::Scalar(0,0,128),
		cv::Scalar(192,128,128),
		cv::Scalar(255,0,255),
		cv::Scalar(64,128,128),
		cv::Scalar(0,128,0),
		cv::Scalar(0,0,0),
		cv::Scalar(255,255,255)
	};
};

