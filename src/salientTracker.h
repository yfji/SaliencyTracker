#pragma once
#include "MyTracker.h"
#include "target.h"
#include "Implement.h"
#include <fstream>
#include <vector>
#include <thread>

#define MAX_N	6
class SalientTracker
{
public:
	SalientTracker();
	~SalientTracker();

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
	void saveResults(ofstream& out, const string& filename);

	void nms();
	void detectNms();

private:
	int filter_len;
	int cur_index;
	int tracker_num;
	int tracker_id;
	int target_num;
	int gScale;
	const int max_diff = 20;
	float psr_thres{ 0.9f };
	float psr_thres_lower{ 0.75f };
	float psr_adapt_thres{ 1.2f };
	const int nForceUpdate = 4;
	const int nRecycle = 6;
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
		//cv::Scalar(128,128,0),
		//cv::Scalar(0,128,128),
		//cv::Scalar(64,128,255),
		//cv::Scalar(192,192,192),
		//cv::Scalar(255,128,0),
		//cv::Scalar(64,128,0),
		//cv::Scalar(64,0,64),
		//cv::Scalar(0,0,128),
		//cv::Scalar(192,128,128),
		//cv::Scalar(255,0,255),
		//cv::Scalar(64,128,128),
		//cv::Scalar(0,128,0),
		//cv::Scalar(0,0,0),
		//cv::Scalar(255,255,255)
	};
};

