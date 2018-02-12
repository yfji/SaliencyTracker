#pragma once
#include <opencv2\opencv.hpp>
#include "MyTracker.h"
#include "target.h"
#include "Implement.h"
#include <vector>

#define MAX_N	10
class Tracker
{
public:
	Tracker();
	~Tracker();

public:
	void addTarget(int x, int y, int w, int h);
	void updateTarget(std::shared_ptr<target>& t, int x, int y, int w, int h, int directx, int directy, int dist, float score = 0.0, target* prev = nullptr);
	void removeTarget(int uuid);

	void detect_filter(cv::Mat& curFrame);
	void track(cv::Mat& curFrame);
	void drawBoundingBox(cv::Mat& curFrame, int scale=2);

	void nms();

private:
	int filter_len;
	int cur_index;
	int tracker_num;
	const int max_diff = 20;
	const float psr_thres = 0.9;
	const int nForceUpdate = 6;
	const int nRecycle = 12;
	Implement impl;
	std::vector<std::shared_ptr<target>> targets;
	MyTracker trackers[MAX_N];

private:
	void initTrackers(cv::Mat& image);
};

