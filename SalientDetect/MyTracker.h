#pragma once
#include "kcf\kcftracker.hpp"
#include "states.h"
#include "target.h"
#include <memory>

class MyTracker
{
public:
	MyTracker();
	~MyTracker();

	void init(cv::Rect& roi, cv::Mat& image, int tracker_id, int targert_id);
	cv::Rect update(cv::Mat& image);
	void recycle();
	void forceUpdate();
	inline float calcPSR() {
		return mTracker.peak_value/mTracker.sigma;
	}
public:
	States state;
	int trackerId;
	int targetId;
	int puzzleFrames;
	bool allow_train{ true };
	bool update_by_detect{ false };
	std::shared_ptr<target> pTarget;
private:
	KCFTracker mTracker;
};

