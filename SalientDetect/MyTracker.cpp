#include "stdafx.h"
#include "MyTracker.h"


MyTracker::MyTracker()
{
	state = sleeping;
	puzzleFrames = 0;
	life = 0;
}


MyTracker::~MyTracker()
{
}

void MyTracker::init(cv::Rect & roi, cv::Mat & image, int tracker_id, int targert_id)
{
	mTracker.init(roi, image);
	state = tracking;
	allow_train = true;
	update_by_detect = false;
	trackerId = tracker_id;
	targetId = targert_id;
}

cv::Rect MyTracker::update(cv::Mat & image, float thres)
{
	cv::Rect res=mTracker.update(image);
	if (res.width > image.cols / 2 || res.height > image.rows / 2) {
		recycle();
		return cv::Rect();
	}
	if (mTracker.psr < thres)
		mTracker.interp_factor = mTracker.base_lr * 2.5;
	else
		mTracker.interp_factor = mTracker.base_lr;
	//mTracker.interp_factor = mTracker.base_lr / mTracker.psr;
	pTarget->directx = res.x - pTarget->x;
	pTarget->directy = res.y - pTarget->y;
	pTarget->x = res.x;
	pTarget->y = res.y;
	pTarget->width = res.width;
	pTarget->height = res.height;
	pTarget->b_new = false;
	++life;
	return res;
}

void MyTracker::recycle()
{
	std::cout << "tracker " << trackerId << " recycled" << std::endl;
	state = sleeping;
	puzzleFrames = 0;
	pTarget->life = 0;
	life = 0;
}

void MyTracker::forceUpdate()
{
	std::cout << "tracker " << trackerId << " force-updated" << std::endl;
	state = puzzled;
	// puzzleFrames = 0;
	update_by_detect = true;
	//life = 0;
}
