#include "stdafx.h"
#include "MyTracker.h"


MyTracker::MyTracker()
{
	state = sleeping;
	puzzleFrames = 0;
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

cv::Rect MyTracker::update(cv::Mat & image)
{
	cv::Rect res=mTracker.update(image);
	pTarget->directx = res.x - pTarget->x;
	pTarget->directy = res.y - pTarget->y;
	pTarget->x = res.x;
	pTarget->y = res.y;
	pTarget->width = res.width;
	pTarget->height = res.height;
	return res;
}

void MyTracker::recycle()
{
	std::cout << "tracker " << trackerId << " recycled" << std::endl;
	state = sleeping;
	puzzleFrames = 0;
	pTarget->life = 0;
}

void MyTracker::forceUpdate()
{
	std::cout << "tracker " << trackerId << " force-updated" << std::endl;
	state = puzzled;
	// puzzleFrames = 0;
	update_by_detect = true;
}
