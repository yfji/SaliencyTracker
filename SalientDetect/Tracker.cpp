#include "stdafx.h"
#include "Tracker.h"


Tracker::Tracker()
{
	tracker_num = 0;
}


Tracker::~Tracker()
{
}

void Tracker::addTarget(int x, int y, int w, int h, int directx, int directy, int dist, float score, target* prev)
{
	std::shared_ptr<target> t=std::make_shared<target>();
	t->x = x;
	t->y = y;
	t->width = w;
	t->height = h;
	t->directx = directx;
	t->directx = directy;
	t->dist = dist;
	t->score = score;
	t->prev = prev;
	t->uuid = targets.size();
	targets.push_back(t);
}

void Tracker::updateTarget(std::shared_ptr<target>& t, int x, int y, int w, int h, int directx, int directy, int dist, float score, target* prev)
{
	t->x = x;
	t->y = y;
	t->width = w;
	t->height = h;
	t->directx = directx;
	t->directx = directy;
	t->dist = dist;
	t->score = score;
	t->prev = prev;
}

void Tracker::removeTarget(int uuid)
{
}

void Tracker::detect_filter(cv::Mat & curFrame)
{
	std::vector<cv::Rect> bboxes = impl.parseCandidatePatches(curFrame);
	if (targets.size() == 0) {
		for (auto i = 0; i < bboxes.size(); ++i) {
			addTarget(bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height, 0, 0, 0);
		}
		return;
	}
	for (auto i = 0; i < bboxes.size(); ++i) {
		cv::Rect& t_box = bboxes[i];
		if (t_box.width == 0 || t_box.height == 0 || t_box.width>curFrame.cols/2 || t_box.height>curFrame.rows/2) {
			continue;
		}
		bool box_new = true;
		//for (auto j = 0; j < targets.size(); ++j) {
		//	target& t = targets[j];
		//	int dist = abs(t_box.x - t.x) + abs(t_box.y - t.y);
		//	if (t.life!=0 && dist < max_diff && dist<t.dist) {
		//		if(t.update_by_detect)
		//			updateTarget(t, t_box.x, t_box.y, t_box.width, t_box.height, t_box.x-t.x, t_box.y-t.y, dist);
		//		box_new = false;
		//	}
		//}
		for (auto j = 0; j < tracker_num; ++j) {
			MyTracker& tracker = trackers[j];
			std::shared_ptr<target>& t = tracker.pTarget;
			int dist= abs(t_box.x - t->x) + abs(t_box.y - t->y);
			if (tracker.state != sleeping && dist < max_diff && dist < t->dist) {
				if (tracker.update_by_detect) {
					updateTarget(t, t_box.x, t_box.y, t_box.width, t_box.height, t_box.x - t->x, t_box.y - t->y, dist);
					tracker.init(t_box, curFrame, tracker.trackerId, tracker.targetId);
				}
				box_new = false;
			}
		}
		if (box_new) {
			addTarget(t_box.x, t_box.y, t_box.width, t_box.height, 0,0,0);
			bool recycle = false;
			for (auto k = 0; k < tracker_num; ++k) {
				if (trackers[k].state == sleeping) {
					recycle = true;
					trackers[k].init(t_box, curFrame, k, targets.size() - 1);
					trackers[k].pTarget = targets[targets.size() - 1];
					break;
				}
			}
			if (!recycle) {
				if (tracker_num < MAX_N) {
					trackers[tracker_num].init(t_box, curFrame, tracker_num, targets.size()-1);
					trackers[tracker_num].pTarget = targets[targets.size() - 1];
					++tracker_num;
				}
			}
		}
	}

	for (auto i = 0; i < targets.size(); ++i)
		targets[i]->dist = 1e5;
	std::cout << "target size: " << targets.size() << "; tracker num: "<<tracker_num << std::endl;
}

void Tracker::track(cv::Mat& curFrame) {
	if (targets.size() == 0)
		return;
	if (cur_index != 0) {
		for (auto i = 0; i < tracker_num; ++i) {
			MyTracker& t = trackers[i];
			t.update(curFrame);
			float psr = t.calcPSR();
			// std::cout << "target " << t.targetId << " psr: " << psr << std::endl;
			if (psr < 0.6) {
				t.puzzleFrames++;
				t.update_by_detect = true;
			}
			if (t.puzzleFrames == 10) {
				t.forceUpdate();
			}
			if (t.puzzleFrames == 20) {
				t.recycle();
			}
		}
	}
	else
		initTrackers(curFrame);
	++cur_index;
}

void Tracker::drawBoundingBox(cv::Mat & curFrame, int scale)
{
	bool use_tracking = false;
	if (use_tracking) {
		for (auto i = 0; i < tracker_num; ++i) {
			MyTracker& t = trackers[i];
			if (t.pTarget->life == 0)
				continue;
			cv::Rect roi(t.pTarget->x*scale, t.pTarget->y*scale, t.pTarget->width*scale, t.pTarget->height*scale);
			cv::rectangle(curFrame, roi, cv::Scalar(0, 255, 255), 2);
		}
	}
	else {
		for (auto i = 0; i < targets.size(); ++i) {
			std::shared_ptr<target> t = targets[i];
			if (t->life == 0)
				continue;
			cv::Rect roi(t->x*scale, t->y*scale, t->width*scale, t->height*scale);
			cv::rectangle(curFrame, roi, cv::Scalar(0, 255, 255), 2);

		}
	}
}

void Tracker::initTrackers(cv::Mat& image)
{
	for (auto i = 0; i < targets.size(); ++i) {
		std::shared_ptr<target> t = targets[i];
		cv::Rect r(t->x, t->y, t->width, t->height);
		trackers[i].init(r, image, i, i);
		trackers[i].pTarget = t;
	}
	tracker_num = targets.size();
}
