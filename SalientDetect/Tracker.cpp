#include "stdafx.h"
#include "Tracker.h"
#include <sstream>

Tracker::Tracker()
{
	tracker_num = 0;
	target_num = 0;
	tracker_id = -1;
}


Tracker::~Tracker()
{
}

int Tracker::addTarget(int x, int y, int w, int h)
{
	for (auto i = 0; i < targets.size(); ++i) {
		if (targets[i]->b_new == false && targets[i]->life == 0) {
			updateTarget(targets[i], x, y, w, h, 0, 0, 1e4);
			targets[i]->b_new = true;
			targets[i]->life = 1;
			targets[i]->uuid = target_num++;
			return i;
		}
	}
	if (tracker_id+1 < MAX_N)
	{
		std::shared_ptr<target> t = std::make_shared<target>();
		t->x = x;
		t->y = y;
		t->width = w;
		t->height = h;
		t->uuid = target_num++;
		targets.push_back(t);
		return targets.size() - 1;
	}
	return MAX_N-1;
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
			addTarget(bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height);
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
		for (auto j = 0; j <= tracker_id; ++j) {
			MyTracker& tracker = trackers[j];
			std::shared_ptr<target>& t = tracker.pTarget;
			// int dist= abs(t_box.x - t->x) + abs(t_box.y - t->y);
			int dist_left = (int)sqrt((t_box.x - t->x)*(t_box.x - t->x) + (t_box.y - t->y)*(t_box.y - t->y));
			int dist_right = (int)sqrt((t_box.x + t_box.width - t->x - t->width)*(t_box.x + t_box.width - t->x - t->width) + (t_box.y + t_box.height - t->y - t->height)*(t_box.y + t_box.height - t->y - t->height));
			if (tracker.state != sleeping && (dist_left < max_diff && dist_right<max_diff)&& max(dist_left,dist_right) < t->dist) {
				if (tracker.update_by_detect) {
					updateTarget(t, t_box.x, t_box.y, t_box.width, t_box.height, t_box.x - t->x, t_box.y - t->y, max(dist_left, dist_right));
					tracker.init(t_box, curFrame, tracker.trackerId, tracker.targetId);
				}
				box_new = false;
			}
		}
		if (box_new) {
			int t_id=addTarget(t_box.x, t_box.y, t_box.width, t_box.height);
			bool recycle = false;
			for (auto k = 0; k <= tracker_id; ++k) {
				if (trackers[k].state == sleeping) {
					recycle = true;
					trackers[k].init(t_box, curFrame, k, target_num-1);
					trackers[k].pTarget = targets[t_id];
					break;
				}
			}
			if (!recycle) {
				if (tracker_id+1 < MAX_N) {
					trackers[tracker_id+1].init(t_box, curFrame, tracker_num, target_num-1);
					trackers[tracker_id+1].pTarget = targets[t_id];
					++tracker_num;
					++tracker_id;
				}
			}
		}
	}
	for (auto i = 0; i < targets.size(); ++i)
		targets[i]->dist = 1e4;
	std::cout << "target size: " << targets.size() << "; tracker num: "<<tracker_num << std::endl;
}

void Tracker::trackThreadRef(MyTracker& t, cv::Mat& im)
{
	t.update(im, psr_thres);
	float psr = t.calcPSR();
	std::cout << "tracker " << t.trackerId << " psr: " << psr << std::endl;
	if (psr < psr_thres) {
		t.puzzleFrames++;
		t.update_by_detect = true;
	}
	if (t.puzzleFrames == nForceUpdate) {
		t.forceUpdate();
	}
	if (t.puzzleFrames == nRecycle) {
		t.recycle();
		tracker_num = max(0, tracker_num - 1);
	}
}

void Tracker::trackThreadPtr(MyTracker * t, cv::Mat * im)
{
	t->update(*im, psr_thres);
	float psr = t->calcPSR();
	//std::cout << "tracker " << t->trackerId << " psr: " << psr << std::endl;
	if (psr < psr_thres) {
		t->puzzleFrames++;
	}
	else {
		t->puzzleFrames = max(0, t->puzzleFrames - 1);
	}
	if (t->puzzleFrames>0 && t->puzzleFrames % nForceUpdate==0) {
		//t->update_by_detect = true;
		t->forceUpdate();
	}
	if (t->puzzleFrames == nRecycle) {
		t->recycle();
		tracker_num = max(0, tracker_num - 1);
	}
}

#define MULTI_THREAD	1
void Tracker::track(cv::Mat& curFrame) {
	if (targets.size() == 0)
		return;
	if (cur_index != 0) {
		for (auto i = 0; i <= tracker_id; ++i) {
			MyTracker& t = trackers[i];
			if (t.state == sleeping)
				continue;
#if MULTI_THREAD==0
			trackThreadRef(t, curFrame);
#else
			mThreads[i]=std::thread (&Tracker::trackThreadPtr, this, &t, &curFrame);
#endif
		}
#if MULTI_THREAD==1
		for (auto i = 0; i <= tracker_id; ++i) {
			if (mThreads[i].joinable()) {
				mThreads[i].join();
			}
		}
#endif
	}
	else
		initTrackers(curFrame);
	++cur_index;
}

void Tracker::drawBoundingBox(cv::Mat & curFrame, int scale)
{
	bool use_tracking = true;
	if (use_tracking) {
		for (auto i = 0; i <= tracker_id; ++i) {
			MyTracker& t = trackers[i];
			if (t.pTarget->life == 0 || t.state==sleeping)
				continue;
			cv::Rect roi(t.pTarget->x*scale, t.pTarget->y*scale, t.pTarget->width*scale, t.pTarget->height*scale);
			cv::rectangle(curFrame, roi, colors[t.trackerId], 2);
			std::stringstream ss;
			ss << "tracker " << i<<" : "<<t.psr;
			cv::putText(curFrame, ss.str(), cv::Point(roi.x - 15, roi.y), cv::FONT_HERSHEY_PLAIN, 0.9, colors[t.trackerId]);
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
	for (auto i = 0; i < min(targets.size(),MAX_N); ++i) {
		std::shared_ptr<target> t = targets[i];
		cv::Rect r(t->x, t->y, t->width, t->height);
		trackers[i].init(r, image, i, i);
		trackers[i].pTarget = t;
	}
	tracker_num = min(targets.size(), MAX_N);
	tracker_id = tracker_num - 1;
	std::cout << "init tracker num: " << tracker_num << std::endl;
}

void Tracker::nms()
{
	std::vector<bool> toDelete(tracker_id+1, false);
	for (auto i = 0; i <= tracker_id; ++i) {
		if (toDelete[i] || trackers[i].state==sleeping)
			continue;
		std::shared_ptr<target>& t_target = trackers[i].pTarget;
		auto t_area = t_target->width*t_target->height;
		for (auto j = i + 1; j <= tracker_id; ++j) {
			if (toDelete[j] || trackers[j].state == sleeping)
				continue;
			std::shared_ptr<target>& q_target = trackers[j].pTarget;
			auto q_area = q_target->width*q_target->height;
			auto lx = max(t_target->x, q_target->x);
			auto ly = max(t_target->y, q_target->y);
			auto rx = min(t_target->x + t_target->width, q_target->x + q_target->width);
			auto ry = min(t_target->y + t_target->height, q_target->y + q_target->height);
			auto overlap = max(0, rx - lx)*max(0, ry - ly);
			auto t_rate = 1.0*overlap / t_area;
			auto q_rate = 1.0*overlap / q_area;
			float nms_thres_low = 0.5;
			float nms_thres_high = 0.9;
			if (t_rate > nms_thres_low) {
				//if (t_rate < nms_thres_high) {
					if (t_area <= q_area && (trackers[i].life == 1)) {
						toDelete[i] = true;
					}
				//}
				//else {
				//	if (t_area <= q_area) {
				//		toDelete[i] = true;
				//	}
				//}
			}
			if (q_rate > nms_thres_low) {
				//if (q_rate < nms_thres_high) {
					if (q_area < t_area && toDelete[i] == false && (trackers[j].life == 1)) {
						toDelete[j] = true;
					}
				//}
				//else {
				//	if (q_area < t_area && toDelete[i] == false) {
				//		toDelete[j] = true;
				//	}
				//}
			}
		}
	}
	for (auto i = 0; i <= tracker_id; ++i) {
		if (toDelete[i]) {
			trackers[i].recycle();
			tracker_num = max(0, tracker_num - 1);
		}
	}
}

void Tracker::detectNms()
{
}
