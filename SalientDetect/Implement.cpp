#include "stdafx.h"
#include "Implement.h"

#define PATCH_DETECT	1

Implement::Implement()
{
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	patchKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
}


Implement::~Implement()
{
}

cv::Mat Implement::getPatch(cv::Mat& im, cv::Rect& rect) {
	cv::Rect roi(rect.x-window_times/2*rect.width, rect.y- window_times / 2 *rect.height, window_times*rect.width, window_times*rect.height);
	roi.x = max(0, roi.x);
	roi.y = max(0, roi.y);
	roi.width = min(roi.width, im.cols - roi.x);
	roi.height = min(roi.height, im.rows - roi.y);
	rect = roi;
	return im(roi).clone();
}

std::vector<cv::Rect> Implement::parseCandidatePatches(cv::Mat& im) {
	cv::Mat baseSalMap = salient.salientDetectFT(im);
	cv::Mat baseBiMap = salient.adaptBinarize(baseSalMap);
	cv::erode(baseBiMap, baseBiMap, kernel);
	std::vector<cv::Rect> boxes = salient.findBoundingBoxes(baseBiMap.clone());
	std::vector<cv::Rect> finals;

	cv::imshow("basebi", baseBiMap);
	cv::imshow("basesal", baseSalMap);
	// cv::waitKey();
	for (auto i = 0; i < boxes.size(); ++i) {
		cv::Rect _box = boxes[i];
#if PATCH_DETECT==1
		cv::Mat patch = getPatch(im, boxes[i]);
		if (patch.rows < 10 || patch.cols < 10)
			continue;
		cv::Mat biMap = salient.salientDetectFT(patch);
		biMap = salient.adaptBinarize(biMap);
		// cv::erode(biMap, biMap, patchKernel);
		std::vector<cv::Rect> patchBoxes= salient.findBoundingBoxes(biMap.clone());
		if (patchBoxes.size() == 0) {
			finals.push_back(_box);
			continue;
		}
		cv::Rect nearBox;
		cv::Point patchCenter{ patch.cols / 2,patch.rows / 2 };
		int distCenter = 1e5;
		bool ok = false;
		for (auto j = 0; j < patchBoxes.size(); ++j) {
			cv::Rect& pbox = patchBoxes[j];
			cv::Point boxCenter(pbox.x + pbox.width / 2, pbox.y + pbox.height / 2);
			int dist = abs(patchCenter.x - boxCenter.x) + abs(patchCenter.y - boxCenter.y);
			if (dist<distCenter) {
				distCenter = dist;
				nearBox = pbox;
				nearBox.x += boxes[i].x;
				nearBox.y += boxes[i].y;
				ok = true;
			}
		}
		if (ok) {
			std::vector<cv::Rect> res = { _box, nearBox };
			nms(res);
			finals.push_back(res[0]);
			//finals.push_back(nearBox);
		}
#else
		finals.push_back(_box);
#endif
	}
	nms(finals);
	return finals;
}

cv::Mat Implement::findPeakRectArea(cv::Mat& im) {
	int width = im.cols;
	int height = im.rows;
	int shift_w = width - block_w;
	int shift_h = height - block_h;
	cv::Mat toLeft(im.size(), CV_8UC1);
	cv::Mat toRight(im.size(), CV_8UC1);
	cv::Mat toUp(im.size(), CV_8UC1);
	cv::Mat toDown(im.size(), CV_8UC1);
	toLeft.setTo(0);
	toRight.setTo(0);
	toUp.setTo(0);
	toDown.setTo(0);
	im(cv::Rect(block_w, 0, shift_w, height)).copyTo(toLeft(cv::Rect(0,0,shift_w,height)));
	im(cv::Rect(0, 0, shift_w, height)).copyTo(toRight(cv::Rect(block_w, 0, shift_w, height)));
	im(cv::Rect(0, block_h, width, shift_h)).copyTo(toUp(cv::Rect(0, 0, width, shift_h))); 
	im(cv::Rect(0, 0, width, shift_h)).copyTo(toDown(cv::Rect(0, block_h, width, shift_h)));
	cv::bitwise_and(toLeft, im, toLeft);
	cv::bitwise_and(toRight, toLeft, toRight);
	cv::bitwise_and(toUp, toRight, toUp);
	cv::bitwise_and(toDown, toUp, toDown);

	return toDown;
}

void Implement::nms(std::vector<cv::Rect>& boxes) {
	std::vector<bool> toDelete(boxes.size(), false);
	for (auto i = 0; i < boxes.size(); ++i) {
		if (toDelete[i])
			continue;
		cv::Rect& t_box = boxes[i];
		auto t_area = t_box.width*t_box.height;
		for (auto j = i+1; j < boxes.size(); ++j) {
			cv::Rect& q_box = boxes[j];
			auto q_area = q_box.width*q_box.height;
			auto lx = max(t_box.x, q_box.x);
			auto ly = max(t_box.y, q_box.y);
			auto rx = min(t_box.x + t_box.width, q_box.x + q_box.width);
			auto ry = min(t_box.y + t_box.height, q_box.y + q_box.height);
			auto overlap = max(0, rx - lx)*max(0, ry - ly);
			auto t_rate = 1.0*overlap / t_area;
			auto q_rate = 1.0*overlap / q_area;
			if (t_rate > 0.25) {
				if (t_area <= q_area) {
					toDelete[i] = true;
				}
			}
			if (q_rate > 0.25) {
				if (q_area < t_area && toDelete[i]==false){
					toDelete[j] = true;
					//std::cout << "box " << i << " nms" << std::endl;
				}
			}
		}
	}
	std::vector<cv::Rect> newBoxes;
	for (auto i = 0; i < boxes.size(); ++i) {
		if (!toDelete[i]) {
			newBoxes.push_back(boxes[i]);
			//std::cout << boxes[i].width << "," << boxes[i].height << std::endl;
		}
	}
	std::vector<cv::Rect>().swap(boxes);
	boxes = newBoxes;
}

void Implement::merge(std::vector<cv::Rect>& boxes) {
	//NOT IMPLEMENTED
}