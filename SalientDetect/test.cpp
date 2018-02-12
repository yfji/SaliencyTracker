#include "stdafx.h"
#include "test.h"

void testDetector(cv::Mat& image) {
	Salient salient;
	cv::Mat salMap = salient.salientDetectFTFull(image);
	cv::Mat biMap = salient.binarize(salMap);
	cv::Mat biMapNot;
	cv::bitwise_not(biMap, biMapNot);
	std::vector<cv::Rect> boxes = salient.findBoundingBoxes(biMap.clone());
	cv::Mat image_display = image.clone();
	for (auto i = 0; i < boxes.size(); ++i) {
		cv::rectangle(image_display, boxes[i], cv::Scalar(0, 255, 255), 2);
	}
	cv::imshow("bbox", image_display);
	cv::waitKey();
}

void testVideo() {
	int scale = 2;
	Implement impl;
	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/pafiss_eval_dataset/sequence02_files.txt");
	for (auto i = 0; i < file_paths.size(); ++i) {
		cv::Mat image = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		std::vector<cv::Rect> boxes = impl.parseCandidatePatches(scaled);
		for (auto j = 0; j < boxes.size(); ++j) {
			cv::Rect& box = boxes[j];
			box.x *= scale;
			box.y *= scale;
			box.width *= scale;
			box.height *= scale;
			cv::rectangle(image, box, cv::Scalar(0, 255, 255), 2);
		}
		cv::imshow("sa", image);
		cv::waitKey(1);
	}
}

void testCvKCF() {
	int scale = 1;
	int frameIndex = 0;
	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/pafiss_eval_dataset/sequence04_files.txt");
	cv::Ptr<cv::TrackerKCF> kcfTracker = cv::TrackerKCF::create();
	cv::Rect2d roi;
	for (auto i = 0; i < file_paths.size(); ++i) {
		cv::Mat image = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		if (frameIndex == 0) {
			roi=cv::selectROI(scaled);
			kcfTracker->init(scaled, roi);
		}
		else {
			kcfTracker->update(scaled, roi);
			cv::rectangle(scaled, roi, cv::Scalar(0, 255, 255), 2);
		}
		cv::imshow("frame", scaled);
		cv::waitKey(30);
		++frameIndex;
	}
}

void testKCF() {
	KCFTracker kcfTracker;
	int scale = 1;
	int frameIndex = 0;
	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/pafiss_eval_dataset/sequence03_files.txt");
	cv::Rect2d roi;
	for (auto i = 0; i < file_paths.size(); ++i) {
		cv::Mat image = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		if (frameIndex == 0) {
			roi = cv::selectROI(scaled);
			kcfTracker.init(roi, scaled);
		}
		else {
			roi = kcfTracker.update(scaled);
			cv::rectangle(scaled, roi, cv::Scalar(0, 255, 255), 2);
		}
		float psr = (kcfTracker.peak_value) / (kcfTracker.sigma);
		std::cout << "psr: " << psr << std::endl;
		cv::imshow("frame", scaled);
		cv::waitKey(30);
		++frameIndex;
	}
}

void testTracker() {
	int scale = 2;
	Tracker tracker;
	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/pafiss_eval_dataset/sequence03_files.txt");
	for (auto i = 0; i < file_paths.size(); ++i) {
		cv::Mat image = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		tracker.detect_filter(scaled);
		tracker.track(scaled);
		tracker.nms();
		tracker.drawBoundingBox(image, scale);
		cv::imshow("frame", image);
		cv::waitKey();
	}
}