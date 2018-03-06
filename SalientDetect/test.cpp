#include "stdafx.h"
#include "test.h"
#include <chrono>

void testDetector() {
	cv::Mat image = cv::imread("I:/Experiment/dataset/pafiss_eval_dataset/sequence04/2043_000001.jpeg");
	Salient salient;
	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat salMap = salient.salientDetectFT(image);
	cv::Mat biMap = salient.adaptBinarize(salMap);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));

	cv::erode(biMap, biMap, kernel);
	cv::imshow("bi", biMap);
	auto now = std::chrono::high_resolution_clock::now();
	double duration_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
	double seconds = duration_ns / 1e9;
	double fps = 1.0 / seconds;
	std::cout << "time: " << seconds << std::endl;
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

//void testCvKCF() {
//	int scale = 1;
//	int frameIndex = 0;
//	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/pafiss_eval_dataset/sequence04_files.txt");
//	cv::Ptr<cv::TrackerKCF> kcfTracker = cv::TrackerKCF::create();
//	cv::Rect2d roi;
//	for (auto i = 0; i < file_paths.size(); ++i) {
//		cv::Mat image = cv::imread(file_paths[i]);
//		cv::Mat scaled;
//		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
//		if (frameIndex == 0) {
//			roi=cv::selectROI(scaled);
//			kcfTracker->init(scaled, roi);
//		}
//		else {
//			kcfTracker->update(scaled, roi);
//			cv::rectangle(scaled, roi, cv::Scalar(0, 255, 255), 2);
//		}
//		cv::imshow("frame", scaled);
//		cv::waitKey(30);
//		++frameIndex;
//	}
//}

void testKCF() {
	KCFTracker kcfTracker;
	int scale = 1;
	int frameIndex = 0;
	std::vector<std::string> file_paths = loadPathFromFile("I:/Experiment/dataset/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps/car6_files.txt");
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
		cv::waitKey(5);
		++frameIndex;
	}
}

void testTracker() {
	int scale = 2;
	SalientTracker tracker;
	std::string path = "I:/Experiment/dataset/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps/car6_files.txt";
	std::vector<std::string> file_paths = loadPathFromFile(path.c_str());
	int cnt = 0;
	auto start = std::chrono::high_resolution_clock::now();
	char key = 0;
	int frameIndex = 0;
	for (auto i = 0; i < file_paths.size(); ++i) {
		cv::Mat image = cv::imread(file_paths[i]);
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		//if(frameIndex==0)
		tracker.detect_filter(scaled);
		tracker.track(scaled);
		tracker.nms();
		tracker.drawBoundingBox(image, scale);
		auto now = std::chrono::high_resolution_clock::now();
		double duration_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
		double seconds = duration_ns / 1e9;
		double fps = 1.0*(++cnt) / seconds;
		std::stringstream ss;
		ss << "fps: " << fps;
		cv::putText(image, ss.str(), cv::Point(image.cols - 200, 20), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0,255,255),2);
		cv::imshow("frame", image);
		key=cv::waitKey(1);
		if (key == 27)
			break;
		++frameIndex;
	}
}