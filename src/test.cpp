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

	//cv::erode(biMap, biMap, kernel);
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

void testKCF(const char* file_path) {
	KCFTracker kcfTracker;
	int scale = 2;
	int frameIndex = 0;
	std::vector<std::string> file_paths = loadPathFromFile(file_path);
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
		cv::waitKey(1);
		++frameIndex;
	}
}


void testTracker(const char* file_path) {
	int scale = 2;
	SalientTracker tracker;
	//std::string path = "I:/Experiment/dataset/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps/car10_files.txt";
	//std::string path = "/home/ubuntu/SalientDetect/experiment/pafiss/sequence02_files.txt";
	std::string path= std::string(file_path);
	std::vector<std::string> file_paths = loadPathFromFile(path.c_str());
	std::cout<<"Length: "<<file_paths.size()<<endl;
	int cnt = 0;
	auto start = std::chrono::high_resolution_clock::now();
	char key = 0;
	int frameIndex = 0;
	const char* result_file = "car1_result.txt";
	//const char* save_dir = "I:/Experiment/dataset/result/sequence02/";
	ofstream out;
	out.open(result_file, ios::out);
	int det_ind=0;
	const int det_track_interval=10;
	for (size_t i = 0; i < file_paths.size(); ++i) {
		std::string img_path=file_paths[i];//.replace(file_paths[i].find("\n"),1,"");
		cv::Mat image = cv::imread(img_path);
		//cout<<img_path<<endl;
		//cout<<image.rows<<","<<image.cols<<endl;
		cv::Mat scaled;
		cv::resize(image, scaled, cv::Size(image.cols / scale, image.rows / scale));
		//if(frameIndex==0)
		if(det_ind==0)
			tracker.detect_filter(scaled);
		det_ind=(det_ind+1)%det_track_interval;
		tracker.track(scaled);
		tracker.nms();
		tracker.drawBoundingBox(image, scale);
		tracker.saveResults(out, file_paths[i]);
		auto now = std::chrono::high_resolution_clock::now();
		double duration_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
		double seconds = duration_ns / 1e9;
		double fps = 1.0*(++cnt) / seconds;
		std::stringstream ss;
		ss << "fps: " << fps;
		cv::putText(image, ss.str(), cv::Point(image.cols - 250, 30), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255,0,0),2);
		cv::imshow("frame", image);
		//ss.str("");
		//ss << save_dir << "frame_" << frameIndex << ".jpg";
		//cv::imwrite(ss.str(), image);
		key=cv::waitKey(1);
		if (key == 27)
			break;
		++frameIndex;
	}
	out.close();
}
