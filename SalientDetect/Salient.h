#pragma once
#include <opencv2\saliency.hpp>
#include "ann.h"
#include "featureExtractorSalient.h"
#include "meanshift/msImageProcessor.h"

class Salient
{
public:
	Salient();
	virtual ~Salient();

	cv::Mat salientDetectSR(cv::Mat& im);
	cv::Mat salientDetectFG(cv::Mat& im);
	cv::Mat salientDetectMotion(cv::Mat& im);
	cv::Mat salientDetectFT(cv::Mat& im);
	cv::Mat salientDetectFTFull(cv::Mat& im);
	void meansShiftSegmentation(cv::Mat& im, cv::Mat& segMat, int*& labels, int& num_labels);
	inline cv::Mat binarize(cv::Mat& im) {
		cv::Mat biMap;
		cv::threshold(im, biMap, 50, 255, cv::THRESH_OTSU);
		return biMap;
	}
	cv::Mat adaptBinarize(cv::Mat& im);
	std::vector<cv::Rect> findBoundingBoxes(const cv::Mat& im);

private:
	const int sigmaS = 7;
	const float sigmaR = 10;
	const int minRegion = 20;
	const int times = 2;
	int salMapValidPix;
	int max_area;
	int min_area;
	msImageProcessor mss;
	cv::saliency::StaticSaliencySpectralResidual salientSR;
	cv::saliency::MotionSaliencyBinWangApr2014 salientMotion;
	cv::saliency::StaticSaliencyFineGrained salientFG;
	
	std::shared_ptr<FeatureExtractor> ptrExtractor;
	std::shared_ptr<ANN> nn;

private:
	inline float gamma(float x) {
		return x>0.04045 ? pow((x + 0.055f) / 1.055f, 2.4f) : x / 12.92;
	}
	
	void RGBToLab(unsigned char * rgbImg, float * labImg);
	
};

