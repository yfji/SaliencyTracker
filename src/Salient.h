#pragma once
#include <opencv2/saliency.hpp>
#include "ann.h"
#include "featureExtractorSalient.h"

class Salient
{
public:
	Salient();
	virtual ~Salient();

	cv::Mat salientDetectSR(const cv::Mat& im);
	cv::Mat salientDetectFG(const cv::Mat& im);
	cv::Mat salientDetectMotion(const cv::Mat& im);
	cv::Mat salientDetectFT(const cv::Mat& im);
	cv::Mat salientDetectFTFull(const cv::Mat& im);
	inline cv::Mat binarize(const cv::Mat& im) {
		cv::Mat biMap;
		cv::threshold(im, biMap, 50, 255, cv::THRESH_OTSU);
		return biMap;
	}
	cv::Mat adaptBinarize(const cv::Mat& im);
	std::vector<cv::Rect> findBoundingBoxes(const cv::Mat& im);

private:
	const int sigmaS = 7;
	const float sigmaR = 10;
	const int minRegion = 20;
	const int times = 2;
	int salMapValidPix;
	int max_area;
	int min_area;

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

