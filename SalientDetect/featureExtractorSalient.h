#pragma once
#include "featureExtractor.h"
class FeatureExtractorSalient :
	public FeatureExtractor
{
public:
	FeatureExtractorSalient();
	~FeatureExtractorSalient();
	
private:
	void calcHog(cv::Mat& image, float* hogFeat);
	void calcHist(cv::Mat& image, float* histFeat);
public:
	float* extract(cv::Mat& image);
	void extract(cv::Mat& image, float* feat);

};

