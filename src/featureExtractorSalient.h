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
	void calcHistPatch(const uchar* pData, float* feat, int binSize, int offset, int h, int w);
public:
	float* extract(const cv::Mat& image);
	void extract(const cv::Mat& image, float* feat);

};

