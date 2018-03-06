#include "stdafx.h"
#include "Salient.h"

#define CUT_BORDER	1

Salient::Salient()
{
	ptrExtractor = std::make_shared<FeatureExtractorSalient>();
	nn = std::make_shared<ANN>();
	nn->loadParams("./model/param_hog_iter_9000.model");
	nn->setFeatureExtractorPtr(ptrExtractor);
}


Salient::~Salient()
{
}

void Salient::RGBToLab(unsigned char * rgbImg, float * labImg) {
	float B = gamma(rgbImg[0] / 255.0f);
	float G = gamma(rgbImg[1] / 255.0f);
	float R = gamma(rgbImg[2] / 255.0f);
	float X = 0.412453*R + 0.357580*G + 0.180423*B;
	float Y = 0.212671*R + 0.715160*G + 0.072169*B;
	float Z = 0.019334*R + 0.119193*G + 0.950227*B;

	X /= 0.95047;
	Y /= 1.0;
	Z /= 1.08883;

	float FX = X > 0.008856f ? pow(X, 1.0f / 3.0f) : (7.787f * X + 0.137931f);
	float FY = Y > 0.008856f ? pow(Y, 1.0f / 3.0f) : (7.787f * Y + 0.137931f);
	float FZ = Z > 0.008856f ? pow(Z, 1.0f / 3.0f) : (7.787f * Z + 0.137931f);
	labImg[0] = Y > 0.008856f ? (116.0f * FY - 16.0f) : (903.3f * Y);
	labImg[1] = 500.f * (FX - FY);
	labImg[2] = 200.f * (FY - FZ);
}

cv::Mat Salient::salientDetectSR(cv::Mat& im) {
	cv::Mat salMap;
	salientSR.computeSaliency(im, salMap);
	return salMap;
}
cv::Mat Salient::salientDetectFG(cv::Mat& im) {
	cv::Mat salMap;
	salientFG.computeSaliency(im, salMap);
	return salMap;
}
cv::Mat Salient::salientDetectMotion(cv::Mat& im) {
	cv::Mat salMap;
	salientMotion.computeSaliency(im, salMap);
	return salMap;
}

cv::Mat Salient::salientDetectFT(cv::Mat& im) {
	assert(src.channels() == 3);
	cv::Mat salMap(im.size(), CV_32FC1);
	cv::Mat lab, labf;
	int h = im.rows, w = im.cols;
	labf.create(cv::Size(w, h), CV_32FC3);
	uchar* fSrc = im.data;
	float* fLab = (float*)labf.data;
	float* fDst = (float*)salMap.data;

	int stride = w * 3;
	//for (int i = 0; i < h; ++i) {
	//	for (int j = 0; j < stride; j += 3) {
	//		RGBToLab(fSrc + i*stride + j, fLab + i*stride + j);
	//	}
	//}
	float MeanL = 0, MeanA = 0, MeanB = 0;
	for (int i = 0; i < h; ++i) {
		int index = i*stride;
		for (int x = 0; x < w; ++x) {
			RGBToLab(fSrc + index, fLab + index);
			MeanL += fLab[index];
			MeanA += fLab[index + 1];
			MeanB += fLab[index + 2];
			index += 3;
		}
	}
	MeanL /= (w * h);
	MeanA /= (w * h);
	MeanB /= (w * h);
	cv::GaussianBlur(labf, labf, cv::Size(5, 5), 1);
	for (int Y = 0; Y < h; Y++)
	{
		int Index = Y * stride;
		int CurIndex = Y * w;
		for (int X = 0; X < w; X++)
		{
			fDst[CurIndex++] = (MeanL - fLab[Index]) *  \
				(MeanL - fLab[Index]) + (MeanA - fLab[Index + 1]) *  \
				(MeanA - fLab[Index + 1]) + (MeanB - fLab[Index + 2]) *  \
				(MeanB - fLab[Index + 2]);
			Index += 3;
		}
	}
	cv::normalize(salMap, salMap, 0, 1, cv::NORM_MINMAX);
	salMap.convertTo(salMap, CV_8UC1, 255);
	return salMap;
}

cv::Mat Salient::salientDetectFTFull(cv::Mat& im) {
	int width = im.cols;
	int height = im.rows;
	int channels = im.channels();
	int sz = width*height;
	cv::Mat baseSalMap = salientDetectFT(im);
	imshow("base", baseSalMap);
	cv::Mat rgbSeg;
	int* pLabels;
	int numLabels;
	meansShiftSegmentation(im, rgbSeg, pLabels, numLabels);
	imshow("seg", rgbSeg);
	uchar* salMapBuffer = (uchar*)baseSalMap.data;
	std::vector<float> valPerSeg(numLabels, 0.0);
	std::vector<int> histPerSeg(numLabels, 0);
	std::vector<bool> touchBorders(numLabels, false);
	float valBaseSeg = 0;
	int i = 0;
	salMapValidPix = 0;
	for (auto j = 0; j < height; ++j) {
		for (auto k = 0; k < width; ++k) {
			valPerSeg[pLabels[i]] += (float)((int)salMapBuffer[i]);
			histPerSeg[pLabels[i]]++;
#if CUT_BORDER==1
			if (false == touchBorders[pLabels[i]] && (j == height - 1 || j == 0 || k == width - 1 || k == 0))
			{
				touchBorders[pLabels[i]] = true;
			}
			else
#endif
				salMapValidPix++;
			++i;
		}
	}
	
	for (int n = 0; n < numLabels; n++)
	{
#if CUT_BORDER==1
		if (true == touchBorders[n])
		{
			valPerSeg[n] = 0;
		}
		else
		{
#endif
			valBaseSeg += valPerSeg[n];
			valPerSeg[n] /= histPerSeg[n];
#if CUT_BORDER==1
		}
#endif
	}
	valBaseSeg /= salMapValidPix;
	std::vector<bool> segtochoose(numLabels, false);
	float thres = times*valBaseSeg;
	for (int n = 0; n < numLabels; n++)
	{
		if (valPerSeg[n] > thres) 
			segtochoose[n] = true;
	}
	cv::Mat segBinary(height, width, CV_8UC1);
	uchar* biBuffer = (uchar*)segBinary.data;
	for (i = 0; i < sz; ++i) {
		if (segtochoose[pLabels[i]])
			biBuffer[i] = 255;
		else
			biBuffer[i] = 0;
	}
	delete pLabels;
	imshow("segBi", segBinary);
	return segBinary;
}

void Salient::meansShiftSegmentation(cv::Mat& im, cv::Mat& segMat, int*& labels, int& num_labels) {
	int height = im.rows;
	int width = im.cols;
	int sz = height*width;
	int channels = im.channels();
	imageType type = (channels == 1?GRAYSCALE : COLOR);
	cv::Mat im_copy = im.clone();
	byte* bytebuff = (uchar*)im_copy.data;
	mss.DefineImage(bytebuff, type, height, width);
	mss.Segment(sigmaS, sigmaR, minRegion, HIGH_SPEEDUP);
	mss.GetResults(bytebuff);
	labels = new int[sz];
	int numlabels = mss.GetLabels(labels);

	int matType = (channels == 1 ? CV_8UC1 : CV_8UC3);
	segMat=cv::Mat(height, width, matType);
	byte* segByteBuff = (uchar*)segMat.data;
	
	for (auto i = 0; i < sz*channels; ++i) {
		segByteBuff[i] = bytebuff[i];
	}
}

std::vector<cv::Rect> Salient::findBoundingBoxes(const cv::Mat& im){
	std::vector<cv::Rect> boxes;
	IplImage ipl = im;
	CvMemStorage* pStorage = cvCreateMemStorage(0);
	CvSeq* pContour = NULL;
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	bool updated = false;
	max_area = 4e4;
	min_area = 300;
	for (; pContour; pContour = pContour->h_next) {
		float true_area = fabs(cvContourArea(pContour));
		cv::Rect bbox = cvBoundingRect(pContour, 0);
		float box_area = 1.0*bbox.height*bbox.width;
		if (bbox.width > 2*bbox.height || bbox.height > 2.5*bbox.width) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (bbox.width > im.cols / 2 || bbox.height > im.rows / 2) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (box_area > max_area || box_area < min_area) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		if (box_area / true_area > 4.1) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		int pred;
		float prob;
		int pad = 3;
		cv::Rect detbox = cv::Rect(max(0, bbox.x - pad), max(0, bbox.y - pad), bbox.width + 2*pad, bbox.height + 2*pad);
		detbox.width = min(im.cols - detbox.x, detbox.width);
		detbox.height = min(im.rows - detbox.y, detbox.height);
		nn->predict(im(detbox), pred, prob);
		if (pred != 0 && pred != 1) {
			cvSeqRemove(pContour, 0);
			continue;
		}
		//std::cout << "ratio: " << box_area / true_area << std::endl;
		boxes.push_back(bbox);
	}
	cvReleaseMemStorage(&pStorage);
	return boxes;
}

cv::Mat Salient::adaptBinarize(cv::Mat& im) {
	assert(im.channels() == 1);
	const int blockSize = 13;
	const int threshold = 12;
	const int sz = blockSize*blockSize;
	int halfSize = blockSize / 2;
	cv::Rect roi(halfSize, halfSize, im.cols, im.rows);
	cv::copyMakeBorder(im, im, halfSize, halfSize, halfSize, halfSize, cv::BORDER_CONSTANT, 0);

	cv::Mat iimage, biMap;
	cv::integral(im, iimage, CV_32S);
	biMap.create(im.size(), CV_8UC1);
	for (int j = halfSize; j < im.rows - halfSize - 1; ++j) {
		uchar* data = biMap.ptr(j);
		uchar* im_data = im.ptr(j);
		int* idata1 = iimage.ptr<int>(j - halfSize);
		int* idata2 = iimage.ptr<int>(j + halfSize + 1);
		for (int i = halfSize; i < im.cols - halfSize - 1; ++i) {
			int sum = (idata2[i + halfSize + 1] - idata2[i - halfSize] - idata1[i + halfSize + 1] + idata1[i - halfSize]);
			sum /= sz;
			if (im_data[i] < sum-threshold)
				data[i] = 0;
			else
				data[i] = 255;
		}
	}
	cv::Mat biMapNot;
	cv::bitwise_not(biMap(roi), biMapNot);
	return biMapNot;
	//return biMap(roi).clone();
}