/*
 * features.h
 *
 *  Created on: 2017Äê7ÔÂ23ÈÕ
 *      Author: JYF
 */

#ifndef SRC_FEATURES_H_
#define SRC_FEATURES_H_

#define SCALED_SIZE	128
#define CELL_SIZE	16
#define BLOCK_SIZE	2*CELL_SIZE
#define STRIDE	32

#define TRAIN_SVM	0
#define TRAIN_LOG	1
/**
 * Hough circle detect
 * saliency detect
 * ORB match
 * SVM using HOG
 * Adaboost using HOG
 * KCF
 */
#include "saliency.h"
#include "util.h"
#include <stdlib.h>

#define POS_CROP	2
#define NEG_CROP	6

typedef unsigned int uint;
typedef SVM MySVM;

void getHogFeature(Mat& image, HogParam& param, vector<float>& hogFeat, bool cvt);

void getMSHogFeature(Mat& image, vector<HogParam>& params, vector<float>& msHogFeat);

void featMaxBlur(Mat& src, Size ksize);

Mat detectSaliency(Mat& src);

void getSVM(MySVM& mSVM);

int detectSVM(Mat& src, vector<HogParam>& params, MySVM& svm);

void createNegativeSamples(Mat& frame, Rect& bbox, vector<Rect>& neg_rois, float overlap=0.1);

void getSamplesFromFrame(Mat& frame, vector<HogParam>& params, Rect& bbox, Mat& sampleMat, Mat& labelMat);

void getAllSamples(vector<string>& file_names, vector<Rect>& rois, vector<HogParam>& params, Mat& sampleMat, Mat& labelMat, int n=2);

void trainOrUpdateSVM(MySVM& svm, Mat& sampleMat, Mat& labelMat, vector<HogParam>& params, int iter=5e2);

#endif /* SRC_FEATURES_H_ */
