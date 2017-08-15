/*
 * util.h
 *
 *  Created on: 2017Äê8ÔÂ10ÈÕ
 *      Author: JYF
 */

#ifndef SRC_UTIL_H_
#define SRC_UTIL_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

struct HogParam{
	int blockSize;
	int cellSize;
	int stride;
	cv::Size winSize;
};

void readGroundTruth(string gt_file, vector<Rect>& roi);

void readFileName(string file_list, vector<string>& file_names);

void readParamsFromFile(string param_file, vector<HogParam>& params);

void printParams(vector<HogParam>& params);

void padROI(Rect& roi, const Size& bounding, float alpha=1.15);

void alignSize(const Size& bbox, Size& size);

void saveOPE(string& ope_file, vector<Rect>& run_rects);
#endif /* SRC_UTIL_H_ */
