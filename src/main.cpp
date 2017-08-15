/*
 * main.cpp
 *
 *  Created on: 2017��8��10��
 *      Author: JYF
 */

#include "util.h"
#include "features.h"
#include "tracker.h"
#include <time.h>


float max_area;
float min_area;
vector<HogParam> params;

string datasets[]={
		"Jogging",
		"Panda",
		"BlurCar2",
		"Dog",
		"Crossing",
		"Bolt",
		"CarScale",
		"Walking",
		"Walking2",
		"Skiing"
};

int main(int argc, char** argv){
	srand(time(NULL));

	string dataset="CarScale";
	string benchmark_root="I:/TestOpenCV/Images/benchmark/";
	string gt_file=benchmark_root+dataset+"/groundtruth_rect.txt";
	string file_list=benchmark_root+dataset+"/image_file.txt";
	string param_file="./params.txt";

	string ope_file=benchmark_root+dataset+"/ope.txt";
	readParamsFromFile(param_file, params);
	//printParams(params);

	//showFrame(file_list, gt_file);
	//runTracker(file_list, gt_file,true);
	runTrackerSimple(file_list, gt_file, ope_file);
	return 0;
}

