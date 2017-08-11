/*
 * main.cpp
 *
 *  Created on: 2017Äê8ÔÂ10ÈÕ
 *      Author: JYF
 */

#include "util.h"
#include "features.h"
#include "tracker.h"
#include <time.h>


double max_area;
double min_area;
vector<HogParam> params;

string datasets[]={
		"Jogging",
		"Panda",
		"BlurCar2",
		"Dog"
};

int main(int argc, char** argv){
	srand(time(NULL));

	string dataset=datasets[3];
	string benchmark_root="I:/TestOpenCV/Images/benchmark/";
	string gt_file=benchmark_root+dataset+"/groundtruth_rect.txt";
	string file_list=benchmark_root+dataset+"/image_file.txt";
	string param_file="./params.txt";

	readParamsFromFile(param_file, params);
	//printParams(params);

	//showFrame(file_list, gt_file);
	runTracker(file_list, gt_file,true);

	return 0;
}

