/*
 * tracker.cpp
 *
 *  Created on: 2017Äê8ÔÂ10ÈÕ
 *      Author: JYF
 */


#include "tracker.h"
#define FILTER_LEN 2*LIFE

void showFrame(string file_list, string gt_file){
	vector<Rect> rois;
	vector<string> file_names;

	readGroundTruth(gt_file, rois);
	readFileName(file_list, file_names);

	namedWindow("frame");

	assert(rois.size()==file_names.size());

	char key=0;
	for(int k=0;k<file_names.size();++k){
		Mat frameImg=imread(file_names[k]);
		Mat sampleMat, labelMat;
		vector<Rect> bbox;
		bbox.push_back(rois[k]);
		vector<Rect> neg_rois;

		createNegativeSamples(frameImg, bbox, neg_rois);

		for(int i=0;i<bbox.size();++i){	rectangle(frameImg, bbox[i], Scalar(0,255,255), 1);	}
		for(int i=0;i<neg_rois.size();++i){	rectangle(frameImg, neg_rois[i], Scalar(255,255,122), 1);	}

		imshow("frame", frameImg);
		key=waitKey(1);
		if(key==27){	break;	}
	}
}

void runTrackerSimple(string file_list, string gt_file){

}

void runTracker(string file_list, string gt_file, bool bOneFrameLag){
	vector<Rect> rois;
	vector<string> file_names;

	readGroundTruth(gt_file, rois);
	readFileName(file_list, file_names);

	namedWindow("frame");

	assert(rois.size()==file_names.size());

	char key=0;
	char bFirstFrame=1;
	char start=0;
	int center_x=0;
	int center_y=0;
	int filter_len=FILTER_LEN;
	int ksize=5;

	Mat frame;
	Mat lastFrame;
	Mat filteredFrame;

	vector<Target> targets;
	vector<Target> valid_targets;
	Rect ref_roi;

	MySVM svm;
	extern vector<HogParam> params;
	Rect search_roi;
	for(uint k=0;k<file_names.size();++k){
		Mat sampleMat, labelMat;
		vector<Rect> bbox;

		frame=imread(file_names[k]);
		if(k==0){
			ref_roi=rois[k];
			bbox.push_back(rois[k]);
		}
		else{
			if(targets.size()==0){
				cout<<"No target detected"<<endl;
				bbox.push_back(ref_roi);
			}
			else{
				cout<<"Target detected"<<endl;
				Rect loc=targets[0].location;
				//loc.x+=search_roi.x;
				//loc.y+=search_roi.y;
				bbox.push_back(loc);
			}
		}

		search_roi=bbox[0];
		search_roi.x=max(0,search_roi.x-search_roi.width/2);
		search_roi.y=max(0,search_roi.y-search_roi.height/2);
		search_roi.width=min(frame.cols-search_roi.x, (int)(search_roi.width*2));
		search_roi.height=min(frame.rows-search_roi.y, (int)(search_roi.height*2));

		cout<<frame.cols<<","<<frame.rows<<endl;
		cout<<search_roi.x<<","<<search_roi.y<<","<<search_roi.width<<","<<search_roi.height<<endl;
		Mat search_mat=frame(search_roi);

		getSamplesFromFrame(frame, params, bbox, sampleMat, labelMat);

		trainOrUpdateSVM(svm, sampleMat, labelMat, params);

		cout<<"SVM updated"<<endl;
		blur(frame,frame,Size(ksize,ksize));
		Mat dummy;
		frame.copyTo(dummy);
		if(bFirstFrame){
			bFirstFrame=0;
			center_x=frame.cols/2;
			center_y=frame.rows/2;
			filteredFrame.create(dummy.size(), CV_8UC1);
		}
		if(start){
			if(filter_len>0){
				detect_bbox(lastFrame, targets, svm, Size(search_roi.x,search_roi.y), false);
				if(valid_targets.size()>0){	draw_bbox(dummy, valid_targets, 1);	}
			}
			else{
				filter_len=FILTER_LEN+1;
				filteredFrame.setTo(0);
				detect_bbox(search_mat, targets, svm, Size(search_roi.x,search_roi.y),true);
				valid_targets=targets;
				draw_bbox(dummy, targets, 1);
			}
		}
		imshow("frame", dummy);
		key=waitKey(1);
		if(key==27){	break;	}
		search_mat.copyTo(lastFrame);
		if(start){	--filter_len;	}
		if(start==0){	++start;	}
	}
	destroyWindow("frame");
}



