/*
 * tracker.cpp
 *
 *  Created on: 2017Äê8ÔÂ10ÈÕ
 *      Author: JYF
 */


#include "tracker.h"
#define FILTER_LEN 2*LIFE

Size initTargetSize;

void showFrame(string file_list, string gt_file){
	vector<Rect> rois;
	vector<string> file_names;

	Rect bbox;

	readGroundTruth(gt_file, rois);
	readFileName(file_list, file_names);

	namedWindow("frame");

	assert(rois.size()==file_names.size());

	char key=0;
	for(uint k=0;k<file_names.size();++k){
		Mat frameImg=imread(file_names[k]);
		Mat sampleMat, labelMat;

		bbox=rois[k];
		vector<Rect> neg_rois;

		createNegativeSamples(frameImg, bbox, neg_rois);

		padROI(bbox,Size(frameImg.cols, frameImg.rows));
		rectangle(frameImg, bbox, Scalar(0,255,255), 1);

		for(uint i=0;i<neg_rois.size();++i){
			rectangle(frameImg, neg_rois[i], Scalar(255,255,122), 1);
		}

		imshow("frame", frameImg);
		key=waitKey(50);
		if(key==27){	break;	}
	}
}

void runTrackerSimple(string file_list, string gt_file){
	vector<Rect> rois;
	vector<string> file_names;

	readGroundTruth(gt_file, rois);
	readFileName(file_list, file_names);

	namedWindow("frame");

	assert(rois.size()==file_names.size());

	extern vector<HogParam> params;

	Mat frame;

	Target curTarget;
	vector<Target> targets(1);

	MySVM svm;

	int ksize=3;
	char key=0;
	bool detected=false;

	Rect bbox, s_roi;

#ifndef _PRETRAIN
	Mat sampleMat, labelMat;

	for(uint j=0;j<params.size();++j){
		alignSize(Size(rois[0].width, rois[0].height), params[j].winSize);
	}
	getAllSamples(file_names, rois, params, sampleMat, labelMat);

	cout<<"data size: "<<sampleMat.cols<<","<<sampleMat.rows<<endl;
	cout<<"training..."<<endl;

	trainOrUpdateSVM(svm, sampleMat, labelMat, params, 1e4);
#endif
	cout<<"tracking..."<<endl;

	for(uint k=0;k<file_names.size();++k){
#ifdef _PRETRAIN
		Mat sampleMat, labelMat;
#endif
		frame=imread(file_names[k]);
		if(k==0){
			bbox=rois[k];
			curTarget.location=bbox;
		}
		else if(detected){
			cout<<"Target detected"<<endl;
			bbox=curTarget.location;
		}
#ifdef _PRETRAIN
		for(uint j=0;j<params.size();++j){
			alignSize(Size(bbox.width, bbox.height), params[j].winSize);
		}

		if(k==0){
			getSamplesFromFrame(frame, params, bbox, sampleMat, labelMat);
			trainOrUpdateSVM(svm, sampleMat, labelMat, params);
		}
#endif

		blur(frame,frame,Size(ksize,ksize));
		Mat dummy;
		frame.copyTo(dummy);

		s_roi=bbox;

		padROI(s_roi, Size(frame.cols, frame.rows));

		detected=detect_bbox_simple(frame(s_roi), curTarget, svm, Point(s_roi.x,s_roi.y));

		targets[0].location=curTarget.location;
		draw_bbox(dummy, targets, 1);
		//rectangle(dummy,s_roi,Scalar(0,255,255),1);
		imshow("frame", dummy);
		key=waitKey(10);
		if(key==27){	break;	}
	}

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
	char start=0;	int filter_len=FILTER_LEN;
	int ksize=5;

	Mat frame;
	Mat lastFrame;
	Mat filteredFrame;

	vector<Target> targets;
	vector<Target> valid_targets;
	Rect ref_roi;

	MySVM svm;
	extern vector<HogParam> params;
	Rect bbox, s_roi;
	for(uint k=0;k<file_names.size();++k){
		Mat sampleMat, labelMat;

		frame=imread(file_names[k]);
		if(k==0){
			bbox=rois[k];
		}
		else if(targets.size()>0){
			cout<<"Target detected"<<endl;
			bbox=targets[0].location;
		}

		getSamplesFromFrame(frame, params, bbox, sampleMat, labelMat);

		trainOrUpdateSVM(svm, sampleMat, labelMat, params);

		//cout<<"SVM updated"<<endl;
		blur(frame,frame,Size(ksize,ksize));
		Mat dummy;
		frame.copyTo(dummy);

		s_roi=bbox;
		padROI(s_roi, Size(frame.cols, frame.rows));

		if(bFirstFrame){
			bFirstFrame=0;
			filteredFrame.create(dummy.size(), CV_8UC1);
		}
		if(start){
			if(filter_len>0){
				detect_bbox(lastFrame, targets, svm, Size(s_roi.x,s_roi.y), false);
				if(valid_targets.size()>0){	draw_bbox(dummy, valid_targets, 1);	}
			}
			else{
				filter_len=FILTER_LEN+1;
				filteredFrame.setTo(0);
				detect_bbox(frame(s_roi), targets, svm, Size(s_roi.x,s_roi.y),true);
				valid_targets=targets;
				draw_bbox(dummy, targets, 1);
			}
		}
		imshow("frame", dummy);
		key=waitKey(1);
		if(key==27){	break;	}
		frame(s_roi).copyTo(lastFrame);
		if(start){	--filter_len;	}
		if(start==0){	++start;	}
	}
	destroyWindow("frame");
}



