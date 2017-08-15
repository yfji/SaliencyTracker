/*
 * impl.cpp
 *
 *  Created on: 2017Äê7ÔÂ25ÈÕ
 *      Author: JYF
 */

#include "impl.h"
#include <math.h>

void draw_bbox(Mat& frame, vector<Target>& targets, int width){
	if(targets.size()==0){
		//cout<<"no target detected"<<endl;
		return;
	}
	Scalar s=Scalar(0,255,255);
	if(frame.channels()==1){	s=255;	}
	for(int i=0;i<targets.size();++i)
		rectangle(frame, targets[i].location, s, width);
}

bool detect_bbox_simple(const Mat& src, Target& target, MySVM& classifier, const Point& offset){
	Mat scaled;
	int scale=1;
	resize(src, scaled, Size(src.cols/scale, src.rows/scale));
	scaled=detectSaliency(scaled);

	imshow("sa", scaled);
	IplImage ipl=scaled;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq* pContour=NULL;
	extern float max_area;
	extern float min_area;
	extern vector<HogParam> params;

	max_area=0.5*src.cols*src.rows;
	min_area=9.0;

	Rect lastLocation=target.location;

	lastLocation.x-=offset.x;
	lastLocation.y-=offset.y;

	max_area=min(max_area, (float)4*lastLocation.height*lastLocation.width);
	min_area=max(min_area, (float)0.2*lastLocation.height*lastLocation.width);

	float r=1.0*lastLocation.width/(1.0*lastLocation.height);

	float ratio[2]={r*0.2,r*4};	//w/h

	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	bool updated=false;
	for(;pContour;pContour=pContour->h_next){
		double area=fabs(cvContourArea(pContour))*scale*scale;
		if(area>max_area or area<min_area){
			//cvSeqRemove(pContour,0);
			//continue;
		}
		CvRect bbox=cvBoundingRect(pContour,0);
		Rect raw_bbox=Rect(bbox.x*scale, bbox.y*scale, bbox.width*scale, bbox.height*scale);

		double bw=1.0*raw_bbox.width;
		double bh=1.0*raw_bbox.height;
		if(bw/bh<ratio[0] or bw/bh>ratio[1]){
			//cvSeqRemove(pContour,0);
			//continue;
		}

		if(raw_bbox.x+raw_bbox.width<lastLocation.x or raw_bbox.y+raw_bbox.height<lastLocation.y or\
				raw_bbox.x>lastLocation.x+lastLocation.width or raw_bbox.y>lastLocation.y+lastLocation.height){
			//cvSeqRemove(pContour,0);
			//continue;
		}
		Mat roi=src(raw_bbox);
		int res=detectSVM(roi, params, classifier);
		if(res==-1){
			cvSeqRemove(pContour,0);
			continue;
		}
		target.location=raw_bbox;
		updated=true;
	}
	if(not updated){
		target.location=lastLocation;
	}
	target.location.x+=offset.x;
	target.location.y+=offset.y;
	cvReleaseMemStorage(&pStorage);
	return updated;
}

bool detect_bbox_contour(const Mat& sa, Target& target, Point& offset){
	IplImage ipl=sa;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq* pContour=NULL;

	int max_diff=10;

	Rect lastLocation=target.location;

	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	bool updated=false;
	for(;pContour;pContour=pContour->h_next){
		CvRect bbox=cvBoundingRect(pContour,0);
		Rect raw_bbox=Rect(bbox.x+offset.x, bbox.y+offset.y, bbox.width, bbox.height);
		if(abs(raw_bbox.x-lastLocation.x)<=max_diff and abs(raw_bbox.y-lastLocation.y)<=max_diff){
			target.location=raw_bbox;
			break;
		}
	}
	cvReleaseMemStorage(&pStorage);
	return true;
}

void detect_bbox(const Mat& src, vector<Target>& targets, MySVM& classifier, const Point& offset, bool bCurFrame){
	//one frame
	Mat scaled;
	int scale=2;
	resize(src, scaled, Size(src.cols/scale, src.rows/scale));
	scaled=detectSaliency(scaled);

	IplImage ipl=scaled;
	CvMemStorage* pStorage=cvCreateMemStorage(0);
	CvSeq* pContour=NULL;
	extern double max_area;
	extern double min_area;
	extern vector<HogParam> params;

	max_area=(double)(src.cols*src.rows/2);
	min_area=25.0;
	if(targets.size()>0){
		Rect& max_loc=targets[0].location;
		Rect& min_loc=targets[targets.size()-1].location;
		max_area=(double)min(max_area, max_loc.height*max_loc.width*2.0);
		min_area=(double)max(min_area, min_loc.height*min_loc.width/2.0);
	}
	double ratio[2]={0.2,0.9};	//w/h
	cvFindContours(&ipl, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	vector<Target>::iterator iter=targets.begin();
	for(;iter!=targets.end();++iter){	(*iter).life-=1;	}

	for(;pContour;pContour=pContour->h_next){
		double area=fabs(cvContourArea(pContour))*scale*scale;
		//if(area>max_area or area<min_area){
		//	cvSeqRemove(pContour,0);
		//	continue;
		//}
		CvRect bbox=cvBoundingRect(pContour,0);
		Rect raw_bbox=Rect(bbox.x*scale, bbox.y*scale, bbox.width*scale, bbox.height*scale);
		double bw=1.0*raw_bbox.width;
		double bh=1.0*raw_bbox.height;
		//if(bw/bh<ratio[0] or bw/bh>ratio[1]){
		//	cvSeqRemove(pContour,0);
		//	continue;
		//}

		Mat roi=src(raw_bbox);
		int res=detectSVM(roi, params, classifier);
		if(res==-1){
			cvSeqRemove(pContour,0);
			continue;
		}

		Target t;
		t.location=raw_bbox;
		int closest_index=0;
		bool is_new=is_new_target(t, targets, closest_index);

		if(is_new){
			if(not bCurFrame){
				if(target_detectable(t, src.size())){
					if(targets.size()<20){
						t.life=LIFE;
						targets.push_back(t);
					}
				}
			}
		}
		else if(target_detectable(t, src.size())){	//update
			targets[closest_index].location=raw_bbox;
			targets[closest_index].life+=1;
		}
		else{
			targets.erase(targets.begin()+closest_index);
		}
		cvSeqRemove(pContour, 0);
	}
	if(targets.size()>0){
		vector<Target>::iterator iter=targets.begin();
		for(;iter!=targets.end();){
			if((*iter).life==0){
				iter=targets.erase(iter);
			}
			else{
				(*iter).location.x=offset.x;
				(*iter).location.y+=offset.y;
				++iter;
			}
		}
	}
	vector<Target> temp=targets;
	vector<Target>().swap(targets);
	targets=temp;
	cvReleaseMemStorage(&pStorage);
}

void detect_bbox_kcf(Mat& src, vector<Target>& targets){
	//use kcf to track the target
}

bool is_new_target(Target& t, vector<Target>& targets, int& index){
	if(targets.size()==0)
		return true;
	bool is_new=true;
	int max_dist=10000;
	Rect& cur_loc=t.location;
	int metric=NEAR_METRIC;

	vector<Target>::iterator iter=targets.begin();
	for(;iter!=targets.end();++iter){
		Rect& loc=(*iter).location;
		if(abs(cur_loc.x-loc.x)>metric or abs(cur_loc.y-loc.y)>metric){
			continue;	//not near to this target
		}
		int dist=abs(cur_loc.x-loc.x)+abs(cur_loc.y-loc.y);
		if(dist<max_dist){
			max_dist=dist;
			index=iter-targets.begin();
		}
		is_new=false;
	}
	return is_new;
}

bool target_detectable(Target& t, Size win){
	Rect& loc=t.location;
	if(loc.x>PAD and loc.y>PAD and loc.x+loc.width<win.width-PAD and loc.y+loc.height<win.height-PAD)
		return true;
	return false;
}

void track_by_detect(){

}

bool target_compare(Target& t1, Target& t2){
	int t1_area=t1.location.width*t1.location.height;
	int t2_area=t2.location.width*t2.location.height;
	return t1_area>t2_area;
}

