/*
 * features.cpp
 *
 *  Created on: 2017Äê7ÔÂ23ÈÕ
 *      Author: JYF
 */

#include "features.h"
#include <math.h>
#include <fstream>
//#define min(x,y) x<y?x:y

void featMaxBlur(Mat& src, Size ksize){
	int h=src.rows;
	int w=src.cols;
	int pooled_x=w/ksize.width;
	int pooled_y=h/ksize.height;
	uchar* data=src.data;
	for(int y=0;y<pooled_y;++y){
		int ystart=y*ksize.height;
		for(int x=0;x<pooled_x;++x){
			int xstart=x*ksize.width;
			int max=-10000;
			for(int yy=ystart;yy<ystart+min(ksize.height,h-ystart);++yy){
				for(int xx=xstart;xx<xstart+min(ksize.width,w-xstart);++xx){
					int index=yy*w+xx;
					if(data[index]>max){
						max=data[index];
					}
				}
			}
			for(int yy=ystart;yy<ystart+min(ksize.height,h-ystart);++yy){
				for(int xx=xstart;xx<xstart+min(ksize.width,w-xstart);++xx){
					int index=yy*w+xx;
					data[index]=max;
				}
			}
		}
	}
}

void getHogFeature(Mat& image, HogParam& param, vector<float>& hogFeat, bool cvt){
	Mat gray=image;
	if(cvt){
		int ch=image.channels();
		if(ch==3){
			cvtColor(image, gray, CV_BGR2GRAY);
		}
		resize(gray, gray, param.winSize);
		normalize(gray, gray, 0, 255, NORM_MINMAX);
	}
	HOGDescriptor hog;
	hog.blockSize=Size(param.blockSize, param.blockSize);
	hog.cellSize=Size(param.cellSize, param.cellSize);
	hog.blockStride=Size(param.stride, param.stride);
	hog.winSize=param.winSize;
	hog.compute(image, hogFeat);
}

void getMSHogFeature(Mat& image, vector<HogParam>& params, vector<float>& msHogFeat){
	Mat gray=image;
	int ch=image.channels();
	if(ch==3){
		cvtColor(image, gray, CV_BGR2GRAY);
	}
	resize(gray, gray, params[0].winSize);
	normalize(gray, gray, 0, 255, NORM_MINMAX);
	for(unsigned int i=0;i<params.size();++i){
		HogParam& param=params[i];
		vector<float> ssHogFeat;
		getHogFeature(gray, param, ssHogFeat, false);
		for(unsigned int j=0;j<ssHogFeat.size();++j)
			msHogFeat.push_back(ssHogFeat[j]);
	}
}

Mat detectSaliency(Mat& src){
	Mat sa;
	saliencyDetectFT(src, sa);
	threshold(sa, sa, 0, 255, THRESH_OTSU);
	return sa;
}

void createNegativeSamples(Mat& frame, vector<Rect>& bbox, vector<Rect>& neg_rois, float overlap){
	int neg_crop=NEG_CROP;	//total
	float sz_overlap=overlap;

	int neg_cnt=0;
	vector<int*> shifts(bbox.size());
	for(uint i=0;i<bbox.size();++i){	shifts[i]=new int[4];		}
	int shiftx[4]={-1,0,1,0};
	int shifty[4]={0,-1,0,1};
	int bounding_sz[4]={1e6,1e6,1e6,1e6};

	for(uint i=0;i<bbox.size();++i){
		Rect& sz=bbox[i];
		bounding_sz[0]=min(bounding_sz[0],sz.x);
		bounding_sz[1]=min(bounding_sz[1],sz.y);
		bounding_sz[2]=min(bounding_sz[2],frame.cols-(sz.x+sz.width));
		bounding_sz[3]=min(bounding_sz[3],frame.rows-(sz.y+sz.height));
	}

	for(uint i=0;i<bbox.size();++i){
		Rect& sz=bbox[i];
		int _0=(int)(sz.width*sz_overlap);
		int _1=(int)(sz.height*sz_overlap);
		shifts[i][0]=min(bounding_sz[0], sz.x);
		shifts[i][1]=min(bounding_sz[1], sz.y);
		shifts[i][2]=min(bounding_sz[2],frame.cols-(sz.x+sz.width));
		shifts[i][3]=min(bounding_sz[3],frame.rows-(sz.y+sz.height));
		if(shifts[i][0]<sz.width-_0){	shifts[i][0]=0;	}
		if(shifts[i][1]<sz.height-_1){	shifts[i][1]=0;	}
		if(shifts[i][2]<sz.width-_0){	shifts[i][2]=0;	}
		if(shifts[i][3]<sz.height-_1){	shifts[i][3]=0;	}
	}
	int index=0;
	while(neg_cnt<neg_crop){
		index=index%(bbox.size());
		Rect& sz=bbox[index];
		int _0=(int)(sz.width*sz_overlap);
		int _1=(int)(sz.height*sz_overlap);
		int pad[4]={shifts[index][0],shifts[index][1],shifts[index][2],shifts[index][3]};
		int min_shift[4]={sz.width-_0,sz.height-_1, sz.width-_0,sz.height-_1};

		for(int k=0;k<4;++k){
			if(pad[k]==0){	continue;	}
			int deltax=shiftx[k]*(rand()%(pad[k]-min_shift[k]+1)+min_shift[k]);
			int deltay=shifty[k]*(rand()%(pad[k]-min_shift[k]+1)+min_shift[k]);
			if(deltax==0){
				if(pad[0]>0 and pad[2]>0){
					deltax=rand()%(pad[2]+pad[0])-pad[0];
				}
			}
			if(deltay==0){
				if(pad[1]>0 and pad[3]>0){
					deltay=rand()%(pad[3]+pad[1])-pad[1];
				}
			}
			Rect _sz;
			_sz.x=max(0,sz.x+deltax);
			_sz.y=max(0,sz.y+deltay);
			_sz.width=min(sz.width, frame.cols-_sz.x-1);
			_sz.height=min(sz.height, frame.rows-_sz.y-1);

			neg_rois.push_back(_sz);
			++neg_cnt;
			if(neg_cnt==neg_crop){	break;	}
		}
		++index;
	}
	for(uint i=0;i<bbox.size();++i){	delete shifts[i];	}
}

void getSamplesFromFrame(Mat& frame, vector<HogParam>& params, vector<Rect>& bbox, Mat& sampleMat, Mat& labelMat){
	int neg_crop=NEG_CROP;	//total
	int pad_len=10;
	float pad_ratio=0.125;
	float crop_ratio=0.9;
	float sz_overlap=0.2;
	Size fixed_size=params[0].winSize;
	vector<float> msHogFeat;
	vector<Rect> crop_boxes;

	for(uint i=0;i<bbox.size();++i){
		Rect& sz=bbox[i];
		Rect padded_gt=Rect(max(0,sz.x-max(pad_len, (int)(sz.width*pad_ratio))), max(0,sz.y-max(pad_len, (int)(sz.height*pad_ratio))),\
				sz.width+2*max(pad_len, (int)(sz.width*pad_ratio)), sz.height+2*max(pad_len,(int)(sz.height*pad_ratio)));
		padded_gt.width=min(padded_gt.width, frame.cols-padded_gt.x);
		padded_gt.height=min(padded_gt.height, frame.rows-padded_gt.y);
		crop_boxes.push_back(bbox[i]);
		crop_boxes.push_back(padded_gt);
		Rect lt(padded_gt.x,padded_gt.y,(int)(padded_gt.width*crop_ratio),(int)(padded_gt.height*crop_ratio));
		Rect lb(padded_gt.x,padded_gt.y+(int)((1-crop_ratio)*padded_gt.height),(int)(padded_gt.width*crop_ratio),(int)(padded_gt.height*crop_ratio));
		Rect rt(padded_gt.x+(int)((1-crop_ratio)*padded_gt.width),padded_gt.y,(int)(padded_gt.width*crop_ratio),(int)(padded_gt.height*crop_ratio));
		Rect rb(padded_gt.x+(int)((1-crop_ratio)*padded_gt.width),padded_gt.y+(int)((1-crop_ratio)*padded_gt.height),(int)(padded_gt.width*crop_ratio),(int)(padded_gt.height*crop_ratio));
		crop_boxes.push_back(lt);crop_boxes.push_back(lb);
		crop_boxes.push_back(rt);crop_boxes.push_back(rb);
	}
	for(uint i=0;i<crop_boxes.size();++i){
		Mat roi=frame(crop_boxes[i]);
		for(int k=-1;k<2;++k){
			Mat flipped;
			roi.copyTo(flipped);
			if(k>=0){	flip(roi,flipped,k);	}
			resize(flipped,flipped,fixed_size);
			getMSHogFeature(flipped, params, msHogFeat);
			if(sampleMat.empty()){	sampleMat.create(Size(msHogFeat.size(), 3*crop_boxes.size()+neg_crop), CV_32FC1);	}
			if(labelMat.empty()){	labelMat.create(Size(1,3*crop_boxes.size()+neg_crop), CV_32FC1);	}
			float* samplePtr=sampleMat.ptr<float>(3*i+(k+1));
			float* labelPtr=labelMat.ptr<float>(3*i+(k+1));
			for(int j=0;j<sampleMat.cols;++j){
					samplePtr[j]=msHogFeat[j];
			}
			labelPtr[0]=1.0;
			vector<float>().swap(msHogFeat);
		}
	}
	//negative
	//cout<<"creating negative samples"<<endl;
	int offset=3*crop_boxes.size();
	vector<Rect>().swap(crop_boxes);
	createNegativeSamples(frame, bbox, crop_boxes, sz_overlap);
	for(uint i=0;i<crop_boxes.size();++i){
		Mat roi=frame(crop_boxes[i]);
		getMSHogFeature(roi, params, msHogFeat);
		float* samplePtr=sampleMat.ptr<float>(i+offset);
		float* labelPtr=labelMat.ptr<float>(i+offset);
		for(int j=0;j<sampleMat.cols;++j){
				samplePtr[j]=msHogFeat[j];
		}
		labelPtr[0]=-1.0;
		vector<float>().swap(msHogFeat);
	}
}

void getSVM(MySVM& mSVM){
	string svm_file_path="./svm_pot.xml";
	mSVM.load(svm_file_path.c_str());
}

int detectSVM(Mat& src, vector<HogParam>& params, MySVM& svm){
	vector<float> msHogFeat;
	getMSHogFeature(src, params, msHogFeat);
	Mat sampleMat(Size(msHogFeat.size(),1),CV_32FC1);
	float* samplePtr=sampleMat.ptr<float>(0);
	for(int k=0;k<sampleMat.cols;++k)
		samplePtr[k]=msHogFeat[k];
	float res=svm.predict(sampleMat);
	return (int)res;
}

void trainOrUpdateSVM(MySVM& svm, Mat& sampleMat, Mat& labelMat, vector<HogParam>& params, int iter){
	TermCriteria tc(CV_TERMCRIT_ITER, iter, FLT_EPSILON);
	SVMParams param;
	param.C=1;
	param.svm_type=SVM::C_SVC;
	param.kernel_type = SVM::LINEAR;
	param.term_crit=tc;
	svm.train(sampleMat, labelMat, Mat(), Mat(), param);//train
}
