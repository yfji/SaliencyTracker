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

void createNegativeSamples(Mat& frame, Rect& bbox, vector<Rect>& neg_rois, float overlap){
	int neg_crop=NEG_CROP;	//total
	float sz_overlap=overlap;

	int neg_cnt=0;
	int shifts[4];
	int shiftx[4]={-1,0,1,0};
	int shifty[4]={0,-1,0,1};
	int bounding_sz[4]={1e6,1e6,1e6,1e6};

	bounding_sz[0]=min(bounding_sz[0],bbox.x);
	bounding_sz[1]=min(bounding_sz[1],bbox.y);
	bounding_sz[2]=min(bounding_sz[2],frame.cols-(bbox.x+bbox.width));
	bounding_sz[3]=min(bounding_sz[3],frame.rows-(bbox.y+bbox.height));

	int _0=(int)(bbox.width*sz_overlap);
	int _1=(int)(bbox.height*sz_overlap);
	shifts[0]=bbox.x;
	shifts[1]=bbox.y;
	shifts[2]=frame.cols-(bbox.x+bbox.width);
	shifts[3]=frame.rows-(bbox.y+bbox.height);
	if(shifts[0]<bbox.width-_0){	shifts[0]=0;	}
	if(shifts[1]<bbox.height-_1){	shifts[1]=0;	}
	if(shifts[2]<bbox.width-_0){	shifts[2]=0;	}
	if(shifts[3]<bbox.height-_1){	shifts[3]=0;	}

	int index=0;
	while(neg_cnt<neg_crop){
		Rect& sz=bbox;
		int _0=(int)(sz.width*sz_overlap);
		int _1=(int)(sz.height*sz_overlap);
		int min_shift[4]={sz.width-_0,sz.height-_1, sz.width-_0,sz.height-_1};

		for(int k=0;k<4;++k){
			if(shifts[k]==0){	continue;	}
			int deltax=shiftx[k]*(rand()%(shifts[k]-min_shift[k]+1)+min_shift[k]);
			int deltay=shifty[k]*(rand()%(shifts[k]-min_shift[k]+1)+min_shift[k]);
			if(deltax==0){
				if(shifts[0]>0 and shifts[2]>0){
					deltax=rand()%(shifts[2]+shifts[0])-shifts[0];
				}
			}
			if(deltay==0){
				if(shifts[1]>0 and shifts[3]>0){
					deltay=rand()%(shifts[3]+shifts[1])-shifts[1];
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
}

void getSamplesFromFrame(Mat& frame, vector<HogParam>& params, Rect& bbox, Mat& sampleMat, Mat& labelMat){
	int pos_crop=POS_CROP;
	int neg_crop=NEG_CROP;	//total

	float sz_overlap=0.2;
	vector<float> msHogFeat;
	vector<Rect> crop_boxes;

	//cout<<params[0].winSize.width<<","<<params[0].winSize.height<<endl;
	Rect rc;
	float start=1.0, mult;
	if(pos_crop>=4){	mult=1.05;	}
	else{	mult=1.1;	}
	for(int k=0;k<pos_crop;++k){
		rc.width=bbox.width*start;
		rc.height=bbox.height*start;
		rc.x=max(0, (int)(bbox.x-0.5*(rc.width-bbox.width)));
		rc.y=max(0, (int)(bbox.y-0.5*(rc.height-bbox.height)));
		rc.width=min(frame.cols-rc.x, rc.width);
		rc.height=min(frame.rows-rc.y, rc.height);
		crop_boxes.push_back(rc);
		start*=mult;
	}

	for(uint i=0;i<crop_boxes.size();++i){
		Mat roi=frame(crop_boxes[i]);
		for(int k=-1;k<2;++k){
			Mat flipped;
			//roi.copyTo(flipped);
			if(k>=0){	flip(roi,flipped,k);	}
			else{	flipped=roi;	}
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

	Mat neg;
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

void getAllSamples(vector<string>& file_names, vector<Rect>& rois, vector<HogParam>& params, Mat& sampleMat, Mat& labelMat, int n){
	vector<float> msHogFeat;
	int featLen=0;
	int samplesPerFrame=3*POS_CROP+NEG_CROP;
	int N=file_names.size()/n;
	for(uint k=0;k<params.size();++k){
		HogParam& param=params[k];
		int len=((param.winSize.width-param.blockSize)/param.stride+1)*((param.winSize.height-param.blockSize)/param.stride+1)*36;
		featLen+=len;
	}
	if(sampleMat.empty()){
		sampleMat.create(Size(featLen, samplesPerFrame*N), CV_32FC1);
	}
	if(labelMat.empty()){
		labelMat.create(Size(1, samplesPerFrame*N), CV_32FC1);
	}
	int base=0;
	for(int k=0;k<N;++k){
		Mat frame=imread(file_names[k]);
		Mat _sampleMat=sampleMat(Rect(0,base,featLen,samplesPerFrame));
		Mat _labelMat=labelMat(Rect(0,base,1,samplesPerFrame));

		getSamplesFromFrame(frame, params, rois[k], _sampleMat, _labelMat);
		base+=samplesPerFrame;
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

void erodeAndDilate(Mat& src, int dim){
	int g_nStructElementSize = dim;
	Mat element = getStructuringElement(MORPH_RECT,
			Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),
			Point( g_nStructElementSize, g_nStructElementSize ));

	//dilate black pixels, erode white pixels
	dilate(src, src, element);
	erode(src, src, element);
}

Mat multiScaleSaliency(Mat& src, const Rect& bbox, const Size& bounding, Point& offset){
	float max_scale=2;
	float num_scale=4;
	float res=max_scale/num_scale;
	float scale=max_scale;

	Mat roi, sa, canvas;
	Rect max_roi;
	bool bMax;
	Rect r;
	for(;scale>=1;scale-=res){
		r=bbox;
		padROI(r, bounding, scale);
		if(bMax){
			bMax=false;
			max_roi=r;
			canvas=Mat::zeros(Size(max_roi.width, max_roi.height), CV_8UC1);
			offset.x=max_roi.x;
			offset.y=max_roi.y;
		}
		sa=Mat::zeros(Size(max_roi.width, max_roi.height), CV_8UC1);
		src(r).copyTo(roi);
		r.x-=max_roi.x;
		r.y-=max_roi.y;
		roi=detectSaliency(roi);
		roi.copyTo(sa(r));
		bitwise_or(canvas, sa, canvas);
	}
	sa(r).setTo(255);
	bitwise_and(canvas, sa, canvas);
	return canvas;
}
