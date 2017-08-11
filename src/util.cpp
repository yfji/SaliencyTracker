/*
 * util.cpp
 *
 *  Created on: 2017��8��10��
 *      Author: JYF
 */
#include "util.h"

void readGroundTruth(string gt_file, vector<Rect>& roi){
	ifstream in;
	in.open(gt_file.c_str(), ios::in);
	int x,y,w,h;
	char ch;
	char checked=0;
	char join_by_comma=0;
	while(not in.eof()){
		string line;
		getline(in, line);
		if(line.length()<=1){	continue;	}
		if(not checked){
			checked=1;
			const char* buff=line.c_str();
			for(int i=0;i<line.length();++i){
				if(buff[i]==','){
					join_by_comma=1;
					break;
				}
			}
		}
		stringstream ss(line);
		if(not join_by_comma){
			ss>>x>>y>>w>>h;
			ch=' ';
		}
		else{
			ss>>x;ss>>ch;
			ss>>y;ss>>ch;
			ss>>w;ss>>ch;
			ss>>h;
		}
		roi.push_back(Rect(x,y,w,h));
	}
	in.close();
}

void readFileName(string file_list, vector<string>& file_names){
	ifstream in;
	in.open(file_list.c_str(), ios::in);
	while(not in.eof()){
		string line;
		getline(in,line);
		if(line.length()>1)
			file_names.push_back(line);
	}
	in.close();
}

void readParamsFromFile(string param_file, vector<HogParam>& params){
	ifstream in;
	in.open(param_file.c_str(), ios::in);
	if(not in){	cout<<"Open file failed"<<endl;	return;	}
	HogParam param;
	bool bHasElement=false;
	int width, height;
	while(not in.eof()){
		string line;
		in>>line;
		if(line.length()<=1){	continue;	}
		if(strcmp("[Width]", line.c_str())==0){
			in>>line;
			width=atoi(line.c_str());
			param.winSize.width=width;
		}
		else if(strcmp("[Height]", line.c_str())==0){
			in>>line;
			height=atoi(line.c_str());
			param.winSize.height=height;
		}
		else if(strcmp("[HOG]", line.c_str())==0){
			if(bHasElement){
				param.winSize.width=width;
				param.winSize.height=height;
				params.push_back(param);
			}
			else{	bHasElement=true;	}
		}
		else{
			int num=atoi(line.substr(line.find(":",0)+1).c_str());
			if(line.find("cell")!=std::string::npos){	param.cellSize=num;	}
			else if(line.find("block")!=std::string::npos){	param.blockSize=num;	}
			else if(line.find("stride")!=std::string::npos){	param.stride=num;	}
			else{	assert(0);	}
		}
	}
	param.winSize.width=width;	param.winSize.height=height;
	params.push_back(param);
	in.close();
}

void printParams(vector<HogParam>& params){
	for(unsigned int i=0;i<params.size();++i){
		HogParam& param=params[i];
		cout<<"HOG["<<i<<"]"<<endl;
		cout<<"cell:"<<param.cellSize<<endl;
		cout<<"block:"<<param.blockSize<<endl;
		cout<<"stride:"<<param.stride<<endl;
		cout<<"winSize:("<<param.winSize.width<<","<<param.winSize.height<<")"<<endl;
	}
}



