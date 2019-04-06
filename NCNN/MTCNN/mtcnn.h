//
// Created by Lonqi on 2017/11/18.
//
#pragma once

#ifndef __MTCNN_NCNN_H__
#define __MTCNN_NCNN_H__
#include "net.h"
//#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
using namespace std;
//using namespace cv;

#ifndef __min
#define __min(x,y) ((x<y)?(x):(y))
#endif

#ifndef __max
#define __max(x,y) ((x>y)?(x):(y))
#endif

struct Bbox
{
	bool exist;
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	float ppoint[10];
	float regreCoord[4];
};

class ZQ_CNN_OrderScore
	{
	public:
		float score;
		int oriOrder;

		ZQ_CNN_OrderScore()
		{
			memset(this, 0, sizeof(ZQ_CNN_OrderScore));
		}
	};

bool cmpScore(Bbox lsh, Bbox rsh) {
		if (lsh.score < rsh.score)
			return true;
		else
			return false;
	}
	
	static bool _cmp_score(const ZQ_CNN_OrderScore& lsh, const ZQ_CNN_OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}
	
class MTCNN {

public:
	MTCNN(const string &model_path, int pnet_size = 20, int stride = 4)
	{

		std::string param_files[3] = {
			model_path + "/det1-dw20-fast.param",
			model_path + "/det2-dw24-fast.param",
			model_path + "/det3-dw48-fast.param"
		};

		std::string bin_files[3] = {
			model_path + "/det1-dw20-fast.bin",
			model_path + "/det2-dw24-fast.bin",
			model_path + "/det3-dw48-fast.bin"
		};
		//printf("hello1\n");
		Pnet.load_param(param_files[0].c_str());
		//printf("hello1-1\n");
		Pnet.load_model(bin_files[0].c_str());
		//printf("hello2\n");
		Rnet.load_param(param_files[1].c_str());
		Rnet.load_model(bin_files[1].c_str());
		//printf("hello3\n");
		Onet.load_param(param_files[2].c_str());
		Onet.load_model(bin_files[2].c_str());
		this->pnet_size = pnet_size, this->pnet_stride = stride;
		//printf("hello4\n");
		MIN_DET_SIZE = pnet_size;
	}
	~MTCNN()
	{
		Pnet.clear();
		Rnet.clear();
		Onet.clear();
	}

	void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox, int min_size, bool show_debug_info)
	{
		minsize = min_size;
		clock_t t1 = clock();
		img = img_;
		img_w = img.w;
		img_h = img.h;
		img.substract_mean_normalize(mean_vals, norm_vals);
		clock_t t2 = clock();
		PNet();
		clock_t t3 = clock();

		//the first stage's nms
		if (firstBbox_.size() < 1) return;
		nms(firstBbox_, nms_threshold[0]);
		refine(firstBbox_, img_h, img_w, true);
		printf("firstBbox_.size()=%d\n", firstBbox_.size());

		clock_t t4 = clock();
		//second stage
		RNet();
		clock_t t5 = clock();
		printf("secondBbox_.size()=%d\n", secondBbox_.size());
		if (secondBbox_.size() < 1) return;
		nms(secondBbox_, nms_threshold[1]);
		refine(secondBbox_, img_h, img_w, true);
		clock_t t6 = clock();
		//third stage 
		ONet();
		clock_t t7 = clock();
		printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
		if (thirdBbox_.size() < 1) return;
		refine(thirdBbox_, img_h, img_w, true);
		nms(thirdBbox_, nms_threshold[2], "Min");
		finalBbox = thirdBbox_;
		clock_t t8 = clock();
		if (show_debug_info)
		{
			printf("Pnet: %.3f ms nms: %.3f\n", 0.001*(t3 - t2), 0.001*(t4 - t3));
			printf("Rnet[%d]: %.3f ms nms: %.3f\n", firstBbox_.size(), 0.001*(t5 - t4), 0.001*(t6 - t5));
			printf("Onet[%d]: %.3f ms nms: %.3f\n", secondBbox_.size(), 0.001*(t7 - t6), 0.001*(t8 - t7));
			printf("final found: %d, cost: %.3d ms\n", finalBbox.size(), 0.001*(t8 - t1));
		}
	}

private:
	void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, float scale)
	{
		int stride = pnet_stride;
		int cellsize = pnet_size;
		//score p
		float *p = score.channel(1);//score.data + score.cstep;
		//float *plocal = location.data;
		Bbox bbox;
		float inv_scale = 1.0f / scale;
		for (int row = 0; row < score.h; row++) {
			for (int col = 0; col < score.w; col++) {

				if (*p > threshold[0]) {
					bbox.score = *p;
					bbox.x1 = round((stride*col + 1)*inv_scale);
					bbox.y1 = round((stride*row + 1)*inv_scale);
					bbox.x2 = round((stride*col + 1 + cellsize)*inv_scale);
					bbox.y2 = round((stride*row + 1 + cellsize)*inv_scale);
					bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
					const int index = row * score.w + col;
					for (int channel = 0; channel < 4; channel++) {
						bbox.regreCoord[channel] = location.channel(channel)[index];
					}
					boundingBox_.push_back(bbox);
				}
				p++;
				//plocal++;
			}
		}
	}
	void nmsTwoBoxs(vector<Bbox> &boundingBox_, vector<Bbox> &previousBox_, const float overlap_threshold, string modelname = "Union")
		(vector<Bbox>& boundingBox_, vector<Bbox>& previousBox_, const float overlap_threshold, string modelname)
	{
		if (boundingBox_.empty()) {
			return;
		}
		sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
		float IOU = 0;
		float maxX = 0;
		float maxY = 0;
		float minX = 0;
		float minY = 0;
		//std::cout << boundingBox_.size() << " ";
		for (std::vector<Bbox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++) {
			for (std::vector<Bbox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();) {
				int i = itx - boundingBox_.begin();
				int j = ity - previousBox_.begin();
				maxX = __max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
				maxY = __max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
				minX = __min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
				minY = __min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
				//maxX1 and maxY1 reuse
				maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
				maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
				//IOU reuse for the area of two bbox
				IOU = maxX * maxY;
				if (!modelname.compare("Union"))
					IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
				else if (!modelname.compare("Min")) {
					IOU = IOU / ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area : previousBox_.at(j).area);
				}
				if (IOU > overlap_threshold&&boundingBox_.at(i).score > previousBox_.at(j).score) {
					//if (IOU > overlap_threshold) {
					itx = boundingBox_.erase(itx);
				}
				else {
					itx++;
				}
			}
		}
		//std::cout << boundingBox_.size() << std::endl;
	}
	
		static void _nms(std::vector<Bbox> &boundingBox, std::vector<ZQ_CNN_OrderScore> &bboxScore, const float overlap_threshold, 
			const std::string& modelname = "Union", int overlap_count_thresh = 0)
		{
	if (boundingBox.empty() || overlap_threshold >= 1.0)
			{
				return;
			}
			std::vector<int> heros;
			std::vector<int> overlap_num;
			//sort the score
			sort(bboxScore.begin(), bboxScore.end(), _cmp_score);

			int order = 0;
			float IOU = 0;
			float maxX = 0;
			float maxY = 0;
			float minX = 0;
			float minY = 0;
			while (bboxScore.size() > 0)
			{
				order = bboxScore.back().oriOrder;
				bboxScore.pop_back();
				if (order < 0)continue;
				heros.push_back(order);
				int cur_overlap = 0;
				boundingBox[order].exist = false;//delete it
				int box_num = boundingBox.size();
				for (int num = 0; num < box_num; num++)
					{
						if (boundingBox[num].exist)
						{
							//the iou
							maxY = __max(boundingBox[num].y1, boundingBox[order].y1);
							maxX = __max(boundingBox[num].x1, boundingBox[order].x1);
							minY = __min(boundingBox[num].y2, boundingBox[order].y2);
							minX = __min(boundingBox[num].x2, boundingBox[order].x2);
							//maxX1 and maxY1 reuse 
							maxX = __max(minX - maxX + 1, 0);
							maxY = __max(minY - maxY + 1, 0);
							//IOU reuse for the area of two bbox
							IOU = maxX * maxY;
							float area1 = boundingBox[num].area;
							float area2 = boundingBox[order].area;
							if (!modelname.compare("Union"))
								IOU = IOU / (area1 + area2 - IOU);
							else if (!modelname.compare("Min"))
							{
								IOU = IOU / __min(area1, area2);
							}
							if (IOU > overlap_threshold)
							{
								cur_overlap++;
								boundingBox[num].exist = false;
								for (std::vector<ZQ_CNN_OrderScore>::iterator it = bboxScore.begin(); it != bboxScore.end(); it++)
								{
									if ((*it).oriOrder == num)
									{
										(*it).oriOrder = -1;
										break;
									}
								}
							}
						}
					}
				
				overlap_num.push_back(cur_overlap);
			}
			
			//clear exist= false;
			for (int i = boundingBox.size() - 1; i >= 0; i--)
			{
				if (!boundingBox[i].exist)
				{
					boundingBox.erase(boundingBox.begin() + i);
				}
			}
	}
	void nms(vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname = "Union")
	{
		int num = boundingBox_.size();
		if(num == 0)
			return ;
		std::vector<ZQ_CNN_OrderScore> order_score(num);
		for(int i = 0;i < num;i++)
		{
			boundingBox_[i].exist = true;
			order_score[i].score = boundingBox_[i].score;
			order_score[i].oriOrder = i;
		}
		_nms(boundingBox_, order_score, overlap_threshold, modelname,0);
		
	}

	void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square)
		(vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
		if (vecBbox.empty()) {
			cout << "Bbox is empty!!" << endl;
			return;
		}
		float bbw = 0, bbh = 0, maxSide = 0;
		float h = 0, w = 0;
		float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
		for (vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
			bbw = (*it).x2 - (*it).x1 + 1;
			bbh = (*it).y2 - (*it).y1 + 1;
			x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
			y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
			x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
			y2 = (*it).y2 + (*it).regreCoord[3] * bbh;



			if (square) {
				w = x2 - x1 + 1;
				h = y2 - y1 + 1;
				maxSide = (h > w) ? h : w;
				x1 = x1 + w*0.5 - maxSide*0.5;
				y1 = y1 + h*0.5 - maxSide*0.5;
				(*it).x2 = round(x1 + maxSide - 1);
				(*it).y2 = round(y1 + maxSide - 1);
				(*it).x1 = round(x1);
				(*it).y1 = round(y1);
			}

			//boundary check
			if ((*it).x1 < 0)(*it).x1 = 0;
			if ((*it).y1 < 0)(*it).y1 = 0;
			if ((*it).x2 > width)(*it).x2 = width - 1;
			if ((*it).y2 > height)(*it).y2 = height - 1;

			it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
		}
	}


	void PNet()
	{
		firstBbox_.clear();
		float minl = img_w < img_h ? img_w : img_h;
		float m = (float)MIN_DET_SIZE / minsize;
		minl *= m;
		float factor = pre_facetor;
		vector<float> scales_;
		while (minl > MIN_DET_SIZE) {
			scales_.push_back(m);
			minl *= factor;
			m = m*factor;
		}
		for (size_t i = 0; i < scales_.size(); i++) {
			int hs = (int)ceil(img_h*scales_[i]);
			int ws = (int)ceil(img_w*scales_[i]);
			printf("hxw = %dx%d\n", hs, ws);
			ncnn::Mat in;
			resize_bilinear(img, in, ws, hs);
			ncnn::Extractor ex = Pnet.create_extractor();
			//ex.set_num_threads(2);
			ex.set_light_mode(true);
			ex.input("data", in);
			ncnn::Mat score_, location_;
			ex.extract("prob1", score_);
			ex.extract("conv4-2", location_);
			std::vector<Bbox> boundingBox_;
			generateBbox(score_, location_, boundingBox_, scales_[i]);
			nms(boundingBox_, nms_threshold[0]);
			firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
			boundingBox_.clear();
		}
	}

	void RNet()
	{
		secondBbox_.clear();
		int count = 0;
		for (vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++) {
			ncnn::Mat tempIm;
			copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 24, 24);
			ncnn::Extractor ex = Rnet.create_extractor();
			//ex.set_num_threads(2);
			ex.set_light_mode(true);
			ex.input("data", in);
			ncnn::Mat score, bbox;
			ex.extract("prob1", score);
			ex.extract("conv5-2", bbox);
			if ((float)score[1] > threshold[1]) {
				for (int channel = 0; channel < 4; channel++) {
					it->regreCoord[channel] = (float)bbox[channel];//*(bbox.data+channel*bbox.cstep);
				}
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(1)[0];//*(score.data+score.cstep);
				secondBbox_.push_back(*it);
			}
		}
	}
	void ONet()
	{
		thirdBbox_.clear();
		for (vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++) {
			ncnn::Mat tempIm;
			copy_cut_border(img, tempIm, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 48, 48);
			ncnn::Extractor ex = Onet.create_extractor();
			//ex.set_num_threads(2);
			ex.set_light_mode(true);
			ex.input("data", in);
			ncnn::Mat score, bbox, keyPoint;
			ex.extract("prob1", score);
			ex.extract("conv6-2", bbox);
			//ex.extract("conv6-3", keyPoint);
			if ((float)score[1] > threshold[2]) {
				for (int channel = 0; channel < 4; channel++) {
					it->regreCoord[channel] = (float)bbox[channel];
				}
				it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
				it->score = score.channel(1)[0];
				/*for (int num = 0; num<5; num++) {
					(it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint[num];
					(it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint[num + 5];
				}*/

				thirdBbox_.push_back(*it);
			}
		}
	}



	ncnn::Net Pnet, Rnet, Onet;
	ncnn::Mat img;

	const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };
	const float mean_vals[3] = { 127.5, 127.5, 127.5 };
	const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
	int MIN_DET_SIZE;
	std::vector<Bbox> firstPreviousBbox_, secondPreviousBbox_, thirdPrevioussBbox_;
	std::vector<Bbox> firstBbox_, secondBbox_, thirdBbox_;
	int img_w, img_h;

private://部分可调参数
	const float threshold[3] = { 0.5f, 0.6f, 0.8f };
	int minsize;
	int pnet_size, pnet_stride;
	const float pre_facetor = 0.709f;

};


#endif //__MTCNN_NCNN_H__
