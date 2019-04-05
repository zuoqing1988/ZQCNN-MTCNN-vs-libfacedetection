// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>
#include "net.h"


int main(int argc, char** argv)
{
	ncnn::Net net;
	net.load_param("C:/Users/ZQ/Desktop/model-y1/model-y1-test2/GenderAge-new.param");
	net.load_model("C:/Users/ZQ/Desktop/model-y1/model-y1-test2/GenderAge-new.bin");
    const char* imagepath1 = "C:/Users/ZQ/Desktop/gamodel-r50/00_.jpg";
	cv::Mat img1 = cv::imread(imagepath1);
	ncnn::Mat in1 = ncnn::Mat::from_pixels_resize(img1.data, ncnn::Mat::PIXEL_BGR, img1.cols, img1.rows, 112, 112);

	const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
	const float norm_vals[3] = { 1.0 / 127.5,1.0 / 127.5,1.0 / 127.5 };
	in1.substract_mean_normalize(mean_vals, norm_vals);
	//in2.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Mat out1, out2;
	printf("begin\n");
	
	
	int out_iters = 1;
	for (int out_it = 0; out_it < out_iters; out_it++)
	{
		int iters = 1;
		double t1 = omp_get_wtime();
		for (int i = 0; i < iters; i++)
		{
			ncnn::Extractor ex1 = net.create_extractor();
			ex1.set_light_mode(false);
			ex1.set_num_threads(8);
			ex1.input("data", in1);
			ex1.extract("fc1", out1);
		}
		double t2 = omp_get_wtime();
		printf("[%d] cost %.3f s, 1 iter costs %.3f ms\n", iters, t2 - t1, 1000*(t2 - t1) / iters);
	}
	printf("c = %d, h = %d, w = %d\n", out1. c, out1. h, out1.w);
	int dim = out1.w;
	float* data1 = (float*)out1.data;

	for (int w = 0; w < dim; w+=2)
	{
		printf("%12.3f %12.3f\n", data1[w], data1[w+1]);
	}
	int gender = data1[0] < data1[1] ? 1 : 0;
	int age = 0;
	float range = 2;
	int age_min = 0;
	int age_max = 0;
	for (int w = 2; w < dim; w += 2)
	{
		age += (data1[w] - data1[w + 1]) < 0 ? 1 : 0;
		age_min += (data1[w] - data1[w + 1] + range) < 0 ? 1 : 0;
		age_max += (data1[w] - data1[w + 1] - range) < 0 ? 1 : 0;
	}
	printf("gender= %d, age = %d, (%d,%d)\n", gender, age, age_min-age, age_max-age);
	return 0;
}

