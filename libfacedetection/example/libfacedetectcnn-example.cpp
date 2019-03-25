/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2019, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"
#if !defined(_WIN32)
#include <sched.h>
#endif
//define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000
using namespace cv;

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s <image_file_name> [nIters] [core_id]\n", argv[0]);
        return -1;
    }
	int nIters = 1000;
	if(argc > 2)
		nIters = atoi(argv[2]);
#if !defined(_WIN32)
	if (argc > 3)
	{
		cpu_set_t mask;
		CPU_ZERO(&mask);
		CPU_SET(atoi(argv[3]), &mask);
		if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
			perror("sched_setaffinity");
		}
	}
#endif
	//load an image and convert it to gray (single-channel)
	Mat image = imread(argv[1]); 
	if(image.empty())
	{
		fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
		return -1;
	}

	int * pResults = NULL; 
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }
	

	///////////////////////////////////////////
	// CNN face detection 
	// Best detection rate
	//////////////////////////////////////////
	//!!! The input image must be a RGB one (three-channel)
	//!!! DO NOT RELEASE pResults !!!
	clock_t t1 = clock();
	for(int i = 0;i < nIters;i++)
	{
		pResults = facedetect_cnn(pBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step);
	}
	clock_t t2 = clock();
	double time = t2-t1;
#if defined(_WIN32)
	time *= 1e-3;
#else
	time *= 1e-6;
#endif
	printf("%.3f s / %d Iters, 1 Iter costs %.3f ms\n",time,nIters, 1000.0*time/nIters);
    printf("%d faces detected.\n", (pResults ? *pResults : 0));
	Mat result_cnn = image.clone();
	//print the detection results
	for(int i = 0; i < (pResults ? *pResults : 0); i++)
	{
        short * p = ((short*)(pResults+1))+142*i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int confidence = p[4];
		int angle = p[5];

		printf("face_rect=[%d, %d, %d, %d], confidence=%d, angle=%d\n", x,y,w,h,confidence, angle);
		rectangle(result_cnn, Rect(x, y, w, h), Scalar(0, 255, 0), 2);
	}
	imshow("result_cnn", result_cnn);

	waitKey();

    //release the buffer
    free(pBuffer);

	return 0;
}
