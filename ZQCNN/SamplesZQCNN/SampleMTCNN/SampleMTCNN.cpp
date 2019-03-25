#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_MTCNN_old.h"
#include "ZQ_CNN_MTCNN.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_CompileConfig.h"
#if ZQ_CNN_USE_BLAS_GEMM
#if __ARM_NEON
#include <openblas/cblas.h>
#else
#include <openblas/cblas.h>
#pragma comment(lib,"libopenblas.lib")
#endif
#elif ZQ_CNN_USE_MKL_GEMM
#include "mkl/mkl.h"
#pragma comment(lib,"mklml.lib")
#else
#pragma comment(lib,"ZQ_GEMM.lib")
#endif
#if !defined(_WIN32)
#include <sched.h>
#endif
using namespace ZQ;
using namespace std;
using namespace cv;

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.7)
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			else
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
			}

			for (int num = 0; num < 5; num++)
				circle(image, cv::Point(*(it->ppoint + num) + 0.5f, *(it->ppoint + num + 5) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_BBox106>& thirdBbox)
{
	std::vector<ZQ_CNN_BBox106>::const_iterator it = thirdBbox.begin();
	for (; it != thirdBbox.end(); it++)
	{
		if ((*it).exist)
		{
			if (it->score > 0.7)
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			else
			{
				cv::rectangle(image, cv::Point((*it).col1, (*it).row1), cv::Point((*it).col2, (*it).row2), cv::Scalar(0, 255, 0), 2, 8, 0);
			}

			for (int num = 0; num < 106; num++)
				circle(image, cv::Point(*(it->ppoint + num * 2) + 0.5f, *(it->ppoint + num * 2 + 1) + 0.5f), 1, cv::Scalar(0, 255, 255), -1);
		}
		else
		{
			printf("not exist!\n");
		}
	}
}

int main(int argc, const char** argv)
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

	int num_threads = 1;
#if ZQ_CNN_USE_BLAS_GEMM
	printf("set openblas thread_num = %d\n",num_threads);
	openblas_set_num_threads(num_threads);
#elif ZQ_CNN_USE_MKL_GEMM
	mkl_set_num_threads(num_threads);
#endif

	Mat image0 = cv::imread(argv[1], 1);
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}
	if (image0.channels() == 1)
		cv::cvtColor(image0, image0, CV_GRAY2BGR);
	
	std::vector<ZQ_CNN_BBox> thirdBbox;
	std::vector<ZQ_CNN_BBox106> thirdBbox106;
	ZQ_CNN_MTCNN mtcnn;
	std::string result_name;
	mtcnn.TurnOnShowDebugInfo();
	const int use_pnet20 = true;
	bool landmark106 = false;
	int thread_num = 0;
	bool special_handle_very_big_face = false;
	result_name = "resultdet.jpg";
	if (use_pnet20)
	{
		if (landmark106)
		{
#if defined(_WIN32)
			if (!mtcnn.Init("model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				//"model/det2.zqparams", "model/det2_bgr.nchwbin",
				"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin", 
				thread_num, true,
				"model/det5-dw96-v2s.zqparams", "model/det5-dw96-v2s.nchwbin"
				//"model/det3.zqparams", "model/det3_bgr.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin",
				thread_num, true,
				"../../model/det5-dw96-v2s.zqparams", "../../model/det5-dw96-v2s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
		else
		{
#if defined(_WIN32)
			if (!mtcnn.Init("model/det1-dw20-fast.zqparams", "model/det1-dw20-fast.nchwbin",
				"model/det2-dw24-fast.zqparams", "model/det2-dw24-fast.nchwbin",
				//"model\\det2.zqparams", "model\\det2_bgr.nchwbin",
				"model/det3-dw48-fast.zqparams", "model/det3-dw48-fast.nchwbin",
				thread_num, false,
				"model\\det4-dw48-v2n.zqparams", "model\\det4-dw48-v2n.nchwbin"
				//"model/det3.zqparams", "model/det3_bgr.nchwbin"
#else
			if (!mtcnn.Init("../../model/det1-dw20-fast.zqparams", "../../model/det1-dw20-fast.nchwbin",
				"../../model/det2-dw24-fast.zqparams", "../../model/det2-dw24-fast.nchwbin",
				//"model/det2.zqparams", "model/det2_bgr.nchwbin",
				"../../model/det3-dw48-fast.zqparams", "../../model/det3-dw48-fast.nchwbin", 
				thread_num, false,
				"model/det4-dw48-v2s.zqparams", "model/det4-dw48-v2s.nchwbin"
				//"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin"
#endif
			))
			{
				cout << "failed to init!\n";
				return EXIT_FAILURE;
			}
		}
		mtcnn.SetPara(image0.cols, image0.rows, 20, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);
	}
	else
	{
#if defined(_WIN32)
		if (!mtcnn.Init("model/det1.zqparams", "model/det1_bgr.nchwbin",
			"model/det2.zqparams", "model/det2_bgr.nchwbin",
			"model/det3.zqparams", "model/det3_bgr.nchwbin", thread_num))
#else
		if (!mtcnn.Init("../../model/det1.zqparams", "../../model/det1_bgr.nchwbin",
			"../../model/det2.zqparams", "../../model/det2_bgr.nchwbin",
			"../../model/det3.zqparams", "../../model/det3_bgr.nchwbin", thread_num))
#endif
		{
			cout << "failed to init!\n";
			return EXIT_FAILURE;
		}

		mtcnn.SetPara(image0.cols, image0.rows, 20, 0.6, 0.7, 0.7, 0.4, 0.5, 0.5, 0.709, 4, 12, 2, special_handle_very_big_face);
	}
	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.TurnOnShowDebugInfo();
	int out_loop = 4;
	for(int o_it = 0; o_it < out_loop; o_it++)
	{
		int iters = nIters;
		double t1 = omp_get_wtime();
		for (int i = 0; i < iters; i++)
		{
			if (i == iters / 2)
				mtcnn.TurnOnShowDebugInfo();
			else
				mtcnn.TurnOffShowDebugInfo();
			if (landmark106 && use_pnet20)
			{
				if (!mtcnn.Find106(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox106))
				{
					cout << "failed to find face!\n";
					//return EXIT_FAILURE;
					continue;
				}
			}
			else
			{
				if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
				{
					cout << "failed to find face!\n";
					//return EXIT_FAILURE;
					continue;
				}
			}
		}
		double t2 = omp_get_wtime();
		printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);
	}

	namedWindow("result");
	if (landmark106 && use_pnet20)
		Draw(image0, thirdBbox106);
	else
		Draw(image0, thirdBbox);
	imwrite(result_name, image0);
	imshow("result", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}
