#include "net.h"
#include "cpu.h"
#include "ZQ_CNN_MTCNN_ncnn.h"
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
using namespace ZQ;
using namespace std;
using namespace cv;

static void Draw(cv::Mat &image, const std::vector<ZQ_CNN_MTCNN_ncnn::ZQ_CNN_BBox>& thirdBbox)
{
	std::vector<ZQ_CNN_MTCNN_ncnn::ZQ_CNN_BBox>::const_iterator it = thirdBbox.begin();
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

int main(int argc, char** argv)
{
	int thread_num = 2;
#if !defined(_WIN32)
	if (argc > 1)
	{
		int core_id = atoi(argv[1]);
		if (core_id >= 0)
		{
			cpu_set_t mask;
			CPU_ZERO(&mask);
			CPU_SET(atoi(argv[1]), &mask);
			if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
				perror("sched_setaffinity");
			}
		}
	}

	if (argc > 2)
		thread_num = atoi(argv[2]);

#endif

	int iters = 100;
	int min_size = 20;

#if defined(_WIN32)
	Mat image0 = cv::imread("../../../images/4.jpg", 1);
#else
	Mat image0 = cv::imread("../../../images/test2.jpg", 1);
#endif
	if (image0.empty())
	{
		cout << "empty image\n";
		return EXIT_FAILURE;
	}

	static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
	static ncnn::UnlockedPoolAllocator g_workspace_pool_allocator;
	ncnn::Option opt;
	opt.lightmode = true;
	opt.num_threads = 1;
	opt.blob_allocator = &g_blob_pool_allocator;
	opt.workspace_allocator = &g_workspace_pool_allocator;
	ncnn::set_default_option(opt);
	ncnn::set_cpu_powersave(0);
	ncnn::set_omp_dynamic(0);
	ncnn::set_omp_num_threads(1);
	g_blob_pool_allocator.clear();
	g_workspace_pool_allocator.clear();

	//cv::resize(image0, image0, cv::Size(), 2, 2);
	if (image0.channels() == 1)
		cv::cvtColor(image0, image0, CV_GRAY2BGR);
	//cv::convertScaleAbs(image0, image0, 2.0);
	/* TIPS: when finding tiny faces for very big image, gaussian blur is very useful for Pnet*/
	bool run_blur = true;
	int kernel_size = 3, sigma = 2;
	if (image0.cols * image0.rows >= 2500 * 1600)
	{
		run_blur = false;
		kernel_size = 5;
		sigma = 3;
	}
	else if (image0.cols * image0.rows >= 1920 * 1080)
	{
		run_blur = false;
		kernel_size = 3;
		sigma = 2;
	}
	else
	{
		run_blur = false;
	}

	if (run_blur)
	{
		cv::Mat blur_image0;
		int nBlurIters = 1000;
		double t00 = omp_get_wtime();
		for (int i = 0; i < nBlurIters; i++)
			cv::GaussianBlur(image0, blur_image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
		double t01 = omp_get_wtime();
		printf("[%d] blur cost %.3f secs, 1 blur costs %.3f ms\n", nBlurIters, t01 - t00, 1000 * (t01 - t00) / nBlurIters);
		cv::GaussianBlur(image0, image0, cv::Size(kernel_size, kernel_size), sigma, sigma);
	}

	std::vector<ZQ_CNN_MTCNN_ncnn::ZQ_CNN_BBox> thirdBbox;
	ZQ_CNN_MTCNN_ncnn mtcnn;
	std::string result_name;
	mtcnn.TurnOnShowDebugInfo();
	//mtcnn.SetLimit(300, 50, 20);
	
	
	bool special_handle_very_big_face = false;
	result_name = "resultdet.jpg";
	
#if defined(_WIN32)
	if (!mtcnn.Init("../../model/det1-dw20-fast.ncnnparam", "../../model/det1-dw20-fast.ncnnbin",
		"../../model/det2-dw24-fast.ncnnparam", "../../model/det2-dw24-fast.ncnnbin",
		"../../model/det3-dw48-fast.ncnnparam", "../../model/det3-dw48-fast.ncnnbin",
		thread_num, false,
		"../../model/det4-dw48-v2n.ncnnparam", "../../model/det4-dw48-v2n.ncnnbin"
#else
	if (!mtcnn.Init("../../model/det1-dw20-fast.ncnnparam", "../../model/det1-dw20-fast.ncnnbin",
		"../../model/det2-dw24-fast.ncnnparam", "../../model/det2-dw24-fast.ncnnbin",
		"../../model/det3-dw48-fast.ncnnparam", "../../model/det3-dw48-fast.ncnnbin",
		thread_num, false,
		"model/det4-dw48-v2s.ncnnparam", "model/det4-dw48-v2s.ncnnbin"
#endif
	))
	{
		cout << "failed to init!\n";
		return EXIT_FAILURE;
	}
	mtcnn.SetPara(image0.cols, image0.rows, min_size, 0.5, 0.6, 0.8, 0.4, 0.5, 0.5, 0.709, 3, 20, 4, special_handle_very_big_face);

	
	/****************************************/
	mtcnn.TurnOffShowDebugInfo();
	//mtcnn.TurnOnShowDebugInfo();
	double t1 = omp_get_wtime();
	for (int i = 0; i < iters; i++)
	{
		if (i == iters / 2)
			mtcnn.TurnOnShowDebugInfo();
		else
			mtcnn.TurnOffShowDebugInfo();

		if (!mtcnn.Find(image0.data, image0.cols, image0.rows, image0.step[0], thirdBbox))
		{
			cout << "failed to find face!\n";
			//return EXIT_FAILURE;
			continue;
		}
	}
	double t2 = omp_get_wtime();
	printf("total %.3f s / %d = %.3f ms\n", t2 - t1, iters, 1000 * (t2 - t1) / iters);

	namedWindow("result");
	Draw(image0, thirdBbox);
	imwrite(result_name, image0);
	imshow("result", image0);

	waitKey(0);
	return EXIT_SUCCESS;
}
