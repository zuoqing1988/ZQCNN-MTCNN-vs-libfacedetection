#include "mtcnn.h"
#include <opencv2/opencv.hpp>


using namespace cv;

int main(int argc, const char** argv)
{
	if (argc < 2)
	{
		printf("Usage: %s <image_file_name> [nIters] [core_id]\n", argv[0]);
		return -1;
	}
	int nIters = 1000;
	if (argc > 2)
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
	char *model_path = "../models";
	MTCNN mtcnn(model_path, 20, 4);

	cv::Mat image;
	image = cv::imread(argv[1]);
	int out_it = 4;
	int min_size = 20;

	std::vector<Bbox> finalBbox;
	for (int o_it = 0; o_it < out_it; o_it++)
	{
		clock_t t1 = clock();
		for (int i = 0; i < nIters; i++)
		{
			ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows);
			mtcnn.detect(ncnn_img, finalBbox, min_size, i == (nIters / 2));
		}
		clock_t t2 = clock();
#if defined(_WIN32)
		double time = 1e-3*(t2 - t1);
#else
		double time = 1e-6*(t2 - t1);
#endif

		printf("total: %.3f s / %d = %.3f ms\n", time, nIters, 1e3*time / nIters);
	}

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

		for (int j = 0; j < 5; j = j + 1)
		{
			cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);

	cv::waitKey(0);
	return 0;
}
