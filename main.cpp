
//20241119
// zwg for fpc detect
#include <iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
#include "yolov8_obb_onnx.h"
#include<time.h>


using namespace std;
using namespace cv;
using namespace dnn;


void getroifrommark(cv::Mat src_img, cv::RotatedRect mark_rotaterect, int roi_height, cv::Mat &ROI_IMG)
{
	/*
	 * 从标记点获取ROI检测区域
	 * 参数：
	 *   - src_img: 待检测图像
	 *   - mark_rotaterect: mark区域旋转矩阵
	 *	 - roi_height：旋转ROI区域的高度，可以调节；
	 * 返回：
	 *   - 无
	 */
	

	//计算roi中心点
	/*cout << mark_rotaterect.center << endl;
	cout << mark_rotaterect.size << endl;
	cout << mark_rotaterect.angle << endl;*/

	cv::Point2f vertices[4];
	mark_rotaterect.points(vertices);
	// 绘制旋转矩形的边框
	//for (int i = 0; i < 4; i++)
	//{
	//	cv::line(src_img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
	//}

	//ROI的旋转矩阵信息；
	cv::RotatedRect roi_rect(cv::Point2f(mark_rotaterect.center.x, (mark_rotaterect.center.y + (mark_rotaterect.size.width / 2) + roi_height/2)),
		cv::Size2f(roi_height, mark_rotaterect.size.height), mark_rotaterect.angle);

	cv::Point2f vertices_roi[4];
	roi_rect.points(vertices_roi);
	// 绘制旋转矩形的边框
	//for (int i = 0; i < 4; i++)
	//{
	//	cv::line(src_img, vertices_roi[i], vertices_roi[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
	//}
	// 对旋转矩阵进行矫正，保持水平
	cv::Mat rotationMatrix = cv::getRotationMatrix2D(roi_rect.center, roi_rect.angle, 1.0);

	cv::Size rectSize = roi_rect.size;
	cv::warpAffine(src_img, ROI_IMG, rotationMatrix, src_img.size(), cv::INTER_CUBIC);
	cv::getRectSubPix(ROI_IMG, rectSize, roi_rect.center, ROI_IMG);

}

//void getroiimg(cv::Mat src, cv::RotatedRect rotate_rect, std::vector<cv::Mat>& ROI);
template<typename _Tp>
int yolov8_onnx(_Tp& task, cv::Mat& img, std::string& model_path)
{
	if (task.ReadModel(model_path, false)) {
		std::cout << "read net ok!" << std::endl;
	}
	else {
		return -1;
	}

	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
	
	std::vector<OutputParams> result;

	if (task.OnnxDetect(img, result)) 
	{	
		//按照之前和周流沟通，只会有一个mark点，所以此处只处理一个
		cv::RotatedRect mark_rect = result[0].rotatedBox;

		cv::Mat ROI_IMG;
		getroifrommark(img, mark_rect, 400, ROI_IMG);

		cv::Mat ROI_IMG90;
		cv::rotate(ROI_IMG, ROI_IMG90, cv::ROTATE_90_CLOCKWISE);

		
		cv::imshow("ROI", ROI_IMG90);
		cv::waitKey(0);

		DrawPred(img, result, task._className, color);
	}
	else
	{
		std::cout << "Detect Failed!" << std::endl;
	}
	system("pause");
	return 0;
}



int main() {

	std::string img_path = "E:\\14\\Image_20241112231427423.bmp";
	std::string model_path_obb = "E:\\14\\best_fpc.onnx";

	cv::Mat src = imread(img_path);
	cv::Mat img = src.clone();
	Yolov8ObbOnnx		task_obb_ort;

	img = src.clone();
	yolov8_onnx(task_obb_ort, img, model_path_obb); 
	return 0;
}
