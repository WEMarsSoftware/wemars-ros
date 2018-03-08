
#include <sl/Camera.hpp>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <SaveDepth.hpp>
#include <iostream>
#include <stdio.h>

#include <algorithm>

cv::Mat slMat2cvMat(sl::Mat& input);
cv::Point findTennisBalls(cv::Mat im);

int main(int argc, char **argv) {

	// Create a ZED camera object
	sl::Camera zed;

	// Set configuration parameters
	sl::InitParameters init_params;
	init_params.camera_resolution = sl::RESOLUTION_HD720;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	init_params.coordinate_units = sl::UNIT_METER;
	init_params.camera_fps = 60;

	// Open the camera
	sl::ERROR_CODE err = zed.open(init_params);
	if (err != sl::SUCCESS) {
		printf("%s\n", errorCode2str(err).c_str());
		zed.close();
		return 1; // Quit if an error occurred
	}

	// Set runtime parameters after opening the camera
	sl::RuntimeParameters runtime_parameters;
	runtime_parameters.sensing_mode = sl::SENSING_MODE_STANDARD;

	// Prepare new image size to retrieve half-resolution images
	sl::Resolution image_size = zed.getResolution();
	int new_width = image_size.width / 2;
	int new_height = image_size.height / 2;

	// To share data between sl::Mat and cv::Mat, use slMat2cvMat()
	// Only the headers and pointer to the sl::Mat are copied, not the data itself
	sl::Mat image_zed(new_width, new_height, sl::MAT_TYPE_8U_C4);
	cv::Mat image_ocv = slMat2cvMat(image_zed);
	sl::Mat depth_image_zed(new_width, new_height, sl::MAT_TYPE_8U_C4);
	cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
	sl::Mat normal_image_zed(new_width, new_height, sl::MAT_TYPE_8U_C4);
	cv::Mat normal_image_ocv = slMat2cvMat(normal_image_zed);

	//ball detection vars
	int Lmin = 0, amin = 95, bmin = 155, Lmax = 255, amax = 117, bmax = 213;
	cv::Mat mask = image_ocv;
	std::vector<cv::Vec3f> circles;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
	cv::Mat canny_output;
	cv::Mat stats;

	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	//std::vector<std::vector<cv::Point>> centroids;
	std::vector<std::vector<cv::Point> > labels;
	cv::Mat label;
	cv::Mat seeMyLabels;
	sl::Mat depth,point_cloud;
	cv::Mat centroids;

	int thres1=100, thres2=200, apSize=3;
	int L2grad = false;

	// Loop until 'q' is pressed
	char key = ' ';
	while (key != 'q') {

		if (zed.grab(runtime_parameters) == sl::SUCCESS) {

			// Retrieve the left image, depth image in half-resolution
			zed.retrieveImage(image_zed, sl::VIEW_LEFT, sl::MEM_CPU, new_width, new_height);
			zed.retrieveImage(depth_image_zed, sl::VIEW_DEPTH, sl::MEM_CPU, new_width, new_height);

			// Retrieve the RGBA point cloud in half-resolution
			// To learn how to manipulate and display point clouds, see Depth Sensing sample
			zed.retrieveMeasure(depth, sl::MEASURE_DEPTH, sl::MEM_CPU,new_width,new_height);
			zed.retrieveMeasure(point_cloud, sl::MEASURE_XYZRGBA, sl::MEM_CPU, new_width, new_height);

			//filter out yellow
			cv::cvtColor(image_ocv,mask,cv::COLOR_BGR2Lab);
			//cv::medianBlur(mask,mask, 11);
			cv::inRange(mask,cv::Scalar(Lmin,amin,bmin),cv::Scalar(Lmax, amax, bmax), mask);
			//cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
			//cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)),cv::Point(-1,-1),5);
			//cv::medianBlur(mask, mask, 5);
			cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);
			cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

			int n_labels = cv::connectedComponentsWithStats(mask, label, stats, centroids,8, CV_32S);

			float radius;

			for (int i = 1; i < n_labels; i++)
			{
				int x = centroids.at<double>(cv::Point(0, i));
				int y = centroids.at<double>(cv::Point(1, i));
				cv::Point center = cv::Point(x, y);
				//cv::minEnclosingCircle(contours[i], center, radius);

				cv::circle(image_ocv,center,5, cv::Scalar(0, 255, 255), 2);

				sl::float4 point_cloud_value;

				point_cloud.getValue(center.x, center.y, &point_cloud_value);

				//float distance = sqrt(point_cloud_value.x*point_cloud_value.x + point_cloud_value.y*point_cloud_value.y + point_cloud_value.z*point_cloud_value.z);
				std::string point_cloud_z;

				if(isnan(point_cloud_value.z))
					point_cloud_z = "?";
				else
					point_cloud_z = std::to_string(point_cloud_value.z);

				//write descriptive text
				cv::putText(image_ocv, "x:" + std::to_string(center.x), center+cv::Point(0,-20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
				cv::putText(image_ocv, "y:" + std::to_string(center.y), center, CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
				cv::putText(image_ocv, "z:" + point_cloud_z, center+cv::Point(0,20), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));

			}

			/*cv::normalize(label, seeMyLabels, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::imshow("Labels", seeMyLabels);*/

			// Display image and depth using cv:Mat which share sl:Mat data
			cv::imshow("Image", image_ocv);
			cv::imshow("Depth", depth_image_ocv);
			//cv::imshow("Normals", normal_image_ocv);
			cv::imshow("Mask", mask);
			//cv::imshow("Canny", canny_output);
			//cv::imshow("labelled", labelled);

			//outputVideo << image_ocv;

			//outputVideo.write(image_ocv);

			// Handle key event
			key = cv::waitKey(10);
			processKeyEvent(zed, key);
		}
	}
	//outputVideo.release();
	zed.close();
	return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

cv::Point findTennisBalls(cv::Mat im) {
	return cv::Point(0,0);
}
