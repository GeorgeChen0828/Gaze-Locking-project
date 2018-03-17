#pragma once
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <filesystem>
#include <sstream>
#include <fstream>
#include <io.h>
#include <iostream>

using namespace std;
using namespace cv;

static string filepath = "D:\\TrainData\\";
static fstream file;
//§Îª¬¹w´ú¾¹
static dlib::shape_predictor sp;

void ReadData();
bool DoPCA(string filepath, Mat trainingdata, double retainedVariance);
//Mat LoadEyeImageFile(string filepath);

Mat LoadCSVFile(string filepath, string filename);
vector<Mat> ProcessCAMImage(Mat inputMat, dlib::shape_predictor sp);
Mat getEyeImage(Point Lcorner, Point Rcorner, Mat sorceImage);
Mat mergeEyeImage(Mat Leye, Mat Reye);
vector<Mat> ReadImage(int choose, bool storeEyeImage);
Mat subMean(Mat inputMatrix, Mat mean);
Mat calcMean(Mat inputMatrix);