#pragma once
#include <opencv2/core/core.hpp>
#include <fstream>

using namespace std;
using namespace cv;

void writeTrainingData(string filepath,Mat trainingdata, vector<int> vec_label);
void writePCAData(string filepath, Mat eigenValue, Mat eigenVector, Mat mean,int writePosition);