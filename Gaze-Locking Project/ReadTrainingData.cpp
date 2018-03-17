#include "ReadTrainingData.h"
#include "DataWriteToFile.h"

const int imgSize = 40;//�H40x40������Y��Ϥ�
//string filepath = "D:\\TrainData\\";
std::vector<std::experimental::filesystem::path> pathstring;
std::vector<int> vec_label;
stringstream int2string;
string loadnum, filename;
string path;
char filenamechar[50];
int loadtimes = 0;
char lbl[1];

double multiple;
Mat OImg, cvImg, Img;
dlib::cv_image<uchar> tempImg;
dlib::array2d<uchar> dImg;
std::vector<dlib::rectangle> dets;
std::vector<dlib::full_object_detection> shapes;
//�H�y�˴����A�˴��H�y�~��
dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();


Mat getEyeImage(Point Lcorner, Point Rcorner, Mat sorceImage)
{
	Mat eye, eyeTemp, rotate_mat, mask;
	int rotation = 1;
	double diameter, angle;
	Point center;
	int radius, left, top, width, height;

	//�ŵ������A����ܤ���
	if (Lcorner.y > Rcorner.y) { rotation = -1; }//���ɰw����
	else { rotation = 1; }//�f�ɰw����
	diameter = sqrt(pow(Rcorner.x - Lcorner.x, 2) + pow(std::abs(Rcorner.y - Lcorner.y), 2));//�H��������Ϊ��| sqrt=�ڸ� pow=N����
	radius = diameter / 2;//�b�|
	angle = (acos((Rcorner.x - Lcorner.x) / diameter)*180.0 / dlib::pi)*rotation;//acos �D�X�|�A�ݦA���W180/PI���⦨���סArotation��ܶ��ɰw�ΰf�ɰw����
	center = Point((Lcorner.x + Rcorner.x) / 2, (Lcorner.y + Rcorner.y) / 2);
	//circle(sorceImage,center,1,Scalar(255));
	left = center.x - radius;
	top = center.y - radius;
	width = radius * 2;
	height = radius * 2;
	Rect r(left, top, width, height);
	sorceImage(r).copyTo(eyeTemp);
	//imshow("Sorce Eye", eyeTemp);
	center = Point(radius, radius);
	rotate_mat = getRotationMatrix2D(center, angle, 1);//�]�w�x�}�������I�B���סB�Y��ؤo;
	warpAffine(eyeTemp, eyeTemp, rotate_mat, eyeTemp.size());//��g�ܴ� �Ѽ�1=�ӷ��Ϥ� 2=�ت��a 3=����x�}�]�w 4=�s�x�}�j�p 5=�u�ʮt�� 6=��t���A 7=���v��
															 //imshow("Rotated Eye", eyeTemp);
	mask = Mat(eyeTemp.size(), CV_8U, Scalar(0));
	ellipse(mask, center, Size(radius, radius*0.75), 0, 0, 360, Scalar(255), -1);
	eyeTemp.copyTo(eye, mask);
	return eye;
}

Mat mergeEyeImage(Mat Leye, Mat Reye)
{
	Mat mergeImg, step1, step2;
	double mult;//multiple
	mult = (double)imgSize / Leye.size().width;
	resize(Leye, Leye, Size(0, 0), mult, mult);
	mult = (double)imgSize / Reye.size().width;
	resize(Reye, Reye, Size(0, 0), mult, mult);
	//cout << "scale eye picture" << endl;

	mergeImg = Mat(Leye.size().height, Leye.size().width * 2, CV_8U, Scalar(0)); //80*40���Ϥ�
	step1 = mergeImg(Rect(0, 0, Leye.cols, Leye.rows));//�����ϰ�
	step2 = mergeImg(Rect(mergeImg.size().width - Reye.size().width, 0, Reye.cols, Reye.rows));//�k���ϰ�
																							   //����X��
	Leye.copyTo(step1);
	Reye.copyTo(step2);
	//���h�W�U��t
	int nonzeroTop = 0;
	for (int i = 0; i < mergeImg.size().height; i++)
	{
		for (int j = 0; j < mergeImg.size().width; j++)
		{
			if ((int)mergeImg.at<uchar>(i, j) > 0)
			{
				nonzeroTop = i; break;
			}
		}
		if (nonzeroTop > 0) { break; }
	}
	Mat cropEdge((int)(mergeImg.size().height*0.75), mergeImg.size().width, CV_8U, Scalar(0));
	for (int i = 0; i < cropEdge.size().height; i++)
	{
		Mat rowMat;
		mergeImg.row(i + nonzeroTop).copyTo(cropEdge.row(i));
		//for (int j = 0; j < cropEdge.size().width; j++)
		//{
		//	cout << (int)cropEdge.at<uchar>(i, j) << " ";
		//}
		//cout << endl;
	}
	return cropEdge;
}

vector<Mat> ReadImage(int choose, bool storeEyeImage = false)
{
	vector<Mat> eyeCollects;
	switch (choose)
	{
	case 0:
		cout << "Ū������Ū��" << endl;
		pathstring.push_back("D:\\TrainData\\EyeImage\\0\\0\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\0\\1\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\0\\2\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\0\\3\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\0\\4\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\1\\5\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\1\\6\\");
		pathstring.push_back("D:\\TrainData\\EyeImage\\1\\7\\");
		break;
	case 1:
		cout << "Ū����lŪ��" << endl;
		pathstring.push_back("D:\\TrainData\\0\\0\\");
		pathstring.push_back("D:\\TrainData\\0\\1\\");
		pathstring.push_back("D:\\TrainData\\0\\2\\");
		pathstring.push_back("D:\\TrainData\\0\\3\\");
		pathstring.push_back("D:\\TrainData\\0\\4\\");
		pathstring.push_back("D:\\TrainData\\1\\5\\");
		pathstring.push_back("D:\\TrainData\\1\\6\\");
		pathstring.push_back("D:\\TrainData\\1\\7\\");
		break;
	}
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	//====================
	char eyeclass[1];
	string eyepath = "D:\\TrainData\\EyeImage\\";
	

	std::cout << "Start reading training images..." << endl;
	for (int loadCount = 0; loadCount < pathstring.size(); loadCount++)
	{
		int eyecount = 0;
		path = pathstring[loadCount].string();
		string lblPathString =  pathstring[loadCount].parent_path().string();
		string eyeClassPathString = pathstring[loadCount].parent_path().parent_path().string();
		lbl[0] = lblPathString.at(lblPathString.length() - 1);
		eyeclass[0] = eyeClassPathString.at(eyeClassPathString.length() - 1);

		while (true)
		{
			int2string << loadtimes;
			int2string >> loadnum;

			string charcount(5 - loadnum.length(), '0');
			filename = path + charcount + loadnum + ".jpg";
			strcpy(filenamechar, filename.c_str());
			if ((_access(filenamechar, 0)) != -1)
			{//�ɮץiŪ��
				//=====================Ū���ɮ�=====================
				cout << "Loading image " << filename << "     ";
				switch (choose)
				{
				case 0:
					eyeCollects.push_back(imread(filename, 0));
					vec_label.push_back(atoi(lbl));//�O���V�m��ƪ�����
					cout << endl;
					break;
				case 1:
					OImg = imread(filename, 0);
					//�N�Ϥ��Y�� width=1440 �� height=1080
					multiple = 1.0;
					if (OImg.size().width > OImg.size().height) { multiple = (double)1920 / OImg.size().width; }
					else { multiple = (double)1440 / OImg.size().height; }
					cv::resize(OImg, Img, Size(0, 0), multiple, multiple);

					//�o���ӷL���T
					cv::GaussianBlur(Img, cvImg, Size(3, 3), 0);
					//�Nopencv�Ϥ��ରdlib array2d
					tempImg = cvImg;
					assign_image(dImg, tempImg);

					//=====================�˴��H�y=====================
					dets = detector(dImg);//�q�˴������o�Ϥ��W�Ҧ��H�y���~�حȲM��
					cout << "faces detected: " << dets.size() << endl;//�L�X�˴��쪺�H�y��
					if (dets.size() ==1)
					{
						vec_label.push_back(atoi(lbl));//�O���V�m��ƪ�����

						//�n�Dshape_predictor�i�D�ڭ��˴��쪺�C�i�y�����u
						for (int j = 0; j < dets.size(); ++j)
						{
							dlib::full_object_detection shape = sp(dImg, dets[j]);
							shapes.push_back(shape);//�N68�Ӧ��u�I���O�x�s
						}
						//=====================�^���Ϥ�=====================

						for(int b = 0; b < dets.size(); b++)
						{
							Mat Leye, Reye, mergeEye;
							//��X������m
							//       37  38                  43  44
							//  36    ���]    39        42    ���]    45
							//       41  40                  47  46
							Point LL = Point(shapes[b].part(36).x(), shapes[b].part(36).y());
							Point LR = Point(shapes[b].part(39).x(), shapes[b].part(39).y());
							Point RL = Point(shapes[b].part(42).x(), shapes[b].part(42).y());
							Point RR = Point(shapes[b].part(45).x(), shapes[b].part(45).y());
							//cout << "Got eye corner!" << endl;
							//�ŵ������A����ܤ���
							Leye = getEyeImage(LL, LR, cvImg);
							Reye = getEyeImage(RL, RR, cvImg);
							//imshow("L eye", Leye);
							//imshow("R eye", Reye);
							//cout << "Eye processing finished!" << endl;

							mergeEye = mergeEyeImage(Leye, Reye);
							//cout << "Face " << num << " eye merge finished!" << endl << endl;
							if (storeEyeImage == true)
							{
								stringstream ss1;
								string num1, num2;
								//imshow(filename + "Face " + num + " merge eye", mergeEye);
								ss1 << eyecount;
								ss1 >> num1;
								string a(5 - num1.length(), '0');
								imwrite(eyepath + eyeclass[0] + "\\" + lbl[0] + "\\" + a + num1+ ".jpg", mergeEye);
								ss1.clear();
								eyecount++;
							}
							eyeCollects.push_back(mergeEye);
						}
						OImg.release(); cvImg.release(); Img.release(); dets.clear(); shapes.clear();
					}
					else
					{
						//�˴�����H�y�A�O�����|&�Ϥ��W��
						fstream outfile;
						outfile.open("D:\\TrainData\\EyeImage\\FaceNotFound.txt", ios::app);
						outfile << filename << "\n";
						outfile.close();
					}
					break;
				}
				int2string.clear();
				loadtimes++;
			}
			else
			{//�ɮפ��s�b �� �ɮ׵L�k�ϥ�
			 //cout << "�ɮפ��s�b �� �ɮ׵L�k�ϥ�!!" << endl;
				loadtimes = 0;
				int2string.clear();
				break;
			}
		}
	}
	return eyeCollects;
}

void ReadData()
{
	try
	{
		cout << "=========================���b���J�Ϥ�=========================" << endl;
		std::vector<Mat> eyeCollects = ReadImage(1,true);//0=Ū����   1=Ū��ϵ�����
		
		int imgWidth = eyeCollects[0].size().width;
		int imgHeight = eyeCollects[0].size().height;

		cout << "====================�Ϥ����J�����A�ഫ�x�}====================" << endl;
		
		int count = 0;
		//�N�Ҧ��Ϥ��ন (width*height)*�Ϥ��� ���j�x�}
		Mat trainingdata(imgWidth*imgHeight, eyeCollects.size(), CV_64FC1, Scalar(0.0));
		for (int eyepics = 0; eyepics < eyeCollects.size(); eyepics++)
		{
			eyeCollects[eyepics].reshape(0, imgWidth*imgHeight).copyTo(trainingdata.col(eyepics));
			//for (int i = 0; i < imgHeight; i++)
			//{
			//	for (int j = 0; j < imgWidth; j++)
			//	{
			//		trainingdata.at<double>(count, eyepics) = (int)eyeCollects[eyepics].at<uchar>(i, j);
			//		count++;
			//	}
			//}
			count = 0;
		}

		cout << "�ഫ�����A���b�g�JCSV�ɮ�..." ;
		writeTrainingData(filepath,trainingdata,vec_label);//�N�V�m�Ϥ���Ƽg�Jcsv��
		cout << "finished!" << endl;

		cout << "�Ϥ��B�z�����I" << endl << endl;

		pathstring.clear();
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
		system("pause");
	}
}

bool DoPCA(string filepath,Mat trainingdata,double retainedVariance)
{
	try
	{
		cout << "===================PCA analysis===================" << endl;
		cout << "caculat mean...";
		Mat mean = calcMean(trainingdata);
		cout << "finished" << endl;
		cout << "get covariance matrix...";
		trainingdata = subMean(trainingdata, mean);//��hmean
		Mat mat_C;//Covariance Matrix
		gemm(trainingdata, trainingdata, 1, Mat(), 0, mat_C, GEMM_2_T);//�Ѽ�"GEMM_2T"�N����m��2�ӯx�}
		mat_C = mat_C / mat_C.size().height;//�@�ܲ��x�}(���t�x�})=��x�}x��m��x�} / rows
		cout << "finished" << endl;
		
		cout << "caculating eigenvalues and eigenvectors......";

		Mat eigenvalues, eigenvectors;
		eigen(mat_C, eigenvalues, eigenvectors);
		cout << "finished" << endl;
		cout << "mean.size=" << mean.size() << endl;
		cout << "eigenvectors.size=" << eigenvectors.size() << endl;
		cout << "eigenvalues.size=" << eigenvalues.size() << endl;

		double sumEvalue = 0.0;
		for (int i = 0; i < eigenvalues.size().height; i++)
		{
			sumEvalue += eigenvalues.at<double>(i, 0);
		}
		double SNR = 0.0; double value = 0.0; int wirtePosition = 0;
		for (wirtePosition = 0; wirtePosition < eigenvalues.size().height; wirtePosition++)
		{
			value += eigenvalues.at<double>(wirtePosition, 0);
			SNR = value / sumEvalue;
			if (SNR >= retainedVariance)
			{
				break;
			}
		}
		if (wirtePosition == 0) wirtePosition = 1;
		//Store the eigenvalues and eigenvectors
		writePCAData(filepath, eigenvalues, eigenvectors, mean,wirtePosition);

		cout << "===================PCA finished===================" << endl;
		return true;
	}
	catch(Exception ex)
	{
		cout << ex.what() << endl;
		system("pause");
		return false;
	}
}

Mat LoadCSVFile(string filepath,string filename)
{
	Mat result;
	string line;
	vector<double> readValue;
	vector<Mat> tempMat;
	int rows = 0; int cols = 0;
	file.open(filepath + filename+".csv");
	while (getline(file, line))
	{
		istringstream tempLine(line); // string �ഫ�� stream
		string data;
		while (getline(tempLine, data, ',')) //Ū��Ū��r��
		{
			readValue.push_back(atof(data.c_str()));  //string �ഫ���Ʀr
		}
		Mat readLine = Mat(readValue);//vector�ন�x�}(column�Φ�)
		tempMat.push_back(readLine.t());
		readValue.clear();
	}
	file.close();
	rows = tempMat.size();
	cols = tempMat[0].size().width;
	if (rows == 0 || cols == 0) { cout << "row or column number can not be zero!!" << endl; return Mat(); }
	result = Mat(rows, cols, CV_64F, Scalar(0.0));
	for (int i = 0; i < rows; i++)
	{
		tempMat[i].copyTo(result.row(i));
	}
	tempMat.clear();
	return result;
}
//
//Mat LoadEigenVector(string filepath)
//{
//	cout << "Loading eigenvectors...";
//	Mat eigenVectors;
//	string line;
//	vector<double> readValue;
//	vector<Mat> tempMat;
//	int rows = 0; int cols = 0;
//	file.open(filepath+"PCA_eigenVector.csv");
//	while (getline(file, line))
//	{
//		istringstream tempLine(line); // string �ഫ�� stream
//		string data;
//		while (getline(tempLine, data, ',')) //Ū��Ū��r��
//		{
//			readValue.push_back(atof(data.c_str()));  //string �ഫ���Ʀr
//		}
//		Mat readLine = Mat(readValue);//vector�ন�x�}(column�Φ�)
//		tempMat.push_back(readLine.t());
//		readValue.clear();
//	}
//	file.close();
//	cout << "finished." << endl;
//	cout << "Creating eigenVectors data matrix...";
//	rows = tempMat.size();
//	cols = tempMat[0].size().width;
//	if (rows == 0 || cols == 0) { cout << "row or column number can not be zero!!" << endl; return Mat(); }
//	eigenVectors=Mat(rows, cols, CV_64F, Scalar(0.0));
//	for (int i = 0; i < rows; i++)
//	{
//		tempMat[i].copyTo(eigenVectors.row(i));
//	}
//	tempMat.clear();
//	cout << "finished." << endl;
//	return eigenVectors;
//}

//Mat LoadEyeImageFile(string filepath)
//{
//	cout << "Loading eye images data...";
//	string line;
//	vector<double> readValue;
//	vector<Mat> fullMat;
//	int rows = 0; int cols = 0;
//	file.open(filepath+"TrainingDataFile.csv");
//	int count = 0;
//	while (getline(file, line))
//	{
//		count++;
//		istringstream tempLine(line); // string �ഫ�� stream
//		string data;
//		while (getline(tempLine, data, ',')) //Ū��Ū��r��
//		{
//			readValue.push_back(atof(data.c_str()));  //string �ഫ���Ʀr
//		}
//		Mat readRow(1, readValue.size(), CV_64F, Scalar(0.0));
//		for (int i = 0; i < readValue.size(); i++)
//		{
//			readRow.at<double>(0, i) = readValue[i];
//		}
//		fullMat.push_back(readRow.clone());
//		readValue.clear();
//	}
//	file.close();
//	cout << "finished." << endl;
//
//	cout << "Creating training data matrix...";
//	rows = fullMat.size();
//	cols = fullMat[0].size().width;
//	Mat trainingdata(rows, cols, CV_64F, Scalar(0.0));
//	for (int i = 0; i < rows; i++)
//	{
//		fullMat[i].copyTo(trainingdata.row(i));
//	}
//	fullMat.clear();
//	readValue.clear();
//	cout << "finished." << endl;
//	return trainingdata;
//}

vector<Mat> ProcessCAMImage(Mat inputMat, dlib::shape_predictor sp)
{
	std::vector<Mat> eyeCollects;
	//=====================�Ϥ��w�B�z=====================
	OImg = inputMat;
	//�N�Ϥ��Y�� width=1440 �� height=1080
	multiple = 1.0;
	if (OImg.size().width > OImg.size().height) { multiple = (double)1920 / OImg.size().width; }
	else { multiple = (double)1440 / OImg.size().height; }
	cv::resize(OImg, Img, Size(0, 0), multiple, multiple);

	//�o���ӷL���T
	cv::GaussianBlur(Img, cvImg, Size(3, 3), 0);
	//�Nopencv�Ϥ��ରdlib array2d
	tempImg = cvImg;
	assign_image(dImg, tempImg);

	//=====================�˴��H�y=====================
	dets = detector(dImg);//�q�˴������o�Ϥ��W�Ҧ��H�y���~�حȲM��
	cout << "faces detected: " << dets.size() << endl;//�L�X�˴��쪺�H�y��
	//�n�Dshape_predictor�i�D�ڭ��˴��쪺�C�i�y�����u
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		dlib::full_object_detection shape = sp(dImg, dets[j]);
		shapes.push_back(shape);//�N68�Ӧ��u�I���O�x�s
	}

	//=====================�^���Ϥ�=====================
	for (int i = 0; i < dets.size(); i++)
	{
		//��X������m
		//       37  38                  43  44
		//  36    ���]    39        42    ���]    45
		//       41  40                  47  46

		Mat Leye, Reye, mergeEye;
		Point LL = Point(shapes[i].part(36).x(), shapes[i].part(36).y());
		Point LR = Point(shapes[i].part(39).x(), shapes[i].part(39).y());
		Point RL = Point(shapes[i].part(42).x(), shapes[i].part(42).y());
		Point RR = Point(shapes[i].part(45).x(), shapes[i].part(45).y());

		//�ŵ������A����ܤ���
		Leye = getEyeImage(LL, LR, cvImg);
		Reye = getEyeImage(RL, RR, cvImg);

		mergeEye = mergeEyeImage(Leye, Reye);
		eyeCollects.push_back(mergeEye);
	}
	OImg.release(); cvImg.release(); Img.release(); dets.clear(); shapes.clear();
	return eyeCollects;
}

Mat subMean(Mat inputMatrix, Mat mean)
{
	for (int i = 0; i < inputMatrix.size().height; i++)
	{
		for (int j = 0; j < inputMatrix.size().width; j++)
		{
			inputMatrix.at<double>(i, j) = inputMatrix.at<double>(i, j) - mean.at<double>(0, j);
		}
	}
	return inputMatrix;
}

Mat calcMean(Mat inputMatrix)
{
	Mat result=Mat::zeros(1, inputMatrix.size().width,CV_64FC1);
	for (int i = 0; i < inputMatrix.size().height; i++)
	{
		for (int j = 0; j < inputMatrix.size().width; j++)
		{
			result.at<double>(0, j) = result.at<double>(0, j) + inputMatrix.at<double>(i, j);
		}
	}
	result = result / inputMatrix.size().height;
	return result;
}


