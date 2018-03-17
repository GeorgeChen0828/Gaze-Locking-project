#include "ReadTrainingData.h"
#include "DataWriteToFile.h"

const int imgSize = 40;//以40x40為基準縮放圖片
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
//人臉檢測器，檢測人臉外框
dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();


Mat getEyeImage(Point Lcorner, Point Rcorner, Mat sorceImage)
{
	Mat eye, eyeTemp, rotate_mat, mask;
	int rotation = 1;
	double diameter, angle;
	Point center;
	int radius, left, top, width, height;

	//剪裁眼睛，旋轉至水平
	if (Lcorner.y > Rcorner.y) { rotation = -1; }//順時針旋轉
	else { rotation = 1; }//逆時針旋轉
	diameter = sqrt(pow(Rcorner.x - Lcorner.x, 2) + pow(std::abs(Rcorner.y - Lcorner.y), 2));//以算斜邊當正方形直徑 sqrt=根號 pow=N次方
	radius = diameter / 2;//半徑
	angle = (acos((Rcorner.x - Lcorner.x) / diameter)*180.0 / dlib::pi)*rotation;//acos 求出徑，需再乘上180/PI換算成角度，rotation表示順時針或逆時針旋轉
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
	rotate_mat = getRotationMatrix2D(center, angle, 1);//設定矩陣旋轉基準點、角度、縮放尺寸;
	warpAffine(eyeTemp, eyeTemp, rotate_mat, eyeTemp.size());//仿射變換 參數1=來源圖片 2=目的地 3=旋轉矩陣設定 4=新矩陣大小 5=線性差值 6=邊緣型態 7=補償色
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

	mergeImg = Mat(Leye.size().height, Leye.size().width * 2, CV_8U, Scalar(0)); //80*40的圖片
	step1 = mergeImg(Rect(0, 0, Leye.cols, Leye.rows));//左眼區域
	step2 = mergeImg(Rect(mergeImg.size().width - Reye.size().width, 0, Reye.cols, Reye.rows));//右眼區域
																							   //執行合併
	Leye.copyTo(step1);
	Reye.copyTo(step2);
	//裁去上下邊緣
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
		cout << "讀取眼睛讀片" << endl;
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
		cout << "讀取原始讀片" << endl;
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
			{//檔案可讀取
				//=====================讀取檔案=====================
				cout << "Loading image " << filename << "     ";
				switch (choose)
				{
				case 0:
					eyeCollects.push_back(imread(filename, 0));
					vec_label.push_back(atoi(lbl));//記錄訓練資料的標籤
					cout << endl;
					break;
				case 1:
					OImg = imread(filename, 0);
					//將圖片縮放成 width=1440 或 height=1080
					multiple = 1.0;
					if (OImg.size().width > OImg.size().height) { multiple = (double)1920 / OImg.size().width; }
					else { multiple = (double)1440 / OImg.size().height; }
					cv::resize(OImg, Img, Size(0, 0), multiple, multiple);

					//濾除細微雜訊
					cv::GaussianBlur(Img, cvImg, Size(3, 3), 0);
					//將opencv圖片轉為dlib array2d
					tempImg = cvImg;
					assign_image(dImg, tempImg);

					//=====================檢測人臉=====================
					dets = detector(dImg);//從檢測器取得圖片上所有人臉的外框值清單
					cout << "faces detected: " << dets.size() << endl;//印出檢測到的人臉數
					if (dets.size() ==1)
					{
						vec_label.push_back(atoi(lbl));//記錄訓練資料的標籤

						//要求shape_predictor告訴我們檢測到的每張臉的曲線
						for (int j = 0; j < dets.size(); ++j)
						{
							dlib::full_object_detection shape = sp(dImg, dets[j]);
							shapes.push_back(shape);//將68個曲線點分別儲存
						}
						//=====================擷取圖片=====================

						for(int b = 0; b < dets.size(); b++)
						{
							Mat Leye, Reye, mergeEye;
							//找出眼睛位置
							//       37  38                  43  44
							//  36    眼珠    39        42    眼珠    45
							//       41  40                  47  46
							Point LL = Point(shapes[b].part(36).x(), shapes[b].part(36).y());
							Point LR = Point(shapes[b].part(39).x(), shapes[b].part(39).y());
							Point RL = Point(shapes[b].part(42).x(), shapes[b].part(42).y());
							Point RR = Point(shapes[b].part(45).x(), shapes[b].part(45).y());
							//cout << "Got eye corner!" << endl;
							//剪裁眼睛，旋轉至水平
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
						//檢測不到人臉，記錄路徑&圖片名稱
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
			{//檔案不存在 或 檔案無法使用
			 //cout << "檔案不存在 或 檔案無法使用!!" << endl;
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
		cout << "=========================正在載入圖片=========================" << endl;
		std::vector<Mat> eyeCollects = ReadImage(1,true);//0=讀眼睛   1=讀原圖裁眼睛
		
		int imgWidth = eyeCollects[0].size().width;
		int imgHeight = eyeCollects[0].size().height;

		cout << "====================圖片載入完成，轉換矩陣====================" << endl;
		
		int count = 0;
		//將所有圖片轉成 (width*height)*圖片數 的大矩陣
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

		cout << "轉換完畢，正在寫入CSV檔案..." ;
		writeTrainingData(filepath,trainingdata,vec_label);//將訓練圖片資料寫入csv檔
		cout << "finished!" << endl;

		cout << "圖片處理完成！" << endl << endl;

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
		trainingdata = subMean(trainingdata, mean);//減去mean
		Mat mat_C;//Covariance Matrix
		gemm(trainingdata, trainingdata, 1, Mat(), 0, mat_C, GEMM_2_T);//參數"GEMM_2T"代表轉置第2個矩陣
		mat_C = mat_C / mat_C.size().height;//共變異矩陣(協方差矩陣)=原矩陣x轉置後矩陣 / rows
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
		istringstream tempLine(line); // string 轉換成 stream
		string data;
		while (getline(tempLine, data, ',')) //讀檔讀到逗號
		{
			readValue.push_back(atof(data.c_str()));  //string 轉換成數字
		}
		Mat readLine = Mat(readValue);//vector轉成矩陣(column形式)
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
//		istringstream tempLine(line); // string 轉換成 stream
//		string data;
//		while (getline(tempLine, data, ',')) //讀檔讀到逗號
//		{
//			readValue.push_back(atof(data.c_str()));  //string 轉換成數字
//		}
//		Mat readLine = Mat(readValue);//vector轉成矩陣(column形式)
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
//		istringstream tempLine(line); // string 轉換成 stream
//		string data;
//		while (getline(tempLine, data, ',')) //讀檔讀到逗號
//		{
//			readValue.push_back(atof(data.c_str()));  //string 轉換成數字
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
	//=====================圖片預處理=====================
	OImg = inputMat;
	//將圖片縮放成 width=1440 或 height=1080
	multiple = 1.0;
	if (OImg.size().width > OImg.size().height) { multiple = (double)1920 / OImg.size().width; }
	else { multiple = (double)1440 / OImg.size().height; }
	cv::resize(OImg, Img, Size(0, 0), multiple, multiple);

	//濾除細微雜訊
	cv::GaussianBlur(Img, cvImg, Size(3, 3), 0);
	//將opencv圖片轉為dlib array2d
	tempImg = cvImg;
	assign_image(dImg, tempImg);

	//=====================檢測人臉=====================
	dets = detector(dImg);//從檢測器取得圖片上所有人臉的外框值清單
	cout << "faces detected: " << dets.size() << endl;//印出檢測到的人臉數
	//要求shape_predictor告訴我們檢測到的每張臉的曲線
	for (unsigned long j = 0; j < dets.size(); ++j)
	{
		dlib::full_object_detection shape = sp(dImg, dets[j]);
		shapes.push_back(shape);//將68個曲線點分別儲存
	}

	//=====================擷取圖片=====================
	for (int i = 0; i < dets.size(); i++)
	{
		//找出眼睛位置
		//       37  38                  43  44
		//  36    眼珠    39        42    眼珠    45
		//       41  40                  47  46

		Mat Leye, Reye, mergeEye;
		Point LL = Point(shapes[i].part(36).x(), shapes[i].part(36).y());
		Point LR = Point(shapes[i].part(39).x(), shapes[i].part(39).y());
		Point RL = Point(shapes[i].part(42).x(), shapes[i].part(42).y());
		Point RR = Point(shapes[i].part(45).x(), shapes[i].part(45).y());

		//剪裁眼睛，旋轉至水平
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


