#include "ReadTrainingData.h"
#include "DataWriteToFile.h"

using namespace std;

void DataPreProcessing();
void DoLDA();

int main()
{
	DataPreProcessing();//訓練前期處理，最後需要註解掉

	cout << endl << "前置處理已完成" << endl;
	system("pause");
	return 0;

	//cout << "Searching eigenVector data file..." << endl;
	//file.open(filepath + "PCA_eigenVector.csv");
	//if (!file)
	//{
	//	cout << "*****找不到PCA_eigenVector.csv檔案*****" << endl;
	//	system("pause");
	//	return 0;
	//}
	////讀取檔案資料
	//Mat eigenVectors = LoadEigenVector(filepath);

	//file.close();

	//cout << "Searching LDA data file..." << endl;
	//file.open(filepath + "LDA_File.csv");
	//if (!file)
	//{
	//	cout << "*****找不到LDA_File.csv檔案*****" << endl;
	//	system("pause");
	//	return 0;
	//}
	////讀取檔案資料
	//Mat LDA = LoadEigenVector(filepath);

	//=====================進行WebCAM圖片判斷=====================
	//step1 : 取得WebCAM圖片
	//step2 : 輸入WebCAM圖片，取得MergeEyeImage
	//step3 : 將MergeEyeImage乘上eigenVector，降成x維
	//step4 : 送入LDA
	//step5 : 送入SVM分類，並輸出結果

	//從shape_predictor_68_face_landmarks.dat載入預測器
	//dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	//ProcessCAMImage(,sp);

	//載入SVM訓練資料
	//float response = model->predict(sampleMat); //進行預測，返回1 或 -1


	cv::waitKey();
	system("pause");
	return 0;
}

void DataPreProcessing()
{
	//檢查圖片處理檔案是否存在
	cout << "Searching training images data file..." << endl;
	file.open(filepath + "TrainingDataFile.csv");
	if (!file)
	{
		try
		{
			file.close();
			cout << "Data not found, reading training images and creat data file." << endl;
			ReadData();
		}
		catch (exception& e)
		{
			cout << "\nexception thrown!" << endl;
			cout << e.what() << endl;
		}
		//cout << "錯誤：找不到訓練圖片檔案!!!" << endl;
	}
	file.close();
	//讀取眼睛CSV檔
	//Mat tempdata = LoadEyeImageFile(filepath);
	cout << "Loading eye images data...";
	Mat tempdata = LoadCSVFile(filepath,"TrainingDataFile");
	cout << "finished." << endl;
	Mat trainingdata=Mat::zeros(tempdata.size(),CV_64FC1);
	//歸一化成0~1的數值
	cv::normalize(tempdata, trainingdata, 1, -1, NORM_L2);
	tempdata.release();

	//檢查eigenVector是否存在
	cout << "Searching eigenVector data file..." << endl;
	file.open(filepath + "PCA_eigenVector.csv");
	if (!file)
	{
		try
		{
			cout << "Data not found, reading training data and start PCA analysis" << endl;
			if (!DoPCA(filepath, trainingdata,0.95))
			{
				cout << "PCA 發生錯誤!!" << endl;
				system("pause");
				return;
			}
		}
		catch (exception& e)
		{
			cout << "\nexception thrown!" << endl;
			cout << e.what() << endl;
			system("pause");
			return;
		}
	}
	file.close();
	//讀取PCA_eigenVector
	cout << "Loading eigenvectors...";
	Mat eigenVectors = LoadCSVFile(filepath,"PCA_eigenVector");
	cout << "finished." << endl;

	cout << "Loading mean...";
	Mat mean = LoadCSVFile(filepath, "PCA_mean");
	cout << "finished." << endl;

	cout << "Creating project space...";
	Mat projectMat;
	trainingdata = subMean(trainingdata,mean);
	gemm(trainingdata,eigenVectors,1,Mat(),0, projectMat,GEMM_1_T+GEMM_2_T);//建立降維後的trainingdata，Size=樣本數x像素保留數
	projectMat = projectMat.t();//轉置成 像素保留數x樣本數
	normalize(projectMat,projectMat,1,-1,NORM_L2);//重新歸一化，準備送入LDA
	cout << "finished." << endl;
}

void DoLDA(Mat trainingdata,Mat trainlabel)
{
	cout << "=======================LDA=======================" << endl;
	Mat LDAmat = trainingdata.t();
	LDA ll(LDAmat, trainlabel);
	Mat llevalue = ll.eigenvalues();
	Mat llevector = ll.eigenvectors();
	cout << "LDA eigenvalues size=" << llevalue.size()<< endl;
	cout << "LDA eigenvectors size=" << llevector.size() << endl;
	ll.save(".\\LDA.xml");
}

void DoSVM(Mat trainingdata,Mat trainlabel)
{
	//創建SVM物件
	Ptr<ml::SVM> model = ml::SVM::create();
	model->setType(ml::SVM::C_SVC);
	model->setKernel(ml::SVM::LINEAR);

	//設置訓練資料
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingdata, ml::ROW_SAMPLE, trainlabel);

	//訓練SVM
	model->train(tData);


	//model->setKernel(SVM::POLY); //設定核心函數
	//model->setDegree(0.5);
	//model->setGamma(1);
	//model->setCoef0(1);
	//model->setNu(0.5);
	//model->setP(0);
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));
	//model->setC(C);

}