#include "ReadTrainingData.h"
#include "DataWriteToFile.h"

using namespace std;

void DataPreProcessing();
void DoLDA();

int main()
{
	DataPreProcessing();//�V�m�e���B�z�A�̫�ݭn���ѱ�

	cout << endl << "�e�m�B�z�w����" << endl;
	system("pause");
	return 0;

	//cout << "Searching eigenVector data file..." << endl;
	//file.open(filepath + "PCA_eigenVector.csv");
	//if (!file)
	//{
	//	cout << "*****�䤣��PCA_eigenVector.csv�ɮ�*****" << endl;
	//	system("pause");
	//	return 0;
	//}
	////Ū���ɮ׸��
	//Mat eigenVectors = LoadEigenVector(filepath);

	//file.close();

	//cout << "Searching LDA data file..." << endl;
	//file.open(filepath + "LDA_File.csv");
	//if (!file)
	//{
	//	cout << "*****�䤣��LDA_File.csv�ɮ�*****" << endl;
	//	system("pause");
	//	return 0;
	//}
	////Ū���ɮ׸��
	//Mat LDA = LoadEigenVector(filepath);

	//=====================�i��WebCAM�Ϥ��P�_=====================
	//step1 : ���oWebCAM�Ϥ�
	//step2 : ��JWebCAM�Ϥ��A���oMergeEyeImage
	//step3 : �NMergeEyeImage���WeigenVector�A����x��
	//step4 : �e�JLDA
	//step5 : �e�JSVM�����A�ÿ�X���G

	//�qshape_predictor_68_face_landmarks.dat���J�w����
	//dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	//ProcessCAMImage(,sp);

	//���JSVM�V�m���
	//float response = model->predict(sampleMat); //�i��w���A��^1 �� -1


	cv::waitKey();
	system("pause");
	return 0;
}

void DataPreProcessing()
{
	//�ˬd�Ϥ��B�z�ɮ׬O�_�s�b
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
		//cout << "���~�G�䤣��V�m�Ϥ��ɮ�!!!" << endl;
	}
	file.close();
	//Ū������CSV��
	//Mat tempdata = LoadEyeImageFile(filepath);
	cout << "Loading eye images data...";
	Mat tempdata = LoadCSVFile(filepath,"TrainingDataFile");
	cout << "finished." << endl;
	Mat trainingdata=Mat::zeros(tempdata.size(),CV_64FC1);
	//�k�@�Ʀ�0~1���ƭ�
	cv::normalize(tempdata, trainingdata, 1, -1, NORM_L2);
	tempdata.release();

	//�ˬdeigenVector�O�_�s�b
	cout << "Searching eigenVector data file..." << endl;
	file.open(filepath + "PCA_eigenVector.csv");
	if (!file)
	{
		try
		{
			cout << "Data not found, reading training data and start PCA analysis" << endl;
			if (!DoPCA(filepath, trainingdata,0.95))
			{
				cout << "PCA �o�Ϳ��~!!" << endl;
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
	//Ū��PCA_eigenVector
	cout << "Loading eigenvectors...";
	Mat eigenVectors = LoadCSVFile(filepath,"PCA_eigenVector");
	cout << "finished." << endl;

	cout << "Loading mean...";
	Mat mean = LoadCSVFile(filepath, "PCA_mean");
	cout << "finished." << endl;

	cout << "Creating project space...";
	Mat projectMat;
	trainingdata = subMean(trainingdata,mean);
	gemm(trainingdata,eigenVectors,1,Mat(),0, projectMat,GEMM_1_T+GEMM_2_T);//�إ߭����᪺trainingdata�ASize=�˥���x�����O�d��
	projectMat = projectMat.t();//��m�� �����O�d��x�˥���
	normalize(projectMat,projectMat,1,-1,NORM_L2);//���s�k�@�ơA�ǳưe�JLDA
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
	//�Ы�SVM����
	Ptr<ml::SVM> model = ml::SVM::create();
	model->setType(ml::SVM::C_SVC);
	model->setKernel(ml::SVM::LINEAR);

	//�]�m�V�m���
	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingdata, ml::ROW_SAMPLE, trainlabel);

	//�V�mSVM
	model->train(tData);


	//model->setKernel(SVM::POLY); //�]�w�֤ߨ��
	//model->setDegree(0.5);
	//model->setGamma(1);
	//model->setCoef0(1);
	//model->setNu(0.5);
	//model->setP(0);
	//model->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));
	//model->setC(C);

}