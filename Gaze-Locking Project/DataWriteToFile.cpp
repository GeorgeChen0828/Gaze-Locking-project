#include "DataWriteToFile.h"

fstream file;

void writeTrainingData(string filepath,Mat trainingdata, vector<int> vec_label)
{
	//norm_n1_1(trainingdata);
	//�N�ƭȼg�J�ɮ�
	file.open(filepath+"TrainingDataFile.csv", ios::out);//�Ы��ɮ��x�s�V�m��ơA�Y�s�b�h�M�Ť��e
	for (int i = 0; i<trainingdata.size().height; i++)
	{
		for (int j = 0; j < trainingdata.size().width; j++)
		{
			file << trainingdata.at<double>(i,j) << ",";
		}
		file << "\n";
	}
	file.close();
	file.open(filepath+"TrainingDataLabel.csv",ios::out);//�Ы��ɮ��x�s�V�m��ơA�Y�s�b�h�M�Ť��e
	for (int i = 0; i < vec_label.size(); i++)
	{
		file << vec_label[i] << "\n";
	}
	file.close();
}

void writePCAData(string filepath,Mat eigenValue,Mat eigenVector,Mat mean,int writePosition)
{
	file.open(filepath + "PCA_eigenValues.csv", ios::out);//�x�seigenValue
	for (int i = 0; i<eigenValue.size().height; i++)
	{
		if (i > writePosition) break;
		for (int j = 0; j < eigenValue.size().width; j++)
		{
			file << eigenValue.at<double>(i, j) << "\n";
		}
	}
	file.close();
	file.open(filepath+"PCA_eigenVector.csv", ios::out);//�x�seigenVector
	for (int i = 0; i<eigenVector.size().height; i++)
	{
		if (i > writePosition) break;
		for (int j = 0; j < eigenVector.size().width; j++)
		{
			file << eigenVector.at<double>(i, j) << ",";
		}
		file << "\n";
	}
	file.close();
	file.open(filepath + "PCA_mean.csv", ios::out);//�x�smean
	for (int i = 0; i<mean.size().height; i++)
	{
		for (int j = 0; j < mean.size().width; j++)
		{
			file << mean.at<double>(i, j) << ",";
		}
		file << "\n";
	}
	file.close();
}