#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vector>
#include <iomanip>
using namespace std;
using namespace cv;

//This program use SIFT and SURF algorithm to extract interest point
//And another algorithm is use FAST to detect keypoint and use SIFT to extract descriptor from key points

int main(){

	char * filename = new char[100];
	Mat input;
	vector<KeyPoint> TrainKeypointsSIFT, TrainKeypointsSURF, TrainKeypointsFAST;
	Mat descriptorSIFT, descriptorSURF, descriptorFAST;

	SiftFeatureDetector  detectorSIFT;
	FastFeatureDetector detectorFAST;
	SurfFeatureDetector detectorSURF;
	SiftDescriptorExtractor extractorSIFT;
	SurfDescriptorExtractor extractorSURF;
	SiftDescriptorExtractor extractorFAST;
	//creat unclusterfeatures store descriptor for each image
	Mat UnclusteredfeaturesSIFT,UnclusteredfeaturesSURF,UnclusteredfeaturesFAST;

	//input 100 image, total 5 class and each class have 20 images, 10 for training, 10 for testing
	cout << "Loading traiing image to build dictionary.\n";
	for (int i = 1; i<101; i+=2){
		//open the file
		sprintf(filename, "Training\\Im (%i).jpg", i);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); 

		//sift and SURF detector
		detectorSIFT.detect(input, TrainKeypointsSIFT);
		detectorSURF.detect(input, TrainKeypointsSURF);
		detectorFAST.detect(input, TrainKeypointsFAST);
		extractorSIFT.compute(input, TrainKeypointsSIFT, descriptorSIFT);
		extractorSURF.compute(input, TrainKeypointsSURF, descriptorSURF);
		extractorSIFT.compute(input, TrainKeypointsFAST, descriptorFAST);
		descriptorSIFT.convertTo(descriptorSIFT, CV_32F);
		descriptorSURF.convertTo(descriptorSURF, CV_32F);
		descriptorFAST.convertTo(descriptorFAST, CV_32F);
		//put the all feature descriptors in a Unclusteredfeatures 
		UnclusteredfeaturesSIFT.push_back(descriptorSIFT);
		UnclusteredfeaturesSURF.push_back(descriptorSURF);
		UnclusteredfeaturesFAST.push_back(descriptorFAST);
	}

	//Construct BOWKMeansTrainer
	//we random set there are 20 visual words
	int DictionarySize = 100;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);

	//Create the BoW trainer
	BOWKMeansTrainer bowTrainer(DictionarySize, tc, 3, KMEANS_PP_CENTERS);
	Mat dictionarySIFT = bowTrainer.cluster(UnclusteredfeaturesSIFT);
	Mat dictionarySURF = bowTrainer.cluster(UnclusteredfeaturesSURF); 
	Mat dictionaryFAST = bowTrainer.cluster(UnclusteredfeaturesFAST);
	//create BoF (or BoW) descriptor extractor
	Ptr<DescriptorMatcher> SIFTmatcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> SIFTextractor(new SiftDescriptorExtractor);
	Ptr<DescriptorMatcher> SURFmatcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> SURFextractor(new SurfDescriptorExtractor);
	Ptr<DescriptorMatcher> FASTmatcher(new FlannBasedMatcher);
	Ptr<DescriptorExtractor> FASTextractor(new SiftDescriptorExtractor);
	BOWImgDescriptorExtractor bowDESIFT(SIFTextractor, SIFTmatcher);
	BOWImgDescriptorExtractor bowDESURF(SURFextractor, SURFmatcher);
	BOWImgDescriptorExtractor bowDEFAST(FASTextractor, FASTmatcher);

	//Set the dictionary with the vocabulary we created in the first step
	bowDESIFT.setVocabulary(dictionarySIFT);
	bowDESURF.setVocabulary(dictionarySURF);
	bowDEFAST.setVocabulary(dictionaryFAST);

	//use training data to get the average of histagram of training data
	Mat labels(0, 1, CV_32FC1);
	Mat trainingDataSIFT(0, DictionarySize, CV_32FC1);
	Mat trainingDataSURF(0, DictionarySize, CV_32FC1);
	Mat trainingDataFAST(0, DictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint1SIFT, keypoint1SURF, keypoint1FAST;
	Mat bowDescriptor1SIFT, bowDescriptor1SURF, bowDescriptor1FAST;
	//extracting histogram in the form of bow for each image 
	cout << "Loading training image to get training data histagram according to dictionary.\n";
	for (int j = 1; j < 6; j++)
		for (int i = 1; i <= 20; i+=2){
		sprintf(filename, "Training\\Im (%i).jpg", i + (j - 1) * 20);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		detectorSIFT.detect(input, keypoint1SIFT);
		detectorSURF.detect(input, keypoint1SURF);
		detectorFAST.detect(input, keypoint1FAST);
		bowDESIFT.compute(input, keypoint1SIFT, bowDescriptor1SIFT);
		bowDESURF.compute(input, keypoint1SURF, bowDescriptor1SURF);
		bowDEFAST.compute(input, keypoint1FAST, bowDescriptor1FAST);
		trainingDataSIFT.push_back(bowDescriptor1SIFT);
		trainingDataSURF.push_back(bowDescriptor1SURF);
		trainingDataFAST.push_back(bowDescriptor1FAST);
		labels.push_back((float)j);
	}

	//Setting up SVM parameters
	CvSVMParams params;
	params.kernel_type = CvSVM::RBF;
	params.svm_type = CvSVM::C_SVC;
	params.gamma = 0.50625000000000009;
	params.C = 312.50000000000000;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);

	//training SVM
	CvSVM svmSIFT(trainingDataSIFT, labels, cv::Mat(), cv::Mat(), params);
	CvSVM svmSURF(trainingDataSURF, labels, cv::Mat(), cv::Mat(), params);
	CvSVM svmFAST(trainingDataFAST, labels, cv::Mat(), cv::Mat(), params);

	cout << "Training SVM classifier.\n";

	//record test image value
	Mat groundTruth(0, 1, CV_32FC1);
	Mat evalDataSIFT(0, DictionarySize, CV_32FC1);
	Mat evalDataSURF(0, DictionarySize, CV_32FC1);
	Mat evalDataFAST(0, DictionarySize, CV_32FC1);
	vector<KeyPoint> keypoint2SIFT, keypoint2SURF, keypoint2FAST;
	Mat bowDescriptor2SIFT, bowDescriptor2SURF, bowDescriptor2FAST;

	//using svm and dictionary to get test results
	cout << "Evaluating test image according to Dictionary and using SVM get 5 clusters.\n";
	Mat resultsSIFT(0, 1, CV_32FC1);
	Mat resultsSURF(0, 1, CV_32FC1);
	Mat resultsFAST(0, 1, CV_32FC1);
	vector<int> TestoutputSIFT, TestoutputSURF, TestoutputFAST;
	float response;
	for (int j = 1; j < 6; j++)
		for (int i = 2; i <= 20; i += 2){
		sprintf(filename, "Training\\Im (%i).jpg", i + (j - 1) * 20);
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		detectorSIFT.detect(input, keypoint2SIFT);
		detectorSURF.detect(input, keypoint2SURF);
		detectorFAST.detect(input, keypoint2FAST);
		bowDESIFT.compute(input, keypoint2SIFT, bowDescriptor2SIFT);
		bowDESURF.compute(input, keypoint2SURF, bowDescriptor2SURF);
		bowDEFAST.compute(input, keypoint2FAST, bowDescriptor2FAST);

		evalDataSIFT.push_back(bowDescriptor2SIFT);
		evalDataSURF.push_back(bowDescriptor2SURF);
		evalDataFAST.push_back(bowDescriptor2FAST);

		groundTruth.push_back((float)j);
		response = svmSIFT.predict(bowDescriptor2SIFT);
		TestoutputSIFT.push_back(response);
		resultsSIFT.push_back(response);
		response = svmSURF.predict(bowDescriptor2SURF);
		TestoutputSURF.push_back(response);
		resultsSURF.push_back(response);
		response = svmFAST.predict(bowDescriptor2FAST);
		TestoutputFAST.push_back(response);
		resultsFAST.push_back(response);
		}
	cout << "SIFT: The result for test image is: \n";
	for (int i = 1; i <= 50; i++){
		cout << resultsSIFT.row(i - 1) << ", ";
		if (i % 10 == 0)
			cout << "\n";
	}

	double errorRate;
	//calculate the number of unmatched classes 
	errorRate = (double)countNonZero(groundTruth - resultsSIFT) / evalDataSIFT.rows;
	cout << "Total error rate is: " << errorRate << endl;

	//print confusion matrix
	int count;
	float Probability;
	cout <<" " << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << endl;
	for (int i = 1; i <= 5; i++){
		cout << i;
		for (int j = 1; j <= 5; j++){
			count = 0;
			for (int k = 1; k <= 10; k++){
				if( TestoutputSIFT[(i-1)*10 + k -1] == j)
					count++;
			}
			Probability = float(count) / 10.0;
			cout << setw(8) << Probability;
		}
		cout << "\n";
	}
		
	cout << "SURF: The result for test image is: \n";
	for (int i = 1; i <= 50; i++){
		cout << resultsSURF.row(i - 1) << ", ";
		if (i % 10 == 0)
			cout << "\n";
	}
	//calculate the number of unmatched classes 
	errorRate = (double)countNonZero(groundTruth - resultsSURF) / evalDataSURF.rows;
	cout << "Total error rate is: " << errorRate << endl;

	//print confusion matrix
	cout << " " << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << endl;
	for (int i = 1; i <= 5; i++){
		cout << i;
		for (int j = 1; j <= 5; j++){
			count = 0;
			for (int k = 1; k <= 10; k++){
				if (TestoutputSURF[(i - 1) * 10 + k - 1] == j)
					count++;
			}
			Probability = float(count) / 10.0;
			cout << setw(8) << Probability;
		}
		cout << "\n";
	}

	cout << "FAST + SIFT: The result for test image is: \n";
	for (int i = 1; i <= 50; i++){
		cout << resultsFAST.row(i - 1) << ", ";
		if (i % 10 == 0)
			cout << "\n";
	}
	//calculate the number of unmatched classes 
	errorRate = (double)countNonZero(groundTruth - resultsFAST) / evalDataFAST.rows;
	cout << "Total error rate is: " << errorRate << endl;

	//print confusion matrix
	cout << " " << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << endl;
	for (int i = 1; i <= 5; i++){
		cout << i;
		for (int j = 1; j <= 5; j++){
			count = 0;
			for (int k = 1; k <= 10; k++){
				if (TestoutputFAST[(i - 1) * 10 + k - 1] == j)
					count++;
			}
			Probability = float(count) / 10.0;
			cout << setw(8) << Probability;
		}
		cout << "\n";
	}
	
	return 0;
}