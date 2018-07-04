#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/flann/flann.hpp"

using namespace cv;
using namespace std;

char classes[11][20]={"close","pour","open","spread","scoop","take","fold", "shake", "put","stir","x"};

struct feature{
    int size;
    vector<int> index;
    vector<int> val;
};

void writeToBinaryFile(Mat &dataToWrite , char *fileName)
{
    /**
    *  writes the data to the .bin file .
    *  
    *  Parameters: # dataToWrite - matrix to write in binbary file. Type CV_8UC1.
    *              # fileName - name of binary file.
    **/
  
    fstream binaryFile(fileName,ios::binary|ios::out);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }

    binaryFile.write((char *)dataToWrite.data, dataToWrite.rows*dataToWrite.cols) ;
    
    binaryFile.close();
}


int get_label(char *a)
{
    for(int i=0;i<11;i++)
    {
        if(!strcmp(a,classes[i]))
            return i;
    }

    return 11;
}

Mat get_feature_video(char *command)
{
	FILE *temp_file = popen(command,"r");

        Mat feature_video = Mat(0,436,CV_32FC1);
        while(!feof(temp_file))
        {
            float feature_array[436];
            for(int i=0;i<436 && !feof(temp_file);i++)
            {
                fscanf(temp_file, "%f", &feature_array[i]);
            }
        
            Mat feature_row = Mat(1,436,CV_32FC1, &feature_array);
            
            feature_video.push_back(feature_row);
        }
        fclose(temp_file);
        
        return feature_video;
}

feature get_feature(Mat a)
{
	feature f;
	f.size =0;
	for(int i=0;i<a.cols;i++)
	{
		if(a.at<int>(0,i)>0)
		{
			f.size++;
			f.index.push_back(i);
			f.val.push_back(a.at<int>(0,i));
		}
	}

	return f;
}


void get_video_BOFhist_one_by_one(vector<Mat> &vocabulary, char *path, vector<feature> &feature_histogram, int skip, char *prefix)
{
    
    FILE *fp = fopen(path, "r");
    int status=0;
 
    cv::flann::KDTreeIndexParams indexParams0,indexParams1,indexParams2,indexParams3,indexParams4;
    cv::flann::Index kdtree0(vocabulary[0], indexParams0);
    cv::flann::Index kdtree1(vocabulary[1], indexParams1);
    cv::flann::Index kdtree2(vocabulary[2], indexParams2);
    cv::flann::Index kdtree3(vocabulary[3], indexParams3);
    cv::flann::Index kdtree4(vocabulary[4], indexParams4);
    
    

	vector<int> index(1);
	vector<float> dist(1);
   
    
    while(!feof(fp))
    {
        char feature_file_name[200], command[200], label[50];
        
        for(int i=0;i<=skip;i++)
	        fscanf(fp,"%s%s", feature_file_name, label);
        
        sprintf(command, "cat %s > %s", feature_file_name, prefix);
        system(command);
    	sprintf(command, "/home/suriya/ego/utils/dense_trajectory_release_v1.2/release/DenseTrack %s", prefix);
        
        Mat feature_video;
        feature_video.push_back(get_feature_video(command));
        
        Mat feature_frame = feature_video(Rect(0,0,1,feature_video.rows)).clone();   //get frame no. when the trajectory ends
 	    feature_frame = feature_frame-15;                                            // get frame no. when the trajectory starts
        
        ////////////////// extract features from video in reverse ////////////////////        
	    sprintf(command, "tac %s > %s", feature_file_name, prefix);
        system(command);
	    sprintf(command, "/home/suriya/ego/utils/dense_trajectory_release_v1.2/release/DenseTrack %s", prefix);
        
        Mat feature_video2;
        feature_video2.push_back(get_feature_video(command));
        Mat feature_frame2 = feature_video2(Rect(0,0,1,feature_video2.rows)).clone();
 	    feature_frame2 = feature_frame2+15;
 	

 	    for(int i=0;i<feature_frame2.rows;i++)
	    {
            feature_frame2.at<float>(i,0) = feature_frame2.at<float>(i,0) - 2*(((int)feature_frame2.at<float>(i,0))%30);
	    }
 	
 	    feature_video.push_back(feature_video2);
 	    feature_frame.push_back(feature_frame2);
        
        if(feature_video.rows<2)
            continue;
 
 	
        feature_video = feature_video(Rect(10,0,feature_video.cols-10, feature_video.rows)).clone();

        Mat BOFhist;
        vector<Mat> B;
        for(int i=0;i<5;i++)
	    {
	       Mat b = Mat::zeros(1,vocabulary[i].rows*7,CV_32SC1 );
	       B.push_back(b);
	    }

        for(int j=0;j<feature_video.rows;j++)           // iterates over all trajectories
        {
       	    int index_1 = -1;
       	    int index_2 = -1;
            if(feature_frame.at<float>(j,0) <= 15)
            {
            	index_1 = vocabulary[0].rows;           // level1 BOF index offset
            	if(feature_frame.at<float>(j,0) <= 7)
            	{
            		index_2 = 3*vocabulary[0].rows;     // level2 BOF index offset
            	}
            	else
            	{
            		index_2 = 4*vocabulary[0].rows;     // level2 BOF index offset
            	}
            }
            else
            {
            	index_1 = 2*vocabulary[0].rows;        // level1 BOF index offset
            	if(feature_frame.at<float>(j,0) <= 22)
            	{
            		index_2 = 5*vocabulary[0].rows;    // level2 BOF index offset
            	}
            	else
            	{
            		index_2 = 6*vocabulary[0].rows;    // level2 BOF index offset
            	}
            }
            
            Mat r = feature_video.row(j).clone();
            
            // BOF assignment for trajectory feature
            Mat t = r(Rect(0,0,30,1)).clone();
            kdtree0.knnSearch(t, index, dist, 1, cv::flann::SearchParams(32));
            B[0].at<int>(0,index[0]) = B[0].at<int>(0,index[0])+1;
            B[0].at<int>(0,index_1+index[0]) = B[0].at<int>(0,index_1+index[0])+1;
            B[0].at<int>(0,index_2+index[0]) = B[0].at<int>(0,index_2+index[0])+1;
            
            // BOF assignment for HOG feature            
            t = r(Rect(30,0,96,1)).clone();
            kdtree1.knnSearch(t, index, dist, 1, cv::flann::SearchParams(32));
            B[1].at<int>(0,index[0]) = B[1].at<int>(0,index[0])+1;
            B[1].at<int>(0,index_1+index[0]) = B[1].at<int>(0,index_1+index[0])+1;
            B[1].at<int>(0,index_2+index[0]) = B[1].at<int>(0,index_2+index[0])+1;
            
            // BOF assignment for HOF feature
            t = r(Rect(30+96,0,108,1)).clone();
            kdtree2.knnSearch(t, index, dist, 1, cv::flann::SearchParams(32));
            B[2].at<int>(0,index[0]) = B[2].at<int>(0,index[0])+1;
            B[2].at<int>(0,index_1+index[0]) = B[2].at<int>(0,index_1+index[0])+1;
            B[2].at<int>(0,index_2+index[0]) = B[2].at<int>(0,index_2+index[0])+1;
            
            // BOF assignment for MBHx feature
            t = r(Rect(30+96+108,0,96,1)).clone();
            kdtree3.knnSearch(t, index, dist, 1, cv::flann::SearchParams(32));
            B[3].at<int>(0,index[0]) = B[3].at<int>(0,index[0])+1;
            B[3].at<int>(0,index_1+index[0]) = B[3].at<int>(0,index_1+index[0])+1;
            B[3].at<int>(0,index_2+index[0]) = B[3].at<int>(0,index_2+index[0])+1;
            
            // BOF assignment for MBHy feature
            t = r(Rect(30+96+108+96,0,96,1)).clone();
            kdtree4.knnSearch(t, index, dist, 1, cv::flann::SearchParams(32));
            B[4].at<int>(0,index[0]) = B[4].at<int>(0,index[0])+1;
            B[4].at<int>(0,index_1+index[0]) = B[4].at<int>(0,index_1+index[0])+1;
            B[4].at<int>(0,index_2+index[0]) = B[4].at<int>(0,index_2+index[0])+1;
        }
        
        BOFhist = B[0].clone();
        for(int i=1;i<5;i++)
        {
        	hconcat(BOFhist, B[i], BOFhist);
        }
       
        Mat l = Mat(1,1,CV_32S);
        l.at<int>(0,0) = get_label(label);

        hconcat(l, BOFhist, BOFhist);
        feature f = get_feature(BOFhist);
	
        feature_histogram.push_back(f);

                
        printf("\n%d %dx%d : %s\t[label = %s (%d)] %d",status++, feature_histogram.size(), BOFhist.cols, feature_file_name, label, get_label(label) , f.size );
    }

    fclose(fp);
}


void writeToBinaryFile(vector<feature> feature_histogram, int feature_size , char *fileName)
{
    fstream binaryFile(fileName, ios::out | ios::binary);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }
    	char name[100];
    	sprintf(name, "%s_sizes.txt", fileName);
	FILE *fp = fopen(name,"w");
	fprintf(fp,"%d %d\n", feature_histogram.size(), feature_size);
    for(int i=0;i<feature_histogram.size();i++)
    {
    	fprintf(fp,"%d\n", feature_histogram[i].size);
        for(int j=0;j<feature_histogram[i].size;j++)
        {
            int index = feature_histogram[i].index[j];
            int val = feature_histogram[i].val[j];
            binaryFile.write((char *)&index,sizeof(index)) ;
            binaryFile.write((char *)&val,sizeof(val)) ;
        }
    }
    
    fclose(fp);
  
    binaryFile.close();
}

int main(int argc, char **argv)
{
    char *path_to_train_clips = argv[1];
    char *path_to_test_clips = argv[2];
    char *path_to_vocabulary = argv[3];
    char *vocab_name_prefix = argv[4];
    char *train_features = argv[5];
    char *test_features = argv[6];
    char *prefix = argv[7];
    char *vocab_sizes_file = argv[8];

    printf("\n*********** dense DT start ***********");
    printf("\n");
    
    
    vector<Mat> vocabulary;

	int feature_size=0;
	vector<int> vocab_sizes;
   
  	char path[100], name[100];
  	Mat vocab=Mat();
    printf("\nReading trjectory vocabulary\n");
   	sprintf(path,"%s/%s_trajectory_2000.yml", path_to_vocabulary,vocab_name_prefix);
   	sprintf(name,"%s_trajectory_2000", vocab_name_prefix);

   	FileStorage fileStorage(path, FileStorage::READ);
    fileStorage[name]>> vocab;
    printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
    vocabulary.push_back(vocab);
    feature_size += vocab.rows;
    vocab_sizes.push_back(vocab.rows);
    
    vocab=Mat();
    printf("\nReading HOG vocabulary\n");
    sprintf(path,"%s/%s_HOG_2000.yml", path_to_vocabulary, vocab_name_prefix);
  	sprintf(name,"%s_HOG_2000", vocab_name_prefix);
    fileStorage = FileStorage(path, FileStorage::READ);
    fileStorage[name]>> vocab;
    printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
    vocabulary.push_back(vocab);
    feature_size += vocab.rows;
    vocab_sizes.push_back(vocab.rows);
    
    vocab=Mat();
    printf("\nReading HOF vocabulary\n");
    sprintf(path,"%s/%s_HOF_2000.yml", path_to_vocabulary,vocab_name_prefix);
    sprintf(name,"%s_HOF_2000", vocab_name_prefix);
    fileStorage = FileStorage(path, FileStorage::READ);
    fileStorage[name]>> vocab;
    printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
    vocabulary.push_back(vocab);
    vocab_sizes.push_back(vocab.rows);
    feature_size += vocab.rows;
               
    vocab=Mat();
    printf("\nReading MBHx vocabulary\n");
    sprintf(path,"%s/%s_MBHx_2000.yml", path_to_vocabulary,vocab_name_prefix);
    sprintf(name,"%s_MBHx_2000", vocab_name_prefix);
    fileStorage = FileStorage(path, FileStorage::READ);
    fileStorage[name]>> vocab;
    printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
    vocabulary.push_back(vocab);
    feature_size += vocab.rows;
    vocab_sizes.push_back(vocab.rows);
        
    vocab=Mat();
    printf("\nReading MBHy vocabulary\n");
    sprintf(path,"%s/%s_MBHy_2000.yml", path_to_vocabulary, vocab_name_prefix);
    sprintf(name,"%s_MBHy_2000", vocab_name_prefix);
    fileStorage = FileStorage(path, FileStorage::READ);
    fileStorage[name]>> vocab;
    printf("\nvocabulary : %dx%d ",vocab.rows, vocab.cols );
    vocabulary.push_back(vocab);
    feature_size += vocab.rows;
    vocab_sizes.push_back(vocab.rows);
        
    fileStorage.release();
   
   	FILE *vs = fopen(vocab_sizes_file,"w");
       for(int i =0;i<vocab_sizes.size();i++)
       		fprintf(vs, "%d\n", vocab_sizes[i]);
    fclose(vs);

    //temporal pyramid feature dimension. level0 dim = feature_size, level1 dim = 2*feature_size, level2 dim = 4*feature_size
    feature_size*=7;   
    feature_size+=1; //label to be used to train classifier (not actually part of the feature!)
    printf("\nFeature Size : %d", feature_size);
    
    printf("\n*********** get train feature histogram ***********");
    printf("\n");

    vector<feature> feature_histogram;
    get_video_BOFhist_one_by_one(vocabulary, path_to_train_clips, feature_histogram, 0, prefix);
    writeToBinaryFile(feature_histogram, feature_size, train_features);
    feature_histogram.clear();
    
    printf("\n*********** get test feature histogram ***********");   
    printf("\n"); 
    vector<feature> test_feature_histogram;
    get_video_BOFhist_one_by_one(vocabulary, path_to_test_clips, test_feature_histogram, 0, prefix);
    writeToBinaryFile(test_feature_histogram, feature_size, test_features);

    printf("\n*********** finish ***********");
    printf("\n");

    return 0;
}
