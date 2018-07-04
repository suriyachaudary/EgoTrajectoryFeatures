#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/flann/flann.hpp"

using namespace cv;
using namespace std;

char classes[11][20]={"close","pour","open","spread","scoop","take","fold", "shake", "put","stir","x"};
RNG rng( 0xFFFFFFFF );

Mat kMeansCluster(Mat &data,int clusterSize, int type = CV_8UC1)
{ 
    /**
    *  implements k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
    *  
    *  Parameters: # data- Each sample in rows. Type should be CV_32FC1
    *              # clusterSize- Number of clusters.
    **/

    TermCriteria termCriteria(CV_TERMCRIT_ITER,100,0.01);
    int nAttempts=3;
    int flags=KMEANS_PP_CENTERS;
      
    Mat clusterCenters, temp;
    kmeans(data, clusterSize, temp, termCriteria, nAttempts, flags, clusterCenters);
    clusterCenters.convertTo(clusterCenters,type);
    return clusterCenters;
}

Mat hiKMeansCluster(Mat &data,int clusterSize, int type = CV_8UC1, iterations = 2000)
{   
    /**
    *  implements Hierarchical k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
    *
    *  Parameters: # data- Each sample in rows. Type should be CV_32FC1.
    *              # clusterSize- Number of clusters.
    **/
    Mat clusterCenters=Mat(clusterSize,data.cols,CV_32F);
    cvflann::KMeansIndexParams kParams = cvflann::KMeansIndexParams(2, iterations, cvflann::FLANN_CENTERS_RANDOM,0.2);
    int numClusters =cv::flann::hierarchicalClustering<cvflann::L2<float> >(data, clusterCenters, kParams);
    clusterCenters = clusterCenters.rowRange(cv::Range(0,numClusters));
    clusterCenters.convertTo(clusterCenters, type);
  
    return clusterCenters;
}

void writeToYMLFile(Mat &dataToWrite, char *fileName)
{
    /**
    *  writes the data to the .yml file.
    *
    *  Parameters: # dataToWrite - matrix to write in .yml file.
    *              # fileName - name of yml file.
    **/
  
    stringstream ss;
    ss<<fileName<<".yml";
    string s = ss.str();
    FileStorage fileStorage(s, FileStorage::WRITE);
    s= string(fileName);
    
    fileStorage << s << dataToWrite;
    
    fileStorage.release();
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

            if(rng.uniform(0, 10) < 3)     // use random 30% of trajectories
            {
                feature_video.push_back(feature_row);
            }
        }
    
    fclose(temp_file);
        
    return feature_video;
}

void get_features(char *path, Mat &train_features, vector<int> &features_each_video, vector<int> &labels, char *prefix)
{
    FILE *fp = fopen(path, "r");
    int status = 0;
    while(!feof(fp))
    {
        Mat feature;
        char feature_file_name[400], command[200], label[50];

        /// extract features from every 10th clip (for making vocabulary)
        for(int i=0;i<10;i++)
            fscanf(fp,"%s%s", feature_file_name, label);
    
        sprintf(command, "cat %s > %s", feature_file_name, prefix);
        system(command);
        sprintf(command, "/home/suriya/ego/utils/dense_trajectory_release_v1.2/release/DenseTrack %s", prefix);
        
        Mat feature_video = get_feature_video(command);        
        
        if(feature_video.rows<2)   // discard the clip when not enough trajectories are found
           continue;
        
        feature_video = feature_video(Rect(10, 0, feature_video.cols-(10), feature_video.rows)).clone();
        features_each_video.push_back(feature_video.rows);
        train_features.push_back(feature_video);

        labels.push_back(get_label(label));

        ////////////////// extract features from video in reverse ////////////////////        
        sprintf(command, "tac %s > %s", feature_file_name, prefix);
        system(command);
        sprintf(command, "/home/suriya/ego/utils/dense_trajectory_release_v1.2/release/DenseTrack %s", prefix);
        
        feature_video = get_feature_video(command);        
        
        if(feature_video.rows<2)
           continue;
        
        feature_video = feature_video(Rect(10, 0, feature_video.cols-(10), feature_video.rows)).clone();
        features_each_video.push_back(feature_video.rows);
        train_features.push_back(feature_video);

        labels.push_back(get_label(label));
                
                
        printf("%d %dx%d : %s\t[label = %s (%d)]\n",status++, train_features.rows, train_features.cols, feature_file_name, label, labels.back() );
    }

    fclose(fp);
}





int main(int argc, char **argv)
{

    char *path_to_train_clips = argv[1];
    char *vocabulary_name_prefix = argv[2];
    int clusterSize[3] = {2000,5000,10000};  // cluster sizes.
    
    printf("\n*********** dense DT start ***********");
    printf("\n");
    vector<int> features_each_video, labels;
    Mat train_features;
    get_features(path_to_train_clips, train_features, features_each_video, labels, argv[3]);
    writeToYMLFile(train_features,"train_features");
    
    Mat trajectory = train_features(Rect(0,0,30,train_features.rows)).clone();
    Mat HOG = train_features(Rect(30,0,96,train_features.rows)).clone();
    Mat HOF = train_features(Rect(30+96,0,108,train_features.rows)).clone();
    Mat MBHx = train_features(Rect(30+96+108,0,96,train_features.rows)).clone();
    Mat MBHy = train_features(Rect(30+96+108+96,0,96,train_features.rows)).clone();

    train_features.release();
    printf("\n");
    
    Mat vocabulary;
    
    for(int i=0;i<3;i++)
    {
        char name[100];
        sprintf(name,"%s_trajectory_%d", vocabulary_name_prefix, clusterSize[i]);
        
        printf("\nBuilding vocabulary : %s", name);
        printf("\n");
        vocabulary = hiKMeansCluster(trajectory,clusterSize[i], CV_32FC1);
        writeToYMLFile(vocabulary,(char *)name);
        
        sprintf(name,"%s_HOG_%d", vocabulary_name_prefix, clusterSize[i]);
        
        printf("\nBuilding vocabulary : %s", name);
        printf("\n");
        vocabulary = hiKMeansCluster(HOG,clusterSize[i], CV_32FC1);
        writeToYMLFile(vocabulary,(char *)name);
            
        sprintf(name,"%s_HOF_%d", vocabulary_name_prefix, clusterSize[i]);
        
        printf("\nBuilding vocabulary : %s", name);
        printf("\n");
        vocabulary = hiKMeansCluster(HOF,clusterSize[i], CV_32FC1);
        writeToYMLFile(vocabulary,(char *)name);
            
        sprintf(name,"%s_MBHx_%d", vocabulary_name_prefix, clusterSize[i]);
        
        printf("\nBuilding vocabulary : %s", name);
        printf("\n");
        vocabulary = hiKMeansCluster(MBHx,clusterSize[i], CV_32FC1);
        writeToYMLFile(vocabulary,(char *)name);
            
        sprintf(name,"%s_MBHy_%d", vocabulary_name_prefix, clusterSize[i]);
        
        printf("\nBuilding vocabulary : %s", name);
        printf("\n");
        vocabulary = hiKMeansCluster(MBHy,clusterSize[i], CV_32FC1);
        writeToYMLFile(vocabulary,(char *)name);
    }
        
    printf("\n*********** finish ***********");
    printf("\n");

    return 0;
}
