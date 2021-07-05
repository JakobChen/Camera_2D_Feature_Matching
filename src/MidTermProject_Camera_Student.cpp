/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    //save the  results  as  vectors
    // keypoints in detector
    string decTyp = "SIFT";//SHITOMASI, HARRIS, BRISK, FAST
    string despTyp = "BRIEF";//BRIEF FREAK,ORB, BRISK, FAST,SIFT
    vector<int> keypointsNum;
    vector<int> machtedKeypointsNum;
    vector<double> detectorTime;
    vector<double> descriptorTime;
    vector<double> matcherTime;

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        //keep the only at most defined dataBufferSize frames in the data buffer.
        if(dataBuffer.size()==dataBufferSize){
             // vector is contiguous array, elements are  conautomaticaly shifted to left, when the  first position is deleted.
            dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = decTyp;//SHITOMASI, HARRIS, BRISK, FAST

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double t = (double)cv::getTickCount();

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else 
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);

            //...
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        double detectorRuntime = 1000 * t / 1.0; // in ms
        detectorTime.push_back(detectorRuntime);
        //cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        // 
        /**filtering with rectanle box and  containing only keypoints only inside the rect box**/
        //S1: check if the keypoint inside the defined rectangle
        //S2: Push the keypoint into innerKeypoints vector, if S1 is true
        //S3: replace  the value of keypoints wiith innerKeypoints.
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
             vector<cv::KeyPoint> innerKeypoints;
             for(auto kpt: keypoints){//iterates all the keypoints
                if(vehicleRect.contains(kpt.pt)){
                    innerKeypoints.push_back(kpt);
                }
             }
             keypoints = innerKeypoints;
        }

        // Task MP.7 key points of the detectors
        cout<<"Number of keypoins from detector "<< detectorType<<": " <<keypoints.size()<<endl;
        keypointsNum.push_back(keypoints.size());
        //// EOF STUDENT ASSIGNMENT
        bVis = true;
        if (bVis)
        {
            cv::Mat visImage = imgGray.clone();
            cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "Modern Corner Detector " + detectorType +  " Results";
            cv::namedWindow(windowName, 6);

            imshow(windowName, visImage);

            cv::waitKey(0);
            imwrite("../images/"+detectorType+"_detector.png",visImage); 

        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }
        bVis = false;

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = despTyp; // BRIEF, ORB, FREAK, AKAZE, SIFT
        t = (double)cv::getTickCount();

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        double descriptorRuntime = 1000 * t / 1.0; // in ms
        descriptorTime.push_back(descriptorRuntime);
        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorDataType = "DES_BINARY"; // DES_BINARY(ORB, BRISK, BRIEF, FREAK, AKAEZE), DES_HOG(SIFT)
            if(descriptorType.compare("SIFT") == 0){
                descriptorDataType = "DES_HOG";
            }
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            t = (double)cv::getTickCount();

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorDataType, matcherType, selectorType);
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            double matcherRuntime = 1000 * t / 1.0; // in ms
            matcherTime.push_back(matcherRuntime);
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;
            // TASK MP.8 number of matched keypoints under the distance ratio 0.8
            cout<<"Number of matched keypoins from detector "<< descriptorType<<": " <<matches.size()<<endl;
            machtedKeypointsNum.push_back(matches.size());
            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images


    cout<<"Summary of Performance:********************************"<<endl;
    double keypointsNumAvg =  accumulate(keypointsNum.begin(),keypointsNum.end(),0.) / keypointsNum.size();
    double matchedKeypointsNumAvg =  accumulate(machtedKeypointsNum.begin(),machtedKeypointsNum.end(),0.) / machtedKeypointsNum.size();
    double detectorTimeAvg =  accumulate(detectorTime.begin(),detectorTime.end(),0.) / detectorTime.size();
    double descriptorTimeAvg =  accumulate(descriptorTime.begin(),descriptorTime.end(),0.) / descriptorTime.size();
    double matcherTimeAvg =  accumulate(matcherTime.begin(),matcherTime.end(),0.) / matcherTime.size();
    double totalTimeAvg = detectorTimeAvg + descriptorTimeAvg + matcherTimeAvg;
    cout << decTyp<<" detected keypoits: " << keypointsNumAvg <<endl;
    cout <<" Mathed keypoits: " << matchedKeypointsNumAvg <<endl;
    cout << decTyp<<" detector computes time iin (ms): " << detectorTimeAvg <<endl;
    cout << despTyp <<" descriptor computes time in (ms): " << descriptorTimeAvg <<endl;
    cout << decTyp<<" matcher computes time in (ms) : " << matcherTimeAvg <<endl;
    cout << " Total computes time in (ms) : " << totalTimeAvg <<endl;

    return 0;
}