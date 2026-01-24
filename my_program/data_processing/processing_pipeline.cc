/**
* ORB-SLAM3 RGBD Dataset Processor
* Processes RGB-D image sequences (extracted from bag files)
* Similar to stereo_kitti.cc but for RGBD

Use the program like this 
./my_program/data_processing/processing_pipeline     ./Vocabulary/ORBvoc.txt     ./Examples/RGB-D/RealSense_D435i.yaml     ./my_program/data_processing/my_dataset


*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <System.h>

using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./rgbd_dataset path_to_vocabulary path_to_settings path_to_sequence" << endl;
        cerr << endl;
        cerr << "Example: ./rgbd_dataset Vocabulary/ORBvoc.txt Examples/RGB-D/RealSense_D435i.yaml dataset_output" << endl;
        cerr << endl;
        cerr << "Dataset structure:" << endl;
        cerr << "  dataset_output/" << endl;
        cerr << "    rgb/           - RGB images (000000.png, 000001.png, ...)" << endl;
        cerr << "    depth/         - Depth images (000000.png, 000001.png, ...)" << endl;
        cerr << "    rgb.txt        - RGB timestamps" << endl;
        cerr << "    depth.txt      - Depth timestamps (optional, uses rgb.txt if missing)" << endl;
        cerr << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    
    string strPathToSequence = string(argv[3]);
    
    cout << "[INFO] Loading image paths from: " << strPathToSequence << endl;
    LoadImages(strPathToSequence, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and timestamps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << "[ERROR] No images found in " << strPathToSequence << endl;
        cerr << "[ERROR] Make sure rgb/ and depth/ folders exist with images" << endl;
        return 1;
    }

    if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << "[ERROR] Different number of images for rgb and depth" << endl;
        return 1;
    }

    // Extract dataset name for trajectory filename
    size_t last_slash = strPathToSequence.find_last_of("/\\");
    string dataset_name = (last_slash == string::npos) ? strPathToSequence : strPathToSequence.substr(last_slash + 1);
    if (dataset_name.empty()) {
        dataset_name = "trajectory";
    }

    cout << "[INFO] Dataset: " << dataset_name << endl;
    cout << "[INFO] Number of images: " << nImages << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    cout << "[INFO] Initializing ORB-SLAM3..." << endl;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, false, 0, dataset_name);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read RGB and depth images from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "[ERROR] Failed to load RGB image at: "
                 << string(vstrImageFilenamesRGB[ni]) << endl;
            return 1;
        }

        if(imD.empty())
        {
            cerr << endl << "[ERROR] Failed to load depth image at: "
                 << string(vstrImageFilenamesD[ni]) << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackRGBD(imRGB, imD, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Progress indicator
        if(ni % 30 == 0 || ni == nImages - 1)
        {
            cout << "[PROGRESS] Frame " << ni + 1 << "/" << nImages 
                 << " (tracking time: " << ttrack << "s)" << endl;
        }

        // Wait to simulate real-time playback (optional, comment out for max speed)
        // double T = 0;
        // if(ni < nImages-1)
        //     T = vTimestamps[ni+1] - tframe;
        // else if(ni > 0)
        //     T = tframe - vTimestamps[ni-1];
        // if(ttrack < T)
        //     usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    cout << "\n[INFO] Shutting down SLAM system..." << endl;
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    
    cout << "\n-------" << endl;
    cout << "Processing Statistics:" << endl;
    cout << "-------" << endl;
    cout << "Median tracking time: " << vTimesTrack[nImages/2] << "s" << endl;
    cout << "Mean tracking time: " << totaltime/nImages << "s" << endl;
    cout << "Total time: " << totaltime << "s" << endl;
    cout << "-------" << endl << endl;

    // Save camera trajectory
    string traj_camera = dataset_name + "_CameraTrajectory.txt";
    string traj_keyframe = dataset_name + "_KeyFrameTrajectory.txt";
    
    cout << "[INFO] Saving trajectories..." << endl;
    SLAM.SaveTrajectoryTUM(traj_camera);
    SLAM.SaveKeyFrameTrajectoryTUM(traj_keyframe);

    cout << "\n[SUCCESS] Trajectories saved:" << endl;
    cout << "  - " << traj_camera << endl;
    cout << "  - " << traj_keyframe << endl;
    cout << "\n[INFO] Done!" << endl;

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    // Read timestamps from rgb.txt
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/rgb.txt";
    fTimes.open(strPathTimeFile.c_str());
    
    if(!fTimes.is_open())
    {
        cerr << "[ERROR] Could not open timestamp file: " << strPathTimeFile << endl;
        return;
    }

    vTimestamps.clear();
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if(!s.empty() && s[0] != '#') // Skip empty lines and comments
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    string strPrefixRGB = strPathToSequence + "/rgb/";
    string strPrefixD = strPathToSequence + "/depth/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenamesRGB.resize(nTimes);
    vstrImageFilenamesD.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenamesRGB[i] = strPrefixRGB + ss.str() + ".png";
        vstrImageFilenamesD[i] = strPrefixD + ss.str() + ".png";
    }
}