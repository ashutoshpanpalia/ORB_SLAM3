/*
Use the script on terminal cd /Dev/ORB_SLAM3$ 
./my_program/data_collection_pipeline/processing_bulk_dataset ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/RealSense_D435i.yaml


*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <System.h>
#include <dirent.h>   
#include <sys/stat.h> 

using namespace std;

void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps);

bool directory_exists(const string &path)
{
    struct stat info;
    if(stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & S_IFDIR) != 0;
}

vector<string> list_subdirectories(const string &path)
{
    vector<string> dirs;
    DIR *dir = opendir(path.c_str());
    if(!dir) return dirs;

    struct dirent *entry;
    while((entry = readdir(dir)) != nullptr)
    {
        if(entry->d_type == DT_DIR)
        {
            string name = entry->d_name;
            if(name != "." && name != "..")
                dirs.push_back(path + "/" + name);
        }
    }
    closedir(dir);
    return dirs;
}

string get_folder_name(const string &path)
{
    size_t pos = path.find_last_of("/\\");
    if(pos == string::npos) return path;
    return path.substr(pos + 1);
}

void ProcessDataset(const string &vocab, const string &settings, const string &dataset_path)
{
    cout << "\n========================================" << endl;
    cout << "[INFO] Processing dataset: " << dataset_path << endl;

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;

    LoadImages(dataset_path, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << "[WARNING] No images found in: " << dataset_path << endl;
        return;
    }

    int nImages = vstrImageFilenamesRGB.size();
    cout << "[INFO] Frames: " << nImages << endl;

    string dataset_name = get_folder_name(dataset_path);

    ORB_SLAM3::System SLAM(vocab, settings, ORB_SLAM3::System::RGBD, false, 0, dataset_name);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack(nImages);

    cv::Mat imRGB, imD;

    for(int ni = 0; ni < nImages; ni++)
    {
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        imD   = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty() || imD.empty())
        {
            cerr << "[ERROR] Failed to load images at frame " << ni << endl;
            return;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackRGBD(imRGB, imD, tframe);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration<double>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        if(ni % 30 == 0 || ni == nImages - 1)
        {
            cout << "[PROGRESS] " << dataset_name
                 << " Frame " << ni + 1 << "/" << nImages
                 << " (" << ttrack << "s)" << endl;
        }
    }

    SLAM.Shutdown();

    // string traj_camera = dataset_name + "_CameraTrajectory.txt";
    // string traj_keyframe = dataset_name + "_KeyFrameTrajectory.txt";

    string traj_camera = dataset_path + "/" + dataset_name + "_CameraTrajectory.txt";
    string traj_keyframe = dataset_path + "/" + dataset_name + "_KeyFrameTrajectory.txt";

    SLAM.SaveTrajectoryTUM(traj_camera);
    SLAM.SaveKeyFrameTrajectoryTUM(traj_keyframe);

    cout << "[SUCCESS] Saved trajectories:" << endl;
    cout << "  - " << traj_camera << endl;
    cout << "  - " << traj_keyframe << endl;
}

int main(int argc, char **argv)
{
    if(argc < 3 || argc > 4)
    {
        cerr << endl << "Usage: ./bulk_processing_pipeline path_to_vocabulary path_to_settings [path_to_dataset_root]" << endl;
        return 1;
    }

    string vocab = argv[1];
    string settings = argv[2];

    // Default dataset folder
    string dataset_root = "./my_program/data_collection_pipeline/data/processed_dataset";

    // Override if user provides dataset folder
    if(argc == 4)
        dataset_root = argv[3];

    if(!directory_exists(dataset_root))
    {
        cerr << "[ERROR] Dataset root not found: " << dataset_root << endl;
        return 1;
    }

    cout << "[INFO] Scanning dataset root: " << dataset_root << endl;

    vector<string> subdirs = list_subdirectories(dataset_root);
    for(const auto &dataset_path : subdirs)
    {
        string folder_name = get_folder_name(dataset_path);
        if(folder_name == "raw_bag_files") continue;

        if(directory_exists(dataset_path + "/rgb") && directory_exists(dataset_path + "/depth"))
        {
            ProcessDataset(vocab, settings, dataset_path);
        }
        else
        {
            cout << "[SKIP] Not a dataset folder: " << dataset_path << endl;
        }
    }

    cout << "\n========================================" << endl;
    cout << "[DONE] All datasets processed." << endl;
    cout << "========================================" << endl;

    return 0;
}

void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD,
                vector<double> &vTimestamps)
{
    ifstream fTimes(strPathToSequence + "/rgb.txt");

    if(!fTimes.is_open())
    {
        cerr << "[ERROR] Missing rgb.txt in " << strPathToSequence << endl;
        return;
    }

    vTimestamps.clear();
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if(!s.empty() && s[0] != '#')
        {
            stringstream ss(s);
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }
    fTimes.close();

    string strPrefixRGB = strPathToSequence + "/rgb/";
    string strPrefixD = strPathToSequence + "/depth/";

    int nTimes = vTimestamps.size();
    vstrImageFilenamesRGB.resize(nTimes);
    vstrImageFilenamesD.resize(nTimes);

    for(int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenamesRGB[i] = strPrefixRGB + ss.str() + ".png";
        vstrImageFilenamesD[i] = strPrefixD + ss.str() + ".png";
    }
}
