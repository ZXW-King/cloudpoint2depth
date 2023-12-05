#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <eigen3/Eigen/Dense>
#include <unistd.h>
#include <yaml-cpp/yaml.h>
#include "include/TofDepthData.h"
#include "include/Resolution.h"
#include "include/utils.h"
#include "include/CameraMoudleParam.h"
#include <numeric>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#define EXIST(file) (access((file).c_str(), 0) == 0)
double camera_factor = 100;
double camera_cx;
double camera_cy;
double camera_fx;
double camera_fy;

#define ERROR_PRINT(x) std::cout << "" << (x) << "" << std::endl
const psl::Resolution RESOLUTION = psl::Resolution::RES_640X400;

bool ReadFile(std::string srcFile, std::vector<std::string> &image_files)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return false;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        return false;
    }

    std::string s;
    while (getline(fin, s))
    {
        image_files.push_back(s);
    }

    fin.close();

    return true;
}

void GetImageData(const std::string inputDir, std::vector<std::string> &dataset)
{
    std::string imagesTxt = inputDir + "/image_paths.txt";
    std::vector<std::string> imageNameList;

    if (not ReadFile(imagesTxt, imageNameList)) exit(0);
    const size_t size =
            imageNameList.size() > 0 ? imageNameList.size() : 0;

    for (size_t i = 0; i < size; ++i)
    {
        std::string imagePath = inputDir + "/" + imageNameList.at(i);
        dataset.push_back(imagePath);
    }
}


void Depth2PointCloud(const cv::Mat &depth, std::vector<Eigen::Vector3d> &cloud)
{
    for (int m = 0; m < depth.rows; m++)
        for (int n = 0; n < depth.cols; n++)
        {
            unsigned short d = depth.ptr<uint16_t>(m)[n];
//            d = d / 100;

            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;

            // d 存在值，则向点云增加一个点
            Eigen::Vector3d p;
            // 计算这个点的空间坐标
            p.z() = d / camera_factor;       // 正方向朝前
            p.x() = (n - camera_cx) * p.z() / camera_fx; // 正方向朝右
            p.y() = (m - camera_cy) * p.z() / camera_fy;   // 正方向朝下

            // 把p加入到点云中
            cloud.push_back(p);

        }
}


void PointCloud2Depth(const std::vector<Eigen::Vector3d> &pointCloud, cv::Mat &depthImage, int width, int height)
{
    // 创建深度图像
    depthImage = cv::Mat::zeros(height, width, CV_16U);
    // 遍历点云数据
    for (const auto &point: pointCloud)
    {
        // 计算深度值
        double x = point.x();
        double y = point.y();
        double z = point.z();
        int u = static_cast<int>(std::round(x * camera_fx / z) + camera_cx);
        int v = static_cast<int>(std::round(y * camera_fy / z) + camera_cy);

        // 更新深度图像
        if (u >= 0 && u < width && v >= 0 && v < height)
        {
            unsigned short depthValue = static_cast<unsigned short>(z * camera_factor * 100); // 假设深度单位为米，转换为厘米
            depthImage.at<uint16_t>(v, u) = depthValue;
        }
    }
}
void ReadArray(const YAML::Node &config, std::vector<float> &array)
{
    try
    {
        array = config.as<std::vector<float>>();
    }
    catch (...)
    {
        for (YAML::const_iterator it = config.begin(); it != config.end(); ++it)
        {
            array.push_back((*it).as<float>());
        }
    }
}


bool GetYamlPointCloud(std::string yamlFile, PointCloudData &pointCloud)
{
    try
    {
        YAML::Node config;

        if (not access(yamlFile.c_str(), 0) == 0)
        {
            std::cout << "file not exist <" + yamlFile + ">" << std::endl;
        }
        config = YAML::LoadFile(yamlFile);

        pointCloud.time = config["time"].as<unsigned long>();
        pointCloud.tf = config["tf"].as<std::string>();

        pointCloud.rows = config["rows"].as<int>();
        pointCloud.cols = config["cols"].as<int>();

        std::vector<float> rotation(4);
        ReadArray(config["rotation"], rotation);
        std::vector<float> position(3);
        ReadArray(config["position"], position);

        for (int i = 0; i < 4; i++)
        {
            pointCloud.rotation[i] = rotation[i];
        }

        static std::vector<float> data(448 * 80 * 3);

        ReadArray(config["data"], data);

        int p = 0;
        for (int i = 0; i < 448 * 80; i++)
        {
            pointCloud.pointData.push_back(Eigen::Vector3d(data[p++], data[p++], data[p++]));
        }

        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ReadSyncFile(std::string srcFile, std::vector<SyncDataFile> &files, const bool &flag)
{
    if (not access(srcFile.c_str(), 0) == 0)
    {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return false;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open())
    {
        ERROR_PRINT("read file error (" + srcFile + ")");
        return false;
    }

    std::string s;
    SyncDataFile syncFile;

    do
    {
        fin >> syncFile.imageLeft >> syncFile.imagePose >> syncFile.lidar
            >> syncFile.lidarPose;
        if (flag)
        {
            fin >> syncFile.pointCloud;
        }
        files.push_back(syncFile);
    } while (fin.get() != EOF);

    if (files.size() > 1) files.pop_back();

    fin.close();
    fin.close();

    return true;
}

bool GetData(const std::string inputDir, std::vector<SyncDataFile> &dataset, const bool &flag)
{
    std::string imagesTxt = inputDir + "/image.txt";
    std::string lidarTxt = inputDir + "/lidar.txt";
    std::string syncTxt = inputDir + "/sync.txt";

    const bool synced = not EXIST(imagesTxt);
    bool binocular = false;
    std::vector<std::string> imageNameList, lidarNameList;
    std::vector<SyncDataFile> fileList;

    if (synced)
    {
        if (not ReadSyncFile(syncTxt, fileList, flag)) exit(0);
    } else
    {
        if (not ReadFile(imagesTxt, imageNameList)) exit(0);
        if (not ReadFile(lidarTxt, lidarNameList)) exit(0);
    }

    const size_t size =
            imageNameList.size() > 0 ? imageNameList.size() : fileList.size();

    for (size_t i = 0; i < size; ++i)
    {
        SyncDataFile item;

        if (synced)
        {
            item = fileList.at(i).SetPrefix(inputDir + "/");
            if (not EXIST(item.imageLeft))
            {
                binocular = true;
                item.AddCam01Path();
            }
        } else
        {
            item.imageLeft = inputDir + "/" + imageNameList.at(i);
            item.lidar = inputDir + "/" + lidarNameList[i];
            item.lidarPose = item.lidar;
            item.lidarPose = item.lidarPose.replace(item.lidar.find("lidar"), 5, "slam");
            item.imagePose = item.lidarPose;
        }

        dataset.push_back(item);
    }

    return binocular;
}

bool GetCameraConfig(std::string file, psl::CameraMoudleParam &param)
{
    cv::FileStorage fileStream = cv::FileStorage(file, cv::FileStorage::READ);

    if (not fileStream.isOpened())
    {
        ERROR_PRINT("file not exist <" + file + ">");
        return false;
    }

    // TODO : the exception for lack option
    cv::Mat_<double> kl, dl, pl, rl;
    fileStream["Kl"] >> kl;
    fileStream["Dl"] >> dl;
    fileStream["Pl"] >> pl;
    fileStream["Rl"] >> rl;

    memcpy(param._left_camera[RESOLUTION]._K, kl.data, sizeof(param._left_camera[RESOLUTION]._K));
    memcpy(param._left_camera[RESOLUTION]._R, rl.data, sizeof(param._left_camera[RESOLUTION]._R));
    memcpy(param._left_camera[RESOLUTION]._P, pl.data, sizeof(param._left_camera[RESOLUTION]._P));
    memcpy(param._left_camera[RESOLUTION]._D, dl.data, sizeof(param._left_camera[RESOLUTION]._D));

    // TODO : the exception for lack option
    cv::Mat_<double> kr, dr, pr, rr;
    fileStream["Kr"] >> kr;
    fileStream["Dr"] >> dr;
    fileStream["Pr"] >> pr;
    fileStream["Rr"] >> rr;
    memcpy(param._right_camera[RESOLUTION]._K, kr.data, sizeof(param._right_camera[RESOLUTION]._K));
    memcpy(param._right_camera[RESOLUTION]._R, rr.data, sizeof(param._right_camera[RESOLUTION]._R));
    memcpy(param._right_camera[RESOLUTION]._P, pr.data, sizeof(param._right_camera[RESOLUTION]._P));
    memcpy(param._right_camera[RESOLUTION]._D, dr.data, sizeof(param._right_camera[RESOLUTION]._D));

    fileStream.release();

    return true;
}


void GetPointCloud(cv::Mat image, std::vector<Eigen::Vector3d> &pointCloud)
{
    // 获取图像尺寸
    int rows = image.rows;
    int cols = image.cols;

    // 获取图像三通道的像素值
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            // 获取像素值
            cv::Vec3f pixel = image.at<cv::Vec3f>(i, j);

            // 分别获取三个通道的值
            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];
//            if (std::isnan(blue) || std::isnan(green) || std::isnan(red))
//            {
//                std::cout << pixel << std::endl;
//                continue;
//            }
            pointCloud.push_back(Eigen::Vector3d(pixel[0], pixel[1], pixel[2]));
        }
    }
}

void Show3DPointCloud(std::string &yamlFilePath)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string yamlFile = yamlFilePath;
    cv::FileStorage fileStorageRead = cv::FileStorage(yamlFile, cv::FileStorage::READ);
    cv::Mat imageTT;
    fileStorageRead["depth"] >> imageTT;
    for (int i = 0; i < imageTT.rows; ++i)
    {
        for (int j = 0; j < imageTT.cols; ++j)
        {
            // 获取像素值
            cv::Vec3f pixel = imageTT.at<cv::Vec3f>(i, j);
            cloud->push_back(pcl::PointXYZ(pixel[0], pixel[1], pixel[2]));
        }
    }

    // 可视化点云
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);

    viewer.setRepresentationToWireframeForAllActors(); // 设置线框模式

    viewer.addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

    viewer.addCoordinateSystem(0.2); // 添加坐标轴

    viewer.initCameraParameters();

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
}


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "参数不足！请提供路径和有效数据！" << std::endl;
        return 1;
    }
    std::string inputDir = argv[1];
    std::string saveDir = argv[2];
    psl::CameraMoudleParam param;
    std::string cameraConfigFile = argv[3]; //相机配置文件路径
    GetCameraConfig(cameraConfigFile, param);
    camera_fx = param._right_camera[RESOLUTION]._P[0];
    camera_fy = param._right_camera[RESOLUTION]._P[5];
    camera_cx = param._right_camera[RESOLUTION]._P[2];
    camera_cy = param._right_camera[RESOLUTION]._P[6];
    std::vector<SyncDataFile> dataset;
    GetData(inputDir, dataset, true); // 获取数据集
    const size_t size = dataset.size();
    for (size_t i = 1; i < size; ++i)
    {
        SyncDataFile item = dataset.at(i);
//        std::string imagePath(item.imageLeft);
        std::string imagePath = "/media/xin/data1/data/data_2023_11_21/data_2023_11_21_1/20231121_1145/cam0/53_1700538353654779.jpg";
        cv::Mat depthImage = cv::imread(imagePath, -1);
        PointCloudData cloudPoints;
//        std::string yamlFile = item.pointCloud;
        std::string yamlFile = "/media/xin/data1/data/data_2023_11_21/data_2023_11_21_1/20231121_1145/slam_depth/53_1700538353693783_slam_depth.yaml";
        Show3DPointCloud(yamlFile);
        cv::FileStorage fileStorageRead = cv::FileStorage(yamlFile, cv::FileStorage::READ);
        cv::Mat imageTT;
        fileStorageRead["depth"] >> imageTT;
        cv::Mat resDepth;
        std::vector<Eigen::Vector3d> pointClouds;
        GetPointCloud(imageTT, pointClouds);
        PointCloud2Depth(pointClouds, resDepth, 800, 400);
//        size_t lastSlashPos = imagePath.find_last_of("/\\"); // 查找最后一个路径分隔符的位置
//        std::string fileName = imagePath.substr(lastSlashPos + 1);
//        std::string imageSave_path = saveDir + "/" + std::to_string(i) + "_" + fileName;
//        cv::imwrite("/media/xin/data1/test_data/slam_depth_test/0.png",resDepth);
        cv::imshow("image",depthImage);
        cv::imshow("res", resDepth);
        cv::waitKey();
        break;
    }
    return 0;
}




