#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <eigen3/Eigen/Dense>
#include <unistd.h>

double camera_factor = 100;
double camera_cx = 285.22;// = 325.5;
double camera_cy = 285.22;// = 253.5;
double camera_fx = 316.045;// = 518.0;
double camera_fy = 316.045;// = 519.0;

#define ERROR_PRINT(x) std::cout << "" << (x) << "" << std::endl

bool ReadFile(std::string srcFile, std::vector<std::string> &image_files) {
    if (not access(srcFile.c_str(), 0) == 0) {
        ERROR_PRINT("no such File (" + srcFile + ")");
        return false;
    }

    std::ifstream fin(srcFile.c_str());

    if (!fin.is_open()) {
        ERROR_PRINT("read file error (" + srcFile + ")");
        return false;
    }

    std::string s;
    while (getline(fin, s)) {
        image_files.push_back(s);
    }

    fin.close();

    return true;
}

void GetImageData(const std::string inputDir, std::vector<std::string> &dataset) {
    std::string imagesTxt = inputDir + "/image_paths.txt";
    std::vector<std::string> imageNameList;

    if (not ReadFile(imagesTxt, imageNameList)) exit(0);
    const size_t size =
            imageNameList.size() > 0 ? imageNameList.size() : 0;

    for (size_t i = 0; i < size; ++i) {
        std::string imagePath = inputDir + "/" + imageNameList.at(i);
        dataset.push_back(imagePath);
    }
}



void Depth2PointCloud(const cv::Mat &depth, std::vector<Eigen::Vector3d> &cloud) {
    for (int m = 0; m < depth.rows; m++)
        for (int n = 0; n < depth.cols; n++) {
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


void PointCloud2Depth(const std::vector<Eigen::Vector3d>& pointCloud, cv::Mat& depthImage, int width, int height)
{
    // 创建深度图像
    depthImage = cv::Mat::zeros(height, width, CV_16U);

    // 遍历点云数据
    for (const auto& point : pointCloud)
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
            unsigned short depthValue = static_cast<unsigned short>(z * camera_factor); // 假设深度单位为米，转换为厘米
            depthImage.at<uint16_t>(v, u) = depthValue;
        }
    }
}

int main(int argc, char *argv[]){
    if (argc < 2) {
        std::cout << "参数不足！请提供路径和有效数据！" << std::endl;
        return 1;
    }
    std::string inputDir = argv[1];
    std::string saveDir = argv[2];
    std::vector<std::string> dataset;
    GetImageData(inputDir, dataset);
    const size_t size = dataset.size();
    for (size_t i = 0; i < size; ++i)
    {
        std::string imagePath = dataset.at(i);
        std::cout << imagePath << std::endl;
        cv::Mat depthImage = cv::imread(imagePath, -1);
        int width = depthImage.cols;
        int height = depthImage.rows;
        std::vector<Eigen::Vector3d> cloudPoints;
        Depth2PointCloud(depthImage,cloudPoints);
        cv::Mat resDepth;
        PointCloud2Depth(cloudPoints,resDepth,width,height);
        size_t lastSlashPos = imagePath.find_last_of("/\\"); // 查找最后一个路径分隔符的位置
        std::string fileName = imagePath.substr(lastSlashPos + 1);
        std::string imageSave_path = saveDir + "/" + std::to_string(i) + "_" + fileName;
        cv::imwrite(imageSave_path,resDepth);
//        cv::imshow("res",resDepth);
//        cv::waitKey();
    }
    return 0;
}