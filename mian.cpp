#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <eigen3/Eigen/Dense>

double camera_factor = 100;
double camera_cx = 285.22;// = 325.5;
double camera_cy = 285.22;// = 253.5;
double camera_fx = 316.045;// = 518.0;
double camera_fy = 316.045;// = 519.0;

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

int main(){
    std::string image = "/media/xin/data1/data/parker_data_2023_08_22/result/CREStereo_MiDaS/CREStereo_big_object_100_tof/scale_tof/louti/data_2023_0822_2/20210223_1355/cam0/14_1614045310895639.png";
    cv::Mat depthImage = cv::imread(image, -1);
//    cv::imshow("image",depthImage);
    int width = depthImage.cols;
    int height = depthImage.rows;
    std::vector<Eigen::Vector3d> cloudPoints;
    Depth2PointCloud(depthImage,cloudPoints);
    cv::Mat resDepth;
    PointCloud2Depth(cloudPoints,resDepth,width,height);
    cv::imshow("res",resDepth);
    cv::imwrite("/media/xin/data1/test_data/depth_cloudpoint_test/no_100.png",resDepth);
    cv::waitKey();
}