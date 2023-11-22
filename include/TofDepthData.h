#pragma once

#include <cstdint>
#include <vector>

struct PointCloudData {
  double rotation[4];        //曝光时间
  double position[3];                //序列号，serial number
  int rows;                    //高
  int cols;                   //宽
  std::uint64_t time;  //时间戳
  std::string tf;  //类型
  std::vector<Eigen::Vector3d> pointData;
};
