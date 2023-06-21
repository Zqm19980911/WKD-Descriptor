#pragma once

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/principal_curvatures.h>

namespace WKDpre
{
	// 读取点云文件
	void readFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

	// 计算点云平均距离mr
	float aveDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

	// 体素滤波
	void voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG, const double& inlTh);

	// ISS关键点提取
	void issKeyPointExtration(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ISS, pcl::PointIndicesPtr ISS_Idx, const double& resolution);

	// 点云预处理工作
	void WKDpreparation(const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::PointIndicesPtr iss_IdxS, pcl::PointIndicesPtr iss_IdxT, const double& resolution);

}
