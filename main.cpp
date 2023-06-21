/*
* TOGC: triple orthogonal geometrical characteristic
*/

#define PCL_NO_PRECOMPILE

//windows
#include <iostream>
#include <thread>
#include <chrono>
#include <complex>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdio.h>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/geometry.h>
#include <pcl/common/distances.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_representation.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>
#include <pcl/point_representation.h>
#include <pcl/PointIndices.h>

// ours
#include "WKDpre.h"
#include "WKD.h"


pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS(new pcl::PointCloud<pcl::PointXYZ>);  // 原始的源点云
pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT(new pcl::PointCloud<pcl::PointXYZ>);  // 原始的目标点云
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS(new pcl::PointCloud<pcl::PointXYZ>);  // 经过体素滤波后的源点云
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT(new pcl::PointCloud<pcl::PointXYZ>);  // 经过体素滤波后的目标点云
pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);    // 存放ISS提取后的源点云，索引从0开始
pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);    // 存放ISS提取后的目标点云，索引从0开始
pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);       // 存放原先的源点云里关键点的索引
pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);       // 存放原先的目标点云里关键点的索引
std::vector< pcl::Histogram<dimension_wkd> > WKD_features_S;   // TEST1描述子直方图
std::vector< pcl::Histogram<dimension_wkd> > WKD_features_T;   // TEST1描述子直方图


pcl::Correspondences ransaccorr;


// 获取匹配
void getCorr(const pcl::PointIndicesPtr IdxS, const pcl::PointIndicesPtr IdxT, pcl::CorrespondencesPtr corr) {
	std::cout << ">>>>>计算匹配中" << std::endl;
	int matchNum = 2;

	pcl::PointCloud<WKDSigniture> sourceWKDfeature;
	pcl::PointCloud<WKDSigniture> targetWKDfeature;
	for (const auto& WKDfeature : WKD_features_S) {
		WKDSigniture sig;
		for (int i = 0; i < dimension_wkd; ++i) { sig.histogram[i] = WKDfeature.histogram[i]; }
		sourceWKDfeature.push_back(sig);
	}

	for (const auto& WKDfeature : WKD_features_T) {
		WKDSigniture sig;
		for (int i = 0; i < dimension_wkd; ++i) { sig.histogram[i] = WKDfeature.histogram[i]; }
		targetWKDfeature.push_back(sig);
	}

	// 将点云包装成智能指针
	pcl::PointCloud<WKDSigniture>::Ptr sourceWKDfeaturePtr(new pcl::PointCloud<WKDSigniture>(sourceWKDfeature));
	pcl::PointCloud<WKDSigniture>::Ptr targetWKDfeaturePtr(new pcl::PointCloud<WKDSigniture>(targetWKDfeature));

	// 使用KdTree在特征向量空间内寻找最近邻匹配
	pcl::KdTreeFLANN<WKDSigniture> treeS;
	pcl::KdTreeFLANN<WKDSigniture> treeT;
	treeS.setInputCloud(sourceWKDfeaturePtr);
	treeT.setInputCloud(targetWKDfeaturePtr);

	for (size_t i = 0; i < sourceWKDfeaturePtr->size(); i++)
	{
		std::vector<int> corrIdxTmp(matchNum);    // 存放正向最近邻查找的点的索引
		std::vector<float> corrDisTmp(matchNum);  // 存放正向最近邻查找的点的距离
		treeT.nearestKSearch(*sourceWKDfeaturePtr, i, matchNum, corrIdxTmp, corrDisTmp);
		// 匹配加入集合
		for (size_t j = 0; j < matchNum; ++j) {
			pcl::Correspondence corrTmp;
			// cout << "index_query:" << IdxS->indices[i] << "  index_match:" << IdxT->indices[corrIdxTmp[j]] << endl;
			corrTmp.index_query = IdxS->indices[i];
			corrTmp.index_match = IdxT->indices[corrIdxTmp[j]];
			corrTmp.distance = corrDisTmp[j];
			(*corr).push_back(corrTmp);
		}
	}
	std::cout << ">>>>>匹配对数：" << corr->size() << std::endl;
}


// 主函数
int main(int argc, char** argv) {
	//INPUT:
	// 1. 源点云的文件路径
	std::string fnameS = argv[1];
	// 2. 目标点云的文件路径
	std::string fnameT = argv[2];
	// 3. 下采样倍率
	size_t leaf_size = std::stoi(argv[3]);   // 网格下采样倍率


	//=================读取点云文件==================
	WKDpre::readFile(fnameS, origin_cloudS);
	WKDpre::readFile(fnameT, origin_cloudT);
	float resolution = WKDpre::aveDistance(origin_cloudS);
	float resolution2 = WKDpre::aveDistance(origin_cloudT);
	std::cout << ">>>>>源点云平均距离mr:" << resolution << std::endl;
	std::cout << ">>>>>目标点云平均距离mr:" << resolution2 << std::endl;

	
	//==========点云预处理========== 
	
	resolution = leaf_size * resolution;
	WKDpre::WKDpreparation(origin_cloudS, origin_cloudT, cloudS, cloudT, issS, issT, iss_IdxS, iss_IdxT, resolution);

	/*===============计算WKD特征描述子===============*/ 
	auto t = std::chrono::system_clock::now();
	WKDEstimation wkdEstimation;
	wkdEstimation.setInputCloud(cloudS);
	wkdEstimation.setIndices(iss_IdxS);
	wkdEstimation.setResolution(resolution);
	wkdEstimation.setSearchRadius(25);
	wkdEstimation.useWeight3(false);
	wkdEstimation.computeWKD(WKD_features_S);

	wkdEstimation.setInputCloud(cloudT);
	wkdEstimation.setIndices(iss_IdxT);
	wkdEstimation.computeWKD(WKD_features_T);

	auto t1 = std::chrono::system_clock::now();
	std::cout << ">>>>>计算描述子耗时: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;


	//===============RANSAC计算变换矩阵===============
	//(1)获得匹配
	pcl::CorrespondencesPtr corr(new pcl::Correspondences);   // 存储所有的匹配点
	getCorr(iss_IdxS, iss_IdxT, corr);
	
	//(2)RANSAC
	CorrespondenceRANSAC corrRansac;
	corrRansac.setInputSource(cloudS);
	corrRansac.setInputTarget(cloudT);
	corrRansac.setInputCorrespondences(corr);
	corrRansac.setIterationTimes(10000);
	corrRansac.setThreshold(20 * resolution);
	corrRansac.runRANSAC();
	ransaccorr = corrRansac.getRemainCorrespondences();
	std::cout << "ransac筛选后匹配数量：" << ransaccorr.size() << std::endl;

	Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity();
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>::Ptr trans(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>);
	trans->estimateRigidTransformation(*cloudS, *cloudT, ransaccorr, Transform);
	std::cout << "registration Transform without ICP:" << std::endl;
	std::cout << Transform << std::endl;


	//===============对源点云刚体变换===============
	pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloudS(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloudS, *reg_cloudS, Transform);
	
	bool if_icp = 0;    // 是否需要ICP精配准(低质量点云如果使用ICP反而效果差)
	if (if_icp)
	{
		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
		tree1->setInputCloud(reg_cloudS);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
		tree2->setInputCloud(cloudT);
		icp.setSearchMethodSource(tree1);
		icp.setSearchMethodTarget(tree2);
		icp.setInputSource(reg_cloudS);
		icp.setInputTarget(cloudT);
		icp.setTransformationEpsilon(1e-12);
		//icp.setEuclideanFitnessEpsilon(0.04);
		icp.setMaxCorrespondenceDistance(resolution * 30);
		icp.setRANSACOutlierRejectionThreshold(resolution * 1000);
		icp.setMaximumIterations(30);
		pcl::PointCloud<pcl::PointXYZ> Final;
		icp.align(Final);
		Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
		pcl::transformPointCloud(*reg_cloudS, *reg_cloudS, icp_transform);
		// pcl::io::savePLYFile("model/fine.ply", *reg_cloudS);
		std::cout << "fined registration" << std::endl;
		auto t2 = std::chrono::system_clock::now();
		std::cout << ">>>>>计算描述子耗时: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
		Eigen::Matrix4f T = Transform * icp_transform;
		std::cout << "registration Transform with ICP:" << std::endl;
		std::cout << T << std::endl;
	}
	

	//=============显示两片配准前的点云以及连线==============
	auto ShowTwoPointCloud = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		// 设定点云颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 0, 0, 255);    // 设定变换后的源点云的颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 255, 0, 0);   // 设定目标点云的颜色

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("two origin cloud"));
		viewer->setBackgroundColor(255, 255, 255);    // 设置背景颜色为白色(255,255,255)
		viewer->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");   // 添加源点云，id为source cloud
		viewer->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");   // 添加目标点云，id为target cloud
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");   // 对id为source cloud的点云设置点的大小为1
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");   // 对id为target cloud的点云设置点的大小为1
		for (const auto& correspondence : ransaccorr)
		{
			const pcl::PointXYZ& point1 = cloudS->points[correspondence.index_query];
			const pcl::PointXYZ& point2 = cloudT->points[correspondence.index_match];
			viewer->addLine(point1, point2, 0, 255, 0, "line_" + std::to_string(correspondence.index_query));
		}
		viewer->spin();

	};

	// 显示配准后的点云
	auto ShowRegistrationResult = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		// 设定点云颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 0, 0, 255);    // 设定变换后的源点云的颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 255, 0, 0);   // 设定目标点云的颜色

		// 显示体素网格滤波后的点云以及提取的关键点
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("two origin cloud"));
		viewer->setBackgroundColor(255, 255, 255);    // 设置背景颜色为白色(255,255,255)
		viewer->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");   // 添加源点云，id为source cloud
		viewer->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");   // 添加目标点云，id为target cloud
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");   // 对id为source cloud的点云设置点的大小为1
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");   // 对id为target cloud的点云设置点的大小为1
		viewer->spin();
	};

	std::thread vis_thread(ShowTwoPointCloud, cloudS, cloudT);
	ShowRegistrationResult(reg_cloudS, cloudT);

	return 0;
}