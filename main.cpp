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


pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS(new pcl::PointCloud<pcl::PointXYZ>);  // ԭʼ��Դ����
pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT(new pcl::PointCloud<pcl::PointXYZ>);  // ԭʼ��Ŀ�����
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS(new pcl::PointCloud<pcl::PointXYZ>);  // ���������˲����Դ����
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT(new pcl::PointCloud<pcl::PointXYZ>);  // ���������˲����Ŀ�����
pcl::PointCloud<pcl::PointXYZ>::Ptr issS(new pcl::PointCloud<pcl::PointXYZ>);    // ���ISS��ȡ���Դ���ƣ�������0��ʼ
pcl::PointCloud<pcl::PointXYZ>::Ptr issT(new pcl::PointCloud<pcl::PointXYZ>);    // ���ISS��ȡ���Ŀ����ƣ�������0��ʼ
pcl::PointIndicesPtr iss_IdxS(new pcl::PointIndices);       // ���ԭ�ȵ�Դ������ؼ��������
pcl::PointIndicesPtr iss_IdxT(new pcl::PointIndices);       // ���ԭ�ȵ�Ŀ�������ؼ��������
std::vector< pcl::Histogram<dimension_wkd> > WKD_features_S;   // TEST1������ֱ��ͼ
std::vector< pcl::Histogram<dimension_wkd> > WKD_features_T;   // TEST1������ֱ��ͼ


pcl::Correspondences ransaccorr;


// ��ȡƥ��
void getCorr(const pcl::PointIndicesPtr IdxS, const pcl::PointIndicesPtr IdxT, pcl::CorrespondencesPtr corr) {
	std::cout << ">>>>>����ƥ����" << std::endl;
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

	// �����ư�װ������ָ��
	pcl::PointCloud<WKDSigniture>::Ptr sourceWKDfeaturePtr(new pcl::PointCloud<WKDSigniture>(sourceWKDfeature));
	pcl::PointCloud<WKDSigniture>::Ptr targetWKDfeaturePtr(new pcl::PointCloud<WKDSigniture>(targetWKDfeature));

	// ʹ��KdTree�����������ռ���Ѱ�������ƥ��
	pcl::KdTreeFLANN<WKDSigniture> treeS;
	pcl::KdTreeFLANN<WKDSigniture> treeT;
	treeS.setInputCloud(sourceWKDfeaturePtr);
	treeT.setInputCloud(targetWKDfeaturePtr);

	for (size_t i = 0; i < sourceWKDfeaturePtr->size(); i++)
	{
		std::vector<int> corrIdxTmp(matchNum);    // �����������ڲ��ҵĵ������
		std::vector<float> corrDisTmp(matchNum);  // �����������ڲ��ҵĵ�ľ���
		treeT.nearestKSearch(*sourceWKDfeaturePtr, i, matchNum, corrIdxTmp, corrDisTmp);
		// ƥ����뼯��
		for (size_t j = 0; j < matchNum; ++j) {
			pcl::Correspondence corrTmp;
			// cout << "index_query:" << IdxS->indices[i] << "  index_match:" << IdxT->indices[corrIdxTmp[j]] << endl;
			corrTmp.index_query = IdxS->indices[i];
			corrTmp.index_match = IdxT->indices[corrIdxTmp[j]];
			corrTmp.distance = corrDisTmp[j];
			(*corr).push_back(corrTmp);
		}
	}
	std::cout << ">>>>>ƥ�������" << corr->size() << std::endl;
}


// ������
int main(int argc, char** argv) {
	//INPUT:
	// 1. Դ���Ƶ��ļ�·��
	std::string fnameS = argv[1];
	// 2. Ŀ����Ƶ��ļ�·��
	std::string fnameT = argv[2];
	// 3. �²�������
	size_t leaf_size = std::stoi(argv[3]);   // �����²�������


	//=================��ȡ�����ļ�==================
	WKDpre::readFile(fnameS, origin_cloudS);
	WKDpre::readFile(fnameT, origin_cloudT);
	float resolution = WKDpre::aveDistance(origin_cloudS);
	float resolution2 = WKDpre::aveDistance(origin_cloudT);
	std::cout << ">>>>>Դ����ƽ������mr:" << resolution << std::endl;
	std::cout << ">>>>>Ŀ�����ƽ������mr:" << resolution2 << std::endl;

	
	//==========����Ԥ����========== 
	
	resolution = leaf_size * resolution;
	WKDpre::WKDpreparation(origin_cloudS, origin_cloudT, cloudS, cloudT, issS, issT, iss_IdxS, iss_IdxT, resolution);

	/*===============����WKD����������===============*/ 
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
	std::cout << ">>>>>���������Ӻ�ʱ: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;


	//===============RANSAC����任����===============
	//(1)���ƥ��
	pcl::CorrespondencesPtr corr(new pcl::Correspondences);   // �洢���е�ƥ���
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
	std::cout << "ransacɸѡ��ƥ��������" << ransaccorr.size() << std::endl;

	Eigen::Matrix4f Transform = Eigen::Matrix4f::Identity();
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>::Ptr trans(new pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float>);
	trans->estimateRigidTransformation(*cloudS, *cloudT, ransaccorr, Transform);
	std::cout << "registration Transform without ICP:" << std::endl;
	std::cout << Transform << std::endl;


	//===============��Դ���Ƹ���任===============
	pcl::PointCloud<pcl::PointXYZ>::Ptr reg_cloudS(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloudS, *reg_cloudS, Transform);
	
	bool if_icp = 0;    // �Ƿ���ҪICP����׼(�������������ʹ��ICP����Ч����)
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
		std::cout << ">>>>>���������Ӻ�ʱ: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
		Eigen::Matrix4f T = Transform * icp_transform;
		std::cout << "registration Transform with ICP:" << std::endl;
		std::cout << T << std::endl;
	}
	

	//=============��ʾ��Ƭ��׼ǰ�ĵ����Լ�����==============
	auto ShowTwoPointCloud = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		// �趨������ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 0, 0, 255);    // �趨�任���Դ���Ƶ���ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 255, 0, 0);   // �趨Ŀ����Ƶ���ɫ

		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("two origin cloud"));
		viewer->setBackgroundColor(255, 255, 255);    // ���ñ�����ɫΪ��ɫ(255,255,255)
		viewer->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");   // ���Դ���ƣ�idΪsource cloud
		viewer->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");   // ���Ŀ����ƣ�idΪtarget cloud
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");   // ��idΪsource cloud�ĵ������õ�Ĵ�СΪ1
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");   // ��idΪtarget cloud�ĵ������õ�Ĵ�СΪ1
		for (const auto& correspondence : ransaccorr)
		{
			const pcl::PointXYZ& point1 = cloudS->points[correspondence.index_query];
			const pcl::PointXYZ& point2 = cloudT->points[correspondence.index_match];
			viewer->addLine(point1, point2, 0, 255, 0, "line_" + std::to_string(correspondence.index_query));
		}
		viewer->spin();

	};

	// ��ʾ��׼��ĵ���
	auto ShowRegistrationResult = [](pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT)
	{
		// �趨������ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorS(cloudS, 0, 0, 255);    // �趨�任���Դ���Ƶ���ɫ
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colorT(cloudT, 255, 0, 0);   // �趨Ŀ����Ƶ���ɫ

		// ��ʾ���������˲���ĵ����Լ���ȡ�Ĺؼ���
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("two origin cloud"));
		viewer->setBackgroundColor(255, 255, 255);    // ���ñ�����ɫΪ��ɫ(255,255,255)
		viewer->addPointCloud<pcl::PointXYZ>(cloudS, colorS, "source cloud");   // ���Դ���ƣ�idΪsource cloud
		viewer->addPointCloud<pcl::PointXYZ>(cloudT, colorT, "target cloud");   // ���Ŀ����ƣ�idΪtarget cloud
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");   // ��idΪsource cloud�ĵ������õ�Ĵ�СΪ1
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");   // ��idΪtarget cloud�ĵ������õ�Ĵ�СΪ1
		viewer->spin();
	};

	std::thread vis_thread(ShowTwoPointCloud, cloudS, cloudT);
	ShowRegistrationResult(reg_cloudS, cloudT);

	return 0;
}