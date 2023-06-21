#include "WKDpre.h"

#include <iostream>
#include <random>
#include <set>
#include <ctime>

#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/common/generate.h>       // �ṩpcl::common::CloudGenerator   ���ɸ�˹����
#include <pcl/common/random.h>         // �ṩpcl::common::NormalGenerator  ���ɸ�˹����

// ��ȡ�����ļ�
void WKDpre::readFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	// pcd�ļ���ȡ
	if (filename.substr(filename.find_last_of('.') + 1) == "pcd") {
		if (pcl::io::loadPCDFile(filename, *cloud) < 0) { PCL_ERROR("\a->file not exist��\n"); }
		else { std::cout << ">>>>>��ȡ�����ļ� " << filename << " �ɹ�!" << std::endl; }
	}
	// ply�ļ���ȡ
	if (filename.substr(filename.find_last_of('.') + 1) == "ply") {
		if (pcl::io::loadPLYFile(filename, *cloud) < 0) { PCL_ERROR("\a->file not exist��\n"); }
		else { std::cout << ">>>>>��ȡ�����ļ� " << filename << " �ɹ�!" << std::endl; }
	}
	std::cout << ">>>>>�����ļ�" << filename << "�ܵ���:" << cloud->points.size() << std::endl;
	std::cout << " " << std::endl;
}


// �����ƽ������
float WKDpre::aveDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	float res = 0.0;//����ƽ������
	int n_points = 0;//�����¼��������
	int nres;//���������������
			 //vector��˳��������һ�֡�vector �ǿɱ䳤�Ķ�̬����
	std::vector<int> indices(2);//����һ������2��int�������ݵ�vector //����һ����̬���飬�洢��ѯ��������� //�ȼ��������д��� using std::vector; vector<int> indices(2);
	std::vector<float> sqr_distances(2);//�洢���ڵ��Ӧƽ������

	for (size_t i = 0; i < cloud->size(); ++i) {  //ѭ������ÿһ����
		if (!pcl_isfinite(cloud->points[i].x)) {  //pcl_isfinite��������һ������ֵ�����ĳ��ֵ�ǲ���������ֵ
			continue;
		}
		// Considering the second neighbor since the first is the point itself.
		// kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) 
		// ����ִ�� K ���ڲ��ҵĳ�Ա���������У���kΪ1��ʱ�򣬾����������������k����1��ʱ�򣬾��Ƕ��������������˴�kΪ2��
		// KΪҪ�������ھ�������k the number of neighbors to search for��
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);//��������ֵ�������ҵ���������������return number of neighbors found
		if (nres == 2) {   //���Ϊ������֮��
			res += sqrt(sqr_distances[1]);//sqrt()����������sqr_distances[1]�Ŀ�ƽ����
			//std::cout << "sqr_distances[1]��" << sqr_distances[1] << std::endl;//��ӡ���ٽ�������ƽ��ֵ
			++n_points;
		}
	}
	if (n_points != 0) {
		res /= n_points;
	}
	return res;
}


// �����˲�
void WKDpre::voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG, const double& inlTh)
{
	// �����˲�����
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);               // �������
	vg.setLeafSize(inlTh, inlTh, inlTh);   // ������С���ر߳�
	// �����˲��������˲������cloudVG
	vg.filter(*cloudVG);
}


// ISS��ȡ�ؼ���
void WKDpre::issKeyPointExtration(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ISS, pcl::PointIndicesPtr ISS_Idx, const double& resolution)
{
	double iss_salient_radius_ = 6 * resolution;   // ����Э��������������뾶  �������뾶���ָ�С���԰ѹؼ�����������
	double iss_non_max_radius_ = 4 * resolution;   // �Ǽ���ֵ����Ӧ���㷨�İ뾶
	//double iss_non_max_radius_ = 2 * resolution;//for office
	//double iss_non_max_radius_ = 9 * resolution;//for railway
	double iss_gamma_21_(0.975);   // �ڶ����͵�һ������ֵ֮�ȵ�����
	double iss_gamma_32_(0.975);   // �������͵ڶ�������ֵ֮�ȵ�����
	double iss_min_neighbors_(4);  // Ӧ�÷Ǽ���ֵ�����㷨ʱ�����ñ����ҵ�����С�ھ�����
	int iss_threads_(8);          // �߳���

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

	// ����ISS������̽��������
	iss_detector.setInputCloud(cloud);
	iss_detector.setSearchMethod(tree);                  // ������������ΪKdTree
	iss_detector.setSalientRadius(iss_salient_radius_);  // �������ڼ���Э��������������뾶
	iss_detector.setNonMaxRadius(iss_non_max_radius_);   // ���÷Ǽ���ֵ����Ӧ���㷨�İ뾶
	iss_detector.setThreshold21(iss_gamma_21_);          // ���õڶ����͵�һ������ֵ֮�ȵ�����
	iss_detector.setThreshold32(iss_gamma_32_);          // ���õ������͵ڶ�������ֵ֮�ȵ�����
	iss_detector.setMinNeighbors(iss_min_neighbors_);    // Ӧ�÷Ǽ���ֵ�����㷨ʱ�����ñ����ҵ�����С�ھ�����
	iss_detector.setNumberOfThreads(iss_threads_);       // ��ʼ��������������Ҫʹ�õ��߳���
	// ����ISS����ֵ�㼯
	iss_detector.compute(*ISS);
	// �洢�ؼ�������
	ISS_Idx->indices = iss_detector.getKeypointsIndices()->indices;
	ISS_Idx->header = iss_detector.getKeypointsIndices()->header;

}


// ����Ԥ��������
void WKDpre::WKDpreparation(const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::PointIndicesPtr iss_IdxS, pcl::PointIndicesPtr iss_IdxT, const double& resolution)
{
	auto t = std::chrono::system_clock::now();
	// (1)�²���
	WKDpre::voxelGridFilter(origin_cloudS, cloudS, resolution);    // �����²���
	WKDpre::voxelGridFilter(origin_cloudT, cloudT, resolution);    // �����²���
	auto t1 = std::chrono::system_clock::now();
	std::cout << ">>>>>�²�����Դ�����ܵ���:" << cloudS->points.size() << std::endl;
	std::cout << ">>>>>�²�����Ŀ������ܵ���:" << cloudT->points.size() << std::endl;
	std::cout << ">>>>>�����²�����ʱ: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
	// (2)ISS�ؼ�����ȡ
	WKDpre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);  // ʹ��ISS��ȡ��
	WKDpre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);  // ʹ��ISS��ȡ��
	auto t2 = std::chrono::system_clock::now();
	std::cout << ">>>>>Դ���ƹؼ�������:" << iss_IdxS->indices.size() << std::endl;
	std::cout << ">>>>>Ŀ����ƹؼ�������:" << iss_IdxT->indices.size() << std::endl;
	std::cout << ">>>>>��ȡ�ؼ����ʱ: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
	std::cout << " " << std::endl;

}
