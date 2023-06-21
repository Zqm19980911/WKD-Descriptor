#include "WKDpre.h"

#include <iostream>
#include <random>
#include <set>
#include <ctime>

#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/common/generate.h>       // 提供pcl::common::CloudGenerator   生成高斯数据
#include <pcl/common/random.h>         // 提供pcl::common::NormalGenerator  生成高斯数据

// 读取点云文件
void WKDpre::readFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	// pcd文件读取
	if (filename.substr(filename.find_last_of('.') + 1) == "pcd") {
		if (pcl::io::loadPCDFile(filename, *cloud) < 0) { PCL_ERROR("\a->file not exist！\n"); }
		else { std::cout << ">>>>>读取点云文件 " << filename << " 成功!" << std::endl; }
	}
	// ply文件读取
	if (filename.substr(filename.find_last_of('.') + 1) == "ply") {
		if (pcl::io::loadPLYFile(filename, *cloud) < 0) { PCL_ERROR("\a->file not exist！\n"); }
		else { std::cout << ">>>>>读取点云文件 " << filename << " 成功!" << std::endl; }
	}
	std::cout << ">>>>>点云文件" << filename << "总点数:" << cloud->points.size() << std::endl;
	std::cout << " " << std::endl;
}


// 求点云平均距离
float WKDpre::aveDistance(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

	pcl::KdTreeFLANN<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	float res = 0.0;//定义平均距离
	int n_points = 0;//定义记录点云数量
	int nres;//定义邻域查找数量
			 //vector是顺序容器的一种。vector 是可变长的动态数组
	std::vector<int> indices(2);//创建一个包含2个int类型数据的vector //创建一个动态数组，存储查询点近邻索引 //等价于这两行代码 using std::vector; vector<int> indices(2);
	std::vector<float> sqr_distances(2);//存储近邻点对应平方距离

	for (size_t i = 0; i < cloud->size(); ++i) {  //循环遍历每一个点
		if (!pcl_isfinite(cloud->points[i].x)) {  //pcl_isfinite函数返回一个布尔值，检查某个值是不是正常数值
			continue;
		}
		// Considering the second neighbor since the first is the point itself.
		// kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) 
		// 这是执行 K 近邻查找的成员函数（其中，当k为1的时候，就是最近邻搜索。当k大于1的时候，就是多个最近邻搜索，此处k为2）
		// K为要搜索的邻居数量（k the number of neighbors to search for）
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);//函数返回值（返回找到的邻域数量），return number of neighbors found
		if (nres == 2) {   //如果为两个点之间
			res += sqrt(sqr_distances[1]);//sqrt()函数，返回sqr_distances[1]的开平方数
			//std::cout << "sqr_distances[1]：" << sqr_distances[1] << std::endl;//打印与临近点距离的平方值
			++n_points;
		}
	}
	if (n_points != 0) {
		res /= n_points;
	}
	return res;
}


// 体素滤波
void WKDpre::voxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVG, const double& inlTh)
{
	// 设置滤波参数
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);               // 输入点云
	vg.setLeafSize(inlTh, inlTh, inlTh);   // 设置最小体素边长
	// 进行滤波并保存滤波结果到cloudVG
	vg.filter(*cloudVG);
}


// ISS提取关键点
void WKDpre::issKeyPointExtration(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr ISS, pcl::PointIndicesPtr ISS_Idx, const double& resolution)
{
	double iss_salient_radius_ = 6 * resolution;   // 计算协方差矩阵的球邻域半径  这两个半径数字改小可以把关键点数量增多
	double iss_non_max_radius_ = 4 * resolution;   // 非极大值抑制应用算法的半径
	//double iss_non_max_radius_ = 2 * resolution;//for office
	//double iss_non_max_radius_ = 9 * resolution;//for railway
	double iss_gamma_21_(0.975);   // 第二个和第一个特征值之比的上限
	double iss_gamma_32_(0.975);   // 第三个和第二个特征值之比的上限
	double iss_min_neighbors_(4);  // 应用非极大值抑制算法时，设置必须找到的最小邻居数量
	int iss_threads_(8);          // 线程数

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

	// 设置ISS特征点探测器参数
	iss_detector.setInputCloud(cloud);
	iss_detector.setSearchMethod(tree);                  // 设置搜索方法为KdTree
	iss_detector.setSalientRadius(iss_salient_radius_);  // 设置用于计算协方差矩阵的球邻域半径
	iss_detector.setNonMaxRadius(iss_non_max_radius_);   // 设置非极大值抑制应用算法的半径
	iss_detector.setThreshold21(iss_gamma_21_);          // 设置第二个和第一个特征值之比的上限
	iss_detector.setThreshold32(iss_gamma_32_);          // 设置第三个和第二个特征值之比的上限
	iss_detector.setMinNeighbors(iss_min_neighbors_);    // 应用非极大值抑制算法时，设置必须找到的最小邻居数量
	iss_detector.setNumberOfThreads(iss_threads_);       // 初始化调度器并设置要使用的线程数
	// 计算ISS特征值点集
	iss_detector.compute(*ISS);
	// 存储关键点索引
	ISS_Idx->indices = iss_detector.getKeypointsIndices()->indices;
	ISS_Idx->header = iss_detector.getKeypointsIndices()->header;

}


// 点云预处理流程
void WKDpre::WKDpreparation(const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudS, const pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudS, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT, pcl::PointCloud<pcl::PointXYZ>::Ptr issS, pcl::PointCloud<pcl::PointXYZ>::Ptr issT, pcl::PointIndicesPtr iss_IdxS, pcl::PointIndicesPtr iss_IdxT, const double& resolution)
{
	auto t = std::chrono::system_clock::now();
	// (1)下采样
	WKDpre::voxelGridFilter(origin_cloudS, cloudS, resolution);    // 网格下采样
	WKDpre::voxelGridFilter(origin_cloudT, cloudT, resolution);    // 网格下采样
	auto t1 = std::chrono::system_clock::now();
	std::cout << ">>>>>下采样后源点云总点数:" << cloudS->points.size() << std::endl;
	std::cout << ">>>>>下采样后目标点云总点数:" << cloudT->points.size() << std::endl;
	std::cout << ">>>>>点云下采样耗时: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t).count()) / 1000.0 << std::endl;
	// (2)ISS关键点提取
	WKDpre::issKeyPointExtration(cloudS, issS, iss_IdxS, resolution);  // 使用ISS提取器
	WKDpre::issKeyPointExtration(cloudT, issT, iss_IdxT, resolution);  // 使用ISS提取器
	auto t2 = std::chrono::system_clock::now();
	std::cout << ">>>>>源点云关键点数量:" << iss_IdxS->indices.size() << std::endl;
	std::cout << ">>>>>目标点云关键点数量:" << iss_IdxT->indices.size() << std::endl;
	std::cout << ">>>>>提取关键点耗时: " << double(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()) / 1000.0 << std::endl;
	std::cout << " " << std::endl;

}
