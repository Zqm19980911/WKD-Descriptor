#include "WKD.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/normal_3d_omp.h>


// 高斯核函数
double WKDEstimation::gaussian(const double& u) {
	return 1.0 / sqrt(2.0 * M_PI) * exp(-(u * u) / 2.0);
}


/* *************************for class ThreePointsFeatureEstimation************************** */
void WKDEstimation::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	cloud_in_ = in;
}


// 设置索引
void WKDEstimation::setIndices(const pcl::PointIndicesPtr& indicesn)
{
	// 假如输入的索引列表大于0，那么就把indices_in设置为输入的索引，并将indices_flag设置为true
	if (indicesn->indices.size() > 0)
	{
		indices_in_ = indicesn;
		indices_flag = true;
	}
	else    // 反之，如果输入的索引列表是空，那么就把indices_flag设置为false，在后续会使用输入点云的所有点进行计算
	{
		indices_flag = false;
	}
}


// 设置分辨率
void WKDEstimation::setResolution(float res)
{
	resolution_ = res;
}


// 设置搜索半径
void WKDEstimation::setSearchRadius(float radius)
{
	searchRadius_ = radius;
}


// 是否使用权重3:邻域局部拥挤度
void WKDEstimation::useWeight3(bool flag) {
	useWeight3_ = flag;
}


void WKDEstimation::computeLRF2Weight(int i, float radius, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, std::vector<Eigen::Vector3f>& XYZaxis)
{
	int keyPointIndex = indices_in_->indices[i];

	Eigen::Vector3f zaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f xaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f yaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f keyPointToNeib; // 关键点到邻域点的向量
	Eigen::Vector3f keyPointToNeibProject; // 关键点到邻域点的向量在L平面的投影

	Eigen::Vector3f sum;   // 计算Z轴二义性的加和量
	Eigen::Vector3f qp;   // 由Q指向P的向量，用于计算Z轴二义性家和量

	float w1, w2;   // 3个权重系数

	// 对于每一个关键点，计算三轴
	// (1)Z轴：邻域点子集的法向量(PCL计算出的是单位向量)  
	zaxis_vector << cloud_normals->points[i].normal_x,
		cloud_normals->points[i].normal_y,
		cloud_normals->points[i].normal_z;

	// 确定Z轴方向，消除二义性
	sum.setZero();
	for (auto& index : pointidxRadiusSearch) {
		qp << cloud_in_->points[keyPointIndex].x - cloud_in_->points[index].x,
			cloud_in_->points[keyPointIndex].y - cloud_in_->points[index].y,
			cloud_in_->points[keyPointIndex].z - cloud_in_->points[index].z;
		sum += qp;
	}
	if (zaxis_vector.dot(sum) < 0) { zaxis_vector *= -1.0; }

	// (2)X轴：将关键点到每个邻域点的向量在L平面上的投影加权叠加后，获得的就是X轴
	xaxis_vector.setZero();   // 定义一个加和的初始量
	for (auto& index : pointidxRadiusSearch)
	{
		/* 对于每一个在半径范围内的邻域点执行如下操作 */
		// 计算由关键点指向邻域点的向量
		keyPointToNeib << cloud_in_->points[index].x - cloud_in_->points[keyPointIndex].x,
			cloud_in_->points[index].y - cloud_in_->points[keyPointIndex].y,
			cloud_in_->points[index].z - cloud_in_->points[keyPointIndex].z;
		//if (sqrt(keyPointToNeib[0] * keyPointToNeib[0] + keyPointToNeib[1] * keyPointToNeib[1] + keyPointToNeib[2] * keyPointToNeib[2]) < 0.85 * radius)
		//{
		//	continue;
		//}

		// 第一个权重系数w1=(radius-d)^2  q点到p点距离相关的权重，距离越远，d越大，权重系数w1越小
		w1 = pow(radius -
			sqrt(
				pow(keyPointToNeib[0], 2) + pow(keyPointToNeib[1], 2) + pow(keyPointToNeib[2], 2)
			), 2);

		// 第二个权重系数w2=(pq·z)^2      q点到L平面投影距离相关的权重，投影距离越大，权重系数w2越大
		w2 = pow(keyPointToNeib.dot(zaxis_vector), 2);   // 向量的点乘是一个向量在另一个向量方向上的投影长度

		// 计算关键点p到邻域点q的向量在L平面的投影向量
		keyPointToNeibProject = keyPointToNeib - (keyPointToNeib.dot(zaxis_vector)) * zaxis_vector;  // 关键点到邻域点在L平面的投影向量
		xaxis_vector += keyPointToNeibProject * w1 * w2;
	}

	// 将X坐标轴向量归一化
	xaxis_vector = xaxis_vector /
		sqrt(pow(xaxis_vector[0], 2) + pow(xaxis_vector[1], 2) + pow(xaxis_vector[2], 2));

	// (3)Y轴：将Z轴与X轴叉乘后，获得的就是Y轴
	yaxis_vector = zaxis_vector.cross(xaxis_vector);

	// 存储XYZ三轴信息
	XYZaxis.push_back(xaxis_vector);
	XYZaxis.push_back(yaxis_vector);
	XYZaxis.push_back(zaxis_vector);
}


void WKDEstimation::computeLRF3Weight(int i, float radius, const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, const std::vector<float>& pointRadiusSquaredDistance, std::vector<Eigen::Vector3f>& XYZaxis)
{
	int keyPointIndex = indices_in_->indices[i];

	Eigen::Vector3f zaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f xaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f yaxis_vector;   // 关键点LRF的Z轴向量
	Eigen::Vector3f keyPointToNeib; // 关键点到邻域点的向量
	Eigen::Vector3f keyPointToNeibProject; // 关键点到邻域点的向量在L平面的投影

	Eigen::Vector3f sum;   // 计算Z轴二义性的加和量
	Eigen::Vector3f qp;   // 由Q指向P的向量，用于计算Z轴二义性家和量

	float w1, w2, w3;   // 3个权重系数
	std::vector<int> w3RadiusSearch;          // 存储邻近点的索引
	std::vector<float> w3RadiusSquaredDistance;  // 存储邻近点的距离

	// 对于每一个关键点，计算三轴
	// (1)Z轴：邻域点子集的法向量(PCL计算出的是单位向量)  
	zaxis_vector << cloud_normals->points[i].normal_x,
		cloud_normals->points[i].normal_y,
		cloud_normals->points[i].normal_z;

	// 确定Z轴方向，消除二义性
	sum.setZero();
	for (int j = 0; j < pointidxRadiusSearch.size(); ++j) {
		qp << cloud_in_->points[keyPointIndex].x - cloud_in_->points[pointidxRadiusSearch[j]].x,
			cloud_in_->points[keyPointIndex].y - cloud_in_->points[pointidxRadiusSearch[j]].y,
			cloud_in_->points[keyPointIndex].z - cloud_in_->points[pointidxRadiusSearch[j]].z;
		sum += qp;
	}
	if (zaxis_vector.dot(sum) >= 0) { zaxis_vector *= -1.0; }

	// (2)X轴：将关键点到每个邻域点的向量在L平面上的投影加权叠加后，获得的就是X轴
	xaxis_vector.setZero();   // 定义一个加和的初始量
	int temp = 0;
	float dis;
	bool flag_t = false;
	for (auto& index : pointidxRadiusSearch)
	{
		/* 对于每一个在半径范围内的邻域点执行如下操作 */
		// 计算由关键点指向邻域点的向量
		keyPointToNeib << cloud_in_->points[index].x - cloud_in_->points[keyPointIndex].x,
			cloud_in_->points[index].y - cloud_in_->points[keyPointIndex].y,
			cloud_in_->points[index].z - cloud_in_->points[keyPointIndex].z;

		dis = sqrt(pointRadiusSquaredDistance[temp]);
		if (dis > 0.5 * radius) { flag_t = true; }

		// 第一个权重系数w1=(radius-d)^2  q点到p点距离相关的权重，距离越远，d越大，权重系数w1越小
		w1 = pow(radius - dis, 2);

		// 第二个权重系数w2=(pq·z)^2      q点到L平面投影距离相关的权重，投影距离越大，权重系数w2越大
		w2 = pow(keyPointToNeib.dot(zaxis_vector), 2);   // 向量的点乘是一个向量在另一个向量方向上的投影长度


		// 第三个权重系数w3
		// if (sqrt(keyPointToNeib[0] * keyPointToNeib[0] + keyPointToNeib[1] * keyPointToNeib[1] + keyPointToNeib[2] * keyPointToNeib[2]) > 0.5 * radius)
		w3 = 1;
		if (flag_t)
		{
			w3 = 0;
			if (kdtree.radiusSearch(cloud_in_->points[index], 0.3 * radius, w3RadiusSearch, w3RadiusSquaredDistance) > 0) {
				w3 = 1.0 / pow(1.0 * w3RadiusSearch.size() / pointidxRadiusSearch.size(), 0.5);
				if (w3 < float(1.0 / 5)) { w3 = 0; }
			}
		}

		// 计算关键点p到邻域点q的向量在L平面的投影向量
		keyPointToNeibProject = keyPointToNeib - (keyPointToNeib.dot(zaxis_vector)) * zaxis_vector;  // 关键点到邻域点在L平面的投影向量
		xaxis_vector += keyPointToNeibProject * w1 * w2 * w3;

		temp++;
	}

	// 将X坐标轴向量归一化
	xaxis_vector = xaxis_vector /
		sqrt(pow(xaxis_vector[0], 2) + pow(xaxis_vector[1], 2) + pow(xaxis_vector[2], 2));

	// (3)Y轴：将Z轴与X轴叉乘后，获得的就是Y轴
	yaxis_vector = zaxis_vector.cross(xaxis_vector);

	// 存储XYZ三轴信息
	XYZaxis.push_back(xaxis_vector);
	XYZaxis.push_back(yaxis_vector);
	XYZaxis.push_back(zaxis_vector);
}


// 计算WKD描述子
void WKDEstimation::computeWKD(std::vector< pcl::Histogram<dimension_wkd> >& WKD_features)
{
	std::cout << ">>>>>开始计算WKD描述子" << std::endl;
	//==========建立kd树，计算关键点的LRF==========
	//计算所有关键点的法向量(Z轴)
	float radius = searchRadius_ * resolution_;   // 支撑半径
	int N = N_WKD;

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimation;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);  // 搜索树
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);     // 存储ISS关键点法向量
	normal_estimation.setInputCloud(cloud_in_);    // 设定输入点云为体素滤波后的点云
	normal_estimation.setIndices(indices_in_);     // 设定要计算法向量的点云索引为提取的ISS关键点的索引
	normal_estimation.setSearchMethod(tree);    // 设定搜索方式为KdTree
	normal_estimation.setNumberOfThreads(8);    // 设定并行线程为8
	normal_estimation.setRadiusSearch(0.6 * radius);  // 设定法向计算方法为半径计算，支撑半径设置为1/3支撑半径(这个是可调参数，算法调优的一部分)
	normal_estimation.compute(*cloud_normals);  // 开始计算，法向量赋值给cloud_normals


	//构建kd搜索树
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_in_);

	std::vector<int> pointidxRadiusSearch;          // 存储邻近点的索引
	std::vector<float> pointRadiusSquaredDistance;  // 存储邻近点的距离

	Eigen::Vector3f keyPointToNeibTemp; // 关键点到邻域点的向量，用于局部坐标转换
	pcl::Histogram<dimension_wkd> WKDhistogram;   // 创建直方图
	float h = 1;   // 核宽度
	int index_y, index_z;   // 投影网格序号
	float KDEsum_yz;  // 核密度总和
	std::vector<float> density_YZ(N * N);   // 存放所有的网格的点的密度，共N^2个元素，全部初始化为0
	float gauss_dens_sum;
	int y_dis, z_dis;   // 核密度Y方向距离和Z方向距离
	float dis;          // 核密度总距离
	float K;            // 核函数值

	int false_point = 0;    // 记录被去除的点的个数

	int sizeOfR;
	for (size_t i = 0; i < indices_in_->indices.size(); i++)
	{
		sizeOfR = kdtree.radiusSearch(cloud_in_->points[indices_in_->indices[i]], radius, pointidxRadiusSearch, pointRadiusSquaredDistance);
		/*
		// 由于半径查询会包含查询点本身，因此要把第一个邻域点（查询点本身）删除
		// pointidxRadiusSearch.erase(pointidxRadiusSearch.begin());
		// 假如此时邻域点数量小于等于3，那么构建的局部坐标系和直方图一定不稳定，全部赋予-1，然后在匹配阶段去除 / 或者法线算出来是nan
		if (sizeOfR <= 5 || cloud_normals->points[Idx->indices[i]].normal_x != cloud_normals->points[Idx->indices[i]].normal_x) {
			continue;
		}*/

		//===============计算LRF===============
		std::vector<Eigen::Vector3f> XYZaxis;
		if (useWeight3_) {
			computeLRF3Weight(i, radius, kdtree, cloud_normals, pointidxRadiusSearch, pointRadiusSquaredDistance, XYZaxis);
		}
		else {
			computeLRF2Weight(i, radius, cloud_normals, pointidxRadiusSearch, XYZaxis);
		}

		// 将关键点附近的邻域点根据LRF重新计算坐标
		pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_cloud(new pcl::PointCloud<pcl::PointXYZ>);  // 储存LRF坐标系下的坐标
		LRF_cloud->resize(pointidxRadiusSearch.size());
		for (size_t k = 0; k < pointidxRadiusSearch.size(); k++)
		{
			keyPointToNeibTemp << cloud_in_->points[pointidxRadiusSearch[k]].x - cloud_in_->points[indices_in_->indices[i]].x,
				cloud_in_->points[pointidxRadiusSearch[k]].y - cloud_in_->points[indices_in_->indices[i]].y,
				cloud_in_->points[pointidxRadiusSearch[k]].z - cloud_in_->points[indices_in_->indices[i]].z;
			LRF_cloud->points[k].x = keyPointToNeibTemp.dot(XYZaxis[0]);
			LRF_cloud->points[k].y = keyPointToNeibTemp.dot(XYZaxis[1]);
			LRF_cloud->points[k].z = keyPointToNeibTemp.dot(XYZaxis[2]);
		}

		// ===============构建描述子===============
		// 初始化直方图
		for (auto& elem : WKDhistogram.histogram) { elem = 0; }

		/*仅计算YZ投影面核密度*/
		for (auto& elem : density_YZ) { elem = 0; }    // 网格密度都初始化为0

		// 计算每一个点投影在YZ面的网格的序号
		for (auto& Point : LRF_cloud->points)
		{
			// 计算每一个点在XY投影平面上，所在栅格的索引
			// index = (r+x) / (r/(n1/2))
			index_y = floor((radius + Point.y) * (N / 2) / radius);
			index_z = floor((radius + Point.z) * (N / 2) / radius);
			float u = 0.5;   // 权重
			float w = (1 - u) + u * (radius - Point.x * Point.x + Point.y * Point.y + Point.z * Point.z) / radius;
			// density_YZ[index_y + N * index_z] += 1;
			density_YZ[index_y + N * index_z] += w;
		}

		// 计算高斯核密度
		KDEsum_yz = 0;
		for (int k = 0; k < N * N; ++k) {      // i是要计算核密度的网格索引
			gauss_dens_sum = 0;
			for (int m = 0; m < N * N; ++m) {  // j是要叠加核密度的网格的索引
				y_dis = abs((k % N) - (m % N));
				z_dis = abs((k / N) - (m / N));
				dis = sqrt(y_dis * y_dis + z_dis * z_dis);
				if (dis > 2) { continue; }
				K = gaussian(dis / h);
				gauss_dens_sum += density_YZ[m] * K / (h * h);
			}
			WKDhistogram.histogram[k] = gauss_dens_sum;    // 计算出的核密度写入直方图
			KDEsum_yz += gauss_dens_sum;  // 计算核密度总和，方便后续归一化
		}
		for (int k = 0; k < N * N; ++k) {   // 归一化
			WKDhistogram.histogram[k] /= KDEsum_yz;
		}

		WKD_features.push_back(WKDhistogram);
	}

	std::cout << ">>>>>WKD描述子计算完成" << std::endl;
}




WKDEstimation::WKDEstimation()
{

}
WKDEstimation::~WKDEstimation()
{

}



/* ********************************class CorrespondenceRANSAC**********************************/
void CorrespondenceRANSAC::setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	source_in = in;
}
void CorrespondenceRANSAC::setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	target_in = in;
}
void CorrespondenceRANSAC::setInputCorrespondences(std::shared_ptr<pcl::Correspondences>& in)
{
	corr_in = in;
}

void CorrespondenceRANSAC::setThreshold(double thre)
{
	threshold = thre;
}
void CorrespondenceRANSAC::setIterationTimes(int it)
{
	iterationtimes = it;
}

pcl::PointIndices CorrespondenceRANSAC::getInliersIndices()
{
	return inliers_indices;
}
pcl::Correspondences CorrespondenceRANSAC::getRemainCorrespondences()
{
	pcl::Correspondences corr;
	for (int i = 0; i < inliers_indices.indices.size(); i++)
	{
		corr.push_back(corr_in->at(inliers_indices.indices[i]));
	}
	inliers_indices;
	return corr;
}
void CorrespondenceRANSAC::runRANSAC()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_tr(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < corr_in->size(); i++)
	{
		source->push_back(source_in->at(corr_in->at(i).index_query));
		target->push_back(target_in->at(corr_in->at(i).index_match));
	}
	std::srand((unsigned)time(NULL));

	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est;
	Eigen::Matrix4f T_SVD;
	int inlier_num = 0;
	int max_inlier = 0;
	for (int ite = 0; ite < iterationtimes; ite++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_sample(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_sample(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i < 4; i++)
		{
			int rd = rand() % (corr_in->size());
			source_sample->push_back(source_in->at(corr_in->at(rd).index_query));
			target_sample->push_back(target_in->at(corr_in->at(rd).index_match));
		}
		trans_est.estimateRigidTransformation(*source_sample, *target_sample, T_SVD);
		pcl::transformPointCloud(*source, *source_tr, T_SVD);
		vector<int> idx;
		for (int i = 0; i < source->size(); i++)
		{
			double dis = pcl::euclideanDistance(source_tr->at(i), target->at(i));
			if (dis < threshold)
			{
				idx.push_back(i);
			}
		}
		inlier_num = idx.size();
		if (inlier_num > max_inlier)
		{
			max_inlier = inlier_num;
			inliers_indices.indices.clear();
			inliers_indices.indices = idx;
		}
	}
}

CorrespondenceRANSAC::CorrespondenceRANSAC()
{

}

CorrespondenceRANSAC::~CorrespondenceRANSAC()
{

}