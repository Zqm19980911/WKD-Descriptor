#include "WKD.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/normal_3d_omp.h>


// ��˹�˺���
double WKDEstimation::gaussian(const double& u) {
	return 1.0 / sqrt(2.0 * M_PI) * exp(-(u * u) / 2.0);
}


/* *************************for class ThreePointsFeatureEstimation************************** */
void WKDEstimation::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& in)
{
	cloud_in_ = in;
}


// ��������
void WKDEstimation::setIndices(const pcl::PointIndicesPtr& indicesn)
{
	// ��������������б����0����ô�Ͱ�indices_in����Ϊ���������������indices_flag����Ϊtrue
	if (indicesn->indices.size() > 0)
	{
		indices_in_ = indicesn;
		indices_flag = true;
	}
	else    // ��֮���������������б��ǿգ���ô�Ͱ�indices_flag����Ϊfalse���ں�����ʹ��������Ƶ����е���м���
	{
		indices_flag = false;
	}
}


// ���÷ֱ���
void WKDEstimation::setResolution(float res)
{
	resolution_ = res;
}


// ���������뾶
void WKDEstimation::setSearchRadius(float radius)
{
	searchRadius_ = radius;
}


// �Ƿ�ʹ��Ȩ��3:����ֲ�ӵ����
void WKDEstimation::useWeight3(bool flag) {
	useWeight3_ = flag;
}


void WKDEstimation::computeLRF2Weight(int i, float radius, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, std::vector<Eigen::Vector3f>& XYZaxis)
{
	int keyPointIndex = indices_in_->indices[i];

	Eigen::Vector3f zaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f xaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f yaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f keyPointToNeib; // �ؼ��㵽����������
	Eigen::Vector3f keyPointToNeibProject; // �ؼ��㵽������������Lƽ���ͶӰ

	Eigen::Vector3f sum;   // ����Z������ԵļӺ���
	Eigen::Vector3f qp;   // ��Qָ��P�����������ڼ���Z������ԼҺ���

	float w1, w2;   // 3��Ȩ��ϵ��

	// ����ÿһ���ؼ��㣬��������
	// (1)Z�᣺������Ӽ��ķ�����(PCL��������ǵ�λ����)  
	zaxis_vector << cloud_normals->points[i].normal_x,
		cloud_normals->points[i].normal_y,
		cloud_normals->points[i].normal_z;

	// ȷ��Z�᷽������������
	sum.setZero();
	for (auto& index : pointidxRadiusSearch) {
		qp << cloud_in_->points[keyPointIndex].x - cloud_in_->points[index].x,
			cloud_in_->points[keyPointIndex].y - cloud_in_->points[index].y,
			cloud_in_->points[keyPointIndex].z - cloud_in_->points[index].z;
		sum += qp;
	}
	if (zaxis_vector.dot(sum) < 0) { zaxis_vector *= -1.0; }

	// (2)X�᣺���ؼ��㵽ÿ��������������Lƽ���ϵ�ͶӰ��Ȩ���Ӻ󣬻�õľ���X��
	xaxis_vector.setZero();   // ����һ���Ӻ͵ĳ�ʼ��
	for (auto& index : pointidxRadiusSearch)
	{
		/* ����ÿһ���ڰ뾶��Χ�ڵ������ִ�����²��� */
		// �����ɹؼ���ָ������������
		keyPointToNeib << cloud_in_->points[index].x - cloud_in_->points[keyPointIndex].x,
			cloud_in_->points[index].y - cloud_in_->points[keyPointIndex].y,
			cloud_in_->points[index].z - cloud_in_->points[keyPointIndex].z;
		//if (sqrt(keyPointToNeib[0] * keyPointToNeib[0] + keyPointToNeib[1] * keyPointToNeib[1] + keyPointToNeib[2] * keyPointToNeib[2]) < 0.85 * radius)
		//{
		//	continue;
		//}

		// ��һ��Ȩ��ϵ��w1=(radius-d)^2  q�㵽p�������ص�Ȩ�أ�����ԽԶ��dԽ��Ȩ��ϵ��w1ԽС
		w1 = pow(radius -
			sqrt(
				pow(keyPointToNeib[0], 2) + pow(keyPointToNeib[1], 2) + pow(keyPointToNeib[2], 2)
			), 2);

		// �ڶ���Ȩ��ϵ��w2=(pq��z)^2      q�㵽Lƽ��ͶӰ������ص�Ȩ�أ�ͶӰ����Խ��Ȩ��ϵ��w2Խ��
		w2 = pow(keyPointToNeib.dot(zaxis_vector), 2);   // �����ĵ����һ����������һ�����������ϵ�ͶӰ����

		// ����ؼ���p�������q��������Lƽ���ͶӰ����
		keyPointToNeibProject = keyPointToNeib - (keyPointToNeib.dot(zaxis_vector)) * zaxis_vector;  // �ؼ��㵽�������Lƽ���ͶӰ����
		xaxis_vector += keyPointToNeibProject * w1 * w2;
	}

	// ��X������������һ��
	xaxis_vector = xaxis_vector /
		sqrt(pow(xaxis_vector[0], 2) + pow(xaxis_vector[1], 2) + pow(xaxis_vector[2], 2));

	// (3)Y�᣺��Z����X���˺󣬻�õľ���Y��
	yaxis_vector = zaxis_vector.cross(xaxis_vector);

	// �洢XYZ������Ϣ
	XYZaxis.push_back(xaxis_vector);
	XYZaxis.push_back(yaxis_vector);
	XYZaxis.push_back(zaxis_vector);
}


void WKDEstimation::computeLRF3Weight(int i, float radius, const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, const std::vector<float>& pointRadiusSquaredDistance, std::vector<Eigen::Vector3f>& XYZaxis)
{
	int keyPointIndex = indices_in_->indices[i];

	Eigen::Vector3f zaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f xaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f yaxis_vector;   // �ؼ���LRF��Z������
	Eigen::Vector3f keyPointToNeib; // �ؼ��㵽����������
	Eigen::Vector3f keyPointToNeibProject; // �ؼ��㵽������������Lƽ���ͶӰ

	Eigen::Vector3f sum;   // ����Z������ԵļӺ���
	Eigen::Vector3f qp;   // ��Qָ��P�����������ڼ���Z������ԼҺ���

	float w1, w2, w3;   // 3��Ȩ��ϵ��
	std::vector<int> w3RadiusSearch;          // �洢�ڽ��������
	std::vector<float> w3RadiusSquaredDistance;  // �洢�ڽ���ľ���

	// ����ÿһ���ؼ��㣬��������
	// (1)Z�᣺������Ӽ��ķ�����(PCL��������ǵ�λ����)  
	zaxis_vector << cloud_normals->points[i].normal_x,
		cloud_normals->points[i].normal_y,
		cloud_normals->points[i].normal_z;

	// ȷ��Z�᷽������������
	sum.setZero();
	for (int j = 0; j < pointidxRadiusSearch.size(); ++j) {
		qp << cloud_in_->points[keyPointIndex].x - cloud_in_->points[pointidxRadiusSearch[j]].x,
			cloud_in_->points[keyPointIndex].y - cloud_in_->points[pointidxRadiusSearch[j]].y,
			cloud_in_->points[keyPointIndex].z - cloud_in_->points[pointidxRadiusSearch[j]].z;
		sum += qp;
	}
	if (zaxis_vector.dot(sum) >= 0) { zaxis_vector *= -1.0; }

	// (2)X�᣺���ؼ��㵽ÿ��������������Lƽ���ϵ�ͶӰ��Ȩ���Ӻ󣬻�õľ���X��
	xaxis_vector.setZero();   // ����һ���Ӻ͵ĳ�ʼ��
	int temp = 0;
	float dis;
	bool flag_t = false;
	for (auto& index : pointidxRadiusSearch)
	{
		/* ����ÿһ���ڰ뾶��Χ�ڵ������ִ�����²��� */
		// �����ɹؼ���ָ������������
		keyPointToNeib << cloud_in_->points[index].x - cloud_in_->points[keyPointIndex].x,
			cloud_in_->points[index].y - cloud_in_->points[keyPointIndex].y,
			cloud_in_->points[index].z - cloud_in_->points[keyPointIndex].z;

		dis = sqrt(pointRadiusSquaredDistance[temp]);
		if (dis > 0.5 * radius) { flag_t = true; }

		// ��һ��Ȩ��ϵ��w1=(radius-d)^2  q�㵽p�������ص�Ȩ�أ�����ԽԶ��dԽ��Ȩ��ϵ��w1ԽС
		w1 = pow(radius - dis, 2);

		// �ڶ���Ȩ��ϵ��w2=(pq��z)^2      q�㵽Lƽ��ͶӰ������ص�Ȩ�أ�ͶӰ����Խ��Ȩ��ϵ��w2Խ��
		w2 = pow(keyPointToNeib.dot(zaxis_vector), 2);   // �����ĵ����һ����������һ�����������ϵ�ͶӰ����


		// ������Ȩ��ϵ��w3
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

		// ����ؼ���p�������q��������Lƽ���ͶӰ����
		keyPointToNeibProject = keyPointToNeib - (keyPointToNeib.dot(zaxis_vector)) * zaxis_vector;  // �ؼ��㵽�������Lƽ���ͶӰ����
		xaxis_vector += keyPointToNeibProject * w1 * w2 * w3;

		temp++;
	}

	// ��X������������һ��
	xaxis_vector = xaxis_vector /
		sqrt(pow(xaxis_vector[0], 2) + pow(xaxis_vector[1], 2) + pow(xaxis_vector[2], 2));

	// (3)Y�᣺��Z����X���˺󣬻�õľ���Y��
	yaxis_vector = zaxis_vector.cross(xaxis_vector);

	// �洢XYZ������Ϣ
	XYZaxis.push_back(xaxis_vector);
	XYZaxis.push_back(yaxis_vector);
	XYZaxis.push_back(zaxis_vector);
}


// ����WKD������
void WKDEstimation::computeWKD(std::vector< pcl::Histogram<dimension_wkd> >& WKD_features)
{
	std::cout << ">>>>>��ʼ����WKD������" << std::endl;
	//==========����kd��������ؼ����LRF==========
	//�������йؼ���ķ�����(Z��)
	float radius = searchRadius_ * resolution_;   // ֧�Ű뾶
	int N = N_WKD;

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimation;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);  // ������
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);     // �洢ISS�ؼ��㷨����
	normal_estimation.setInputCloud(cloud_in_);    // �趨�������Ϊ�����˲���ĵ���
	normal_estimation.setIndices(indices_in_);     // �趨Ҫ���㷨�����ĵ�������Ϊ��ȡ��ISS�ؼ��������
	normal_estimation.setSearchMethod(tree);    // �趨������ʽΪKdTree
	normal_estimation.setNumberOfThreads(8);    // �趨�����߳�Ϊ8
	normal_estimation.setRadiusSearch(0.6 * radius);  // �趨������㷽��Ϊ�뾶���㣬֧�Ű뾶����Ϊ1/3֧�Ű뾶(����ǿɵ��������㷨���ŵ�һ����)
	normal_estimation.compute(*cloud_normals);  // ��ʼ���㣬��������ֵ��cloud_normals


	//����kd������
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_in_);

	std::vector<int> pointidxRadiusSearch;          // �洢�ڽ��������
	std::vector<float> pointRadiusSquaredDistance;  // �洢�ڽ���ľ���

	Eigen::Vector3f keyPointToNeibTemp; // �ؼ��㵽���������������ھֲ�����ת��
	pcl::Histogram<dimension_wkd> WKDhistogram;   // ����ֱ��ͼ
	float h = 1;   // �˿��
	int index_y, index_z;   // ͶӰ�������
	float KDEsum_yz;  // ���ܶ��ܺ�
	std::vector<float> density_YZ(N * N);   // ������е�����ĵ���ܶȣ���N^2��Ԫ�أ�ȫ����ʼ��Ϊ0
	float gauss_dens_sum;
	int y_dis, z_dis;   // ���ܶ�Y��������Z�������
	float dis;          // ���ܶ��ܾ���
	float K;            // �˺���ֵ

	int false_point = 0;    // ��¼��ȥ���ĵ�ĸ���

	int sizeOfR;
	for (size_t i = 0; i < indices_in_->indices.size(); i++)
	{
		sizeOfR = kdtree.radiusSearch(cloud_in_->points[indices_in_->indices[i]], radius, pointidxRadiusSearch, pointRadiusSquaredDistance);
		/*
		// ���ڰ뾶��ѯ�������ѯ�㱾�����Ҫ�ѵ�һ������㣨��ѯ�㱾��ɾ��
		// pointidxRadiusSearch.erase(pointidxRadiusSearch.begin());
		// �����ʱ���������С�ڵ���3����ô�����ľֲ�����ϵ��ֱ��ͼһ�����ȶ���ȫ������-1��Ȼ����ƥ��׶�ȥ�� / ���߷����������nan
		if (sizeOfR <= 5 || cloud_normals->points[Idx->indices[i]].normal_x != cloud_normals->points[Idx->indices[i]].normal_x) {
			continue;
		}*/

		//===============����LRF===============
		std::vector<Eigen::Vector3f> XYZaxis;
		if (useWeight3_) {
			computeLRF3Weight(i, radius, kdtree, cloud_normals, pointidxRadiusSearch, pointRadiusSquaredDistance, XYZaxis);
		}
		else {
			computeLRF2Weight(i, radius, cloud_normals, pointidxRadiusSearch, XYZaxis);
		}

		// ���ؼ��㸽������������LRF���¼�������
		pcl::PointCloud<pcl::PointXYZ>::Ptr LRF_cloud(new pcl::PointCloud<pcl::PointXYZ>);  // ����LRF����ϵ�µ�����
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

		// ===============����������===============
		// ��ʼ��ֱ��ͼ
		for (auto& elem : WKDhistogram.histogram) { elem = 0; }

		/*������YZͶӰ����ܶ�*/
		for (auto& elem : density_YZ) { elem = 0; }    // �����ܶȶ���ʼ��Ϊ0

		// ����ÿһ����ͶӰ��YZ�����������
		for (auto& Point : LRF_cloud->points)
		{
			// ����ÿһ������XYͶӰƽ���ϣ�����դ�������
			// index = (r+x) / (r/(n1/2))
			index_y = floor((radius + Point.y) * (N / 2) / radius);
			index_z = floor((radius + Point.z) * (N / 2) / radius);
			float u = 0.5;   // Ȩ��
			float w = (1 - u) + u * (radius - Point.x * Point.x + Point.y * Point.y + Point.z * Point.z) / radius;
			// density_YZ[index_y + N * index_z] += 1;
			density_YZ[index_y + N * index_z] += w;
		}

		// �����˹���ܶ�
		KDEsum_yz = 0;
		for (int k = 0; k < N * N; ++k) {      // i��Ҫ������ܶȵ���������
			gauss_dens_sum = 0;
			for (int m = 0; m < N * N; ++m) {  // j��Ҫ���Ӻ��ܶȵ����������
				y_dis = abs((k % N) - (m % N));
				z_dis = abs((k / N) - (m / N));
				dis = sqrt(y_dis * y_dis + z_dis * z_dis);
				if (dis > 2) { continue; }
				K = gaussian(dis / h);
				gauss_dens_sum += density_YZ[m] * K / (h * h);
			}
			WKDhistogram.histogram[k] = gauss_dens_sum;    // ������ĺ��ܶ�д��ֱ��ͼ
			KDEsum_yz += gauss_dens_sum;  // ������ܶ��ܺͣ����������һ��
		}
		for (int k = 0; k < N * N; ++k) {   // ��һ��
			WKDhistogram.histogram[k] /= KDEsum_yz;
		}

		WKD_features.push_back(WKDhistogram);
	}

	std::cout << ">>>>>WKD�����Ӽ������" << std::endl;
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