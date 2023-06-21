#pragma once
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/point_representation.h>
#include <pcl/PointIndices.h>
#include <vector>


using namespace std;

const int N_WKD = 12;
const int dimension_wkd = N_WKD * N_WKD;

// 自定义描述符类
struct WKDSigniture {
	float histogram[dimension_wkd] = { 0.f };
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
	WKDSigniture,
	(float[dimension_wkd], histogram, histogram)
)

// 添加点表达
namespace pcl {
	template <>
	class DefaultPointRepresentation<WKDSigniture> : public PointRepresentation <WKDSigniture> {
	public:
		DefaultPointRepresentation() {
			nr_dimensions_ = dimension_wkd; // 特征维度
		}

		// 将MyDescriptor类型的特征向量转换为Eigen向量
		virtual void
			copyToFloatArray(const WKDSigniture& p, float* out) const {
			for (int i = 0; i < nr_dimensions_; ++i)
				out[i] = p.histogram[i];
		}
	};
} // namespace pcl


class WKDEstimation
{
public:
	double gaussian(const double& u);
	void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in);
	void setIndices(const pcl::PointIndicesPtr& indices);
	void setSearchRadius(float radius);
	void setResolution(float res);
	void useWeight3(bool flag);

	void computeWKD(std::vector< pcl::Histogram<dimension_wkd> >& WKD_features);
	
	WKDEstimation();
	~WKDEstimation();
private:
	void computeLRF2Weight(int i, float radius, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, std::vector<Eigen::Vector3f>& XYZaxis);
	void computeLRF3Weight(int i, float radius, const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree, const pcl::PointCloud<pcl::Normal>::Ptr cloud_normals, const std::vector<int>& pointidxRadiusSearch, const std::vector<float>& pointRadiusSquaredDistance, std::vector<Eigen::Vector3f>& XYZaxis);

//	void LRFEstimator(const int& current_point_idx, Eigen::Matrix3f& rf, std::vector<int>& n_indices, std::vector<float>& n_sqr_distances);
//	void init();
//	cv::Mat imagedft(cv::Mat& input);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_;
	pcl::PointIndicesPtr indices_in_;
	float searchRadius_;
	float resolution_;
	bool indices_flag = false;
	bool useWeight3_ = false;
};



class CorrespondenceRANSAC
{
public:
	void setInputSource(pcl::PointCloud<pcl::PointXYZ>::Ptr& source);
	void setInputTarget(pcl::PointCloud<pcl::PointXYZ>::Ptr& target);
	void setInputCorrespondences(std::shared_ptr<pcl::Correspondences>& corr);
	void setIterationTimes(int iter);
	void setThreshold(double thre);
	void runRANSAC();
	pcl::PointIndices getInliersIndices();
	pcl::Correspondences getRemainCorrespondences();
	CorrespondenceRANSAC();
	~CorrespondenceRANSAC();
private:
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_in;
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_in;
	std::shared_ptr<pcl::Correspondences> corr_in;
	pcl::PointIndices inliers_indices;
	double threshold = 0.3;
	int iterationtimes = 500;
};