#pragma once
#include <Eigen/Eigenvalues>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>

// 计算输入指定索引点云的协方差
Eigen::Matrix3d ComputeCovariance(const std::vector<Eigen::Vector3d>& points, const std::vector<int>& indices)
{
	if (indices.empty())
	{
		return Eigen::Matrix3d::Identity();
	}
	Eigen::Matrix3d covariance;
	Eigen::Matrix<double, 9, 1> cumulants;
	cumulants.setZero();
	for (const auto& idx : indices)
	{
		const Eigen::Vector3d& point = points[idx];
		cumulants(0) += point(0);
		cumulants(1) += point(1);
		cumulants(2) += point(2);
		cumulants(3) += point(0) * point(0);
		cumulants(4) += point(0) * point(1);
		cumulants(5) += point(0) * point(2);
		cumulants(6) += point(1) * point(1);
		cumulants(7) += point(1) * point(2);
		cumulants(8) += point(2) * point(2);
	}
	cumulants /= (double)indices.size();
	covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
	covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
	covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
	covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
	covariance(1, 0) = covariance(0, 1);
	covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
	covariance(2, 0) = covariance(0, 2);
	covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
	covariance(2, 1) = covariance(1, 2);
	return covariance;
}
// 计算每一个点的协方差
std::vector<Eigen::Matrix3d> EstimatePerPointCovariances(pcl::PointCloud<pcl::PointXYZ>::Ptr& m_cloud, float m_radius, int nnk)
{
	std::vector<Eigen::Vector3d>points;
	points.resize(m_cloud->size());
#pragma omp parallel for schedule(static)
	for (int i = 0; i < m_cloud->size(); ++i)
	{
		points[i][0] = m_cloud->points[i].x;
		points[i][1] = m_cloud->points[i].y;
		points[i][2] = m_cloud->points[i].z;
	}
	std::vector<Eigen::Matrix3d> covariances;
	covariances.resize(points.size());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree->setInputCloud(m_cloud);

#pragma omp parallel for schedule(static)
	for (int i = 0; i < (int)points.size(); i++)
	{

		std::vector<int> nn_indices;	// kdtree搜索到的点的索引
		std::vector<float> nn_dists;	// kdtree搜索到的点的距离

		if (kdtree->radiusSearch(m_cloud->points[i], m_radius, nn_indices, nn_dists, nnk) >= 3)
		{
			auto covariance = ComputeCovariance(points, nn_indices);
			covariances[i] = covariance;
		}
		else
		{
			covariances[i] = Eigen::Matrix3d::Identity();
		}
	}
	return covariances;
}
// 计算第一个特征向量
Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d& A, double eval0) 
{
	Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
	Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
	Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
	Eigen::Vector3d r0xr1 = row0.cross(row1);
	Eigen::Vector3d r0xr2 = row0.cross(row2);
	Eigen::Vector3d r1xr2 = row1.cross(row2);
	double d0 = r0xr1.dot(r0xr1);
	double d1 = r0xr2.dot(r0xr2);
	double d2 = r1xr2.dot(r1xr2);

	double dmax = d0;
	int imax = 0;
	if (d1 > dmax) 
	{
		dmax = d1;
		imax = 1;
	}
	if (d2 > dmax) 
	{
		imax = 2;
	}

	if (imax == 0) 
	{
		return r0xr1 / std::sqrt(d0);
	}
	else if (imax == 1) 
	{
		return r0xr2 / std::sqrt(d1);
	}
	else 
	{
		return r1xr2 / std::sqrt(d2);
	}
}
// 计算第二个特征向量
Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d& A, const Eigen::Vector3d& evec0, double eval1)
{
	Eigen::Vector3d U, V;
	if (std::abs(evec0(0)) > std::abs(evec0(1))) 
	{
		double inv_length = 1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
		U << -evec0(2) * inv_length, 0, evec0(0)* inv_length;
	}
	else 
	{
		double inv_length = 1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
		U << 0, evec0(2)* inv_length, -evec0(1) * inv_length;
	}
	V = evec0.cross(U);

	Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
		A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
		A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

	Eigen::Vector3d AV = { A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
						  A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
						  A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2) };

	double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
	double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
	double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

	double absM00 = std::abs(m00);
	double absM01 = std::abs(m01);
	double absM11 = std::abs(m11);
	double max_abs_comp;
	if (absM00 >= absM11) 
	{
		max_abs_comp = std::max(absM00, absM01);
		if (max_abs_comp > 0) 
		{
			if (absM00 >= absM01) 
			{
				m01 /= m00;
				m00 = 1 / std::sqrt(1 + m01 * m01);
				m01 *= m00;
			}
			else 
			{
				m00 /= m01;
				m01 = 1 / std::sqrt(1 + m00 * m00);
				m00 *= m01;
			}
			return m01 * U - m00 * V;
		}
		else 
		{
			return U;
		}
	}
	else 
	{
		max_abs_comp = std::max(absM11, absM01);
		if (max_abs_comp > 0) 
		{
			if (absM11 >= absM01) 
			{
				m01 /= m11;
				m11 = 1 / std::sqrt(1 + m01 * m01);
				m01 *= m11;
			}
			else
			{
				m11 /= m01;
				m01 = 1 / std::sqrt(1 + m11 * m11);
				m11 *= m01;
			}
			return m11 * U - m01 * V;
		}
		else 
		{
			return U;
		}
	}
}
// 快速计算特征向量
Eigen::Vector3d FastEigen3x3(const Eigen::Matrix3d& covariance) 
{
	// Previous version based on:
	// https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
	// Current version based on
	// https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
	// which handles edge cases like points on a plane

	Eigen::Matrix3d A = covariance;
	double max_coeff = A.maxCoeff();     // maxCoeff返回矩阵或向量中的最大值
	if (max_coeff == 0) 
	{
		return Eigen::Vector3d::Zero();  // 如果协方差矩阵最大是0，那就返回零矩阵作为特征向量矩阵
	}
	A /= max_coeff;

	double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);  // 左上角每个元素平方相加
	if (norm > 0) 
	{
		Eigen::Vector3d eval;
		Eigen::Vector3d evec0;
		Eigen::Vector3d evec1;
		Eigen::Vector3d evec2;

		double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;    // q是对角线元素的平均值

		double b00 = A(0, 0) - q;      // bxx是对角线元素和对角平均值的差
		double b11 = A(1, 1) - q;
		double b22 = A(2, 2) - q;

		double p = std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);  

		double c00 = b11 * b22 - A(1, 2) * A(1, 2);
		double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
		double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
		double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

		double half_det = det * 0.5;
		half_det = std::min(std::max(half_det, -1.0), 1.0);

		double angle = std::acos(half_det) / (double)3;
		double const two_thirds_pi = 2.09439510239319549;
		double beta2 = std::cos(angle) * 2;
		double beta0 = std::cos(angle + two_thirds_pi) * 2;
		double beta1 = -(beta0 + beta2);

		// 特征值
		eval(0) = q + p * beta0;   
		eval(1) = q + p * beta1;
		eval(2) = q + p * beta2;

		// 特征向量
		if (half_det >= 0) 
		{
			evec2 = ComputeEigenvector0(A, eval(2));
			if (eval(2) < eval(0) && eval(2) < eval(1)) 
			{
				A *= max_coeff;
				return evec2;
			}
			evec1 = ComputeEigenvector1(A, evec2, eval(1));
			A *= max_coeff;
			if (eval(1) < eval(0) && eval(1) < eval(2))
			{
				return evec1;
			}
			evec0 = evec1.cross(evec2);
			return evec0;
		}
		else
		{
			evec0 = ComputeEigenvector0(A, eval(0));
			if (eval(0) < eval(1) && eval(0) < eval(2)) 
			{
				A *= max_coeff;
				return evec0;
			}
			evec1 = ComputeEigenvector1(A, evec0, eval(1));
			A *= max_coeff;
			if (eval(1) < eval(0) && eval(1) < eval(2)) 
			{
				return evec1;
			}
			evec2 = evec0.cross(evec1);
			return evec2;
		}
	}
	else 
	{
		A *= max_coeff;
		if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) 
		{
			return Eigen::Vector3d(1, 0, 0);
		}
		else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) 
		{
			return Eigen::Vector3d(0, 1, 0);
		}
		else 
		{
			return Eigen::Vector3d(0, 0, 1);
		}
	}
}
// 快速计算特征向量
Eigen::Vector3d ComputeNormal(const Eigen::Matrix3d& covariance, bool fast_normal_computation)
{
	// 如果选择快速计算，即fast_normal_computation=true，则使用FastEigen3x3函数来快速计算最小特征向量（法向量）
	if (fast_normal_computation) 
	{
		return FastEigen3x3(covariance);
	}
	// 否则使用Eigen库提供的普通分解法计算最小特征向量
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
	solver.compute(covariance, Eigen::ComputeEigenvectors);
	return solver.eigenvectors().col(0);
}

// 计算每个点的法向量
pcl::PointCloud<pcl::Normal>::Ptr EstimateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr& m_cloud, float m_radius, int nnk)
{
	std::vector<Eigen::Matrix3d> covariances;
	covariances = EstimatePerPointCovariances(m_cloud, m_radius, nnk);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	for (int i = 0; i < (int)covariances.size(); ++i)
	{
		auto normal = ComputeNormal(covariances[i], true);

		pcl::Normal pn;
		pn.normal_x = normal[0];
		pn.normal_y = normal[1];
		pn.normal_z = normal[2];
		pn.curvature = 0.0;

		normals->push_back(pn);
	}

	return normals;
}
