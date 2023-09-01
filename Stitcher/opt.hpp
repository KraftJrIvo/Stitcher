#pragma once

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "types.hpp"

#include <ceres/ceres.h>

cv::Mat composeAffine(const Eigen::Matrix<double, 6, 1>& vals);
void optimize1(const std::map<int, std::map<int, OverlapTransform>>& ots, const std::vector<cv::Mat>& Tin, std::vector<cv::Mat>& Tout, int until = -1);
void optimize2(const std::map<int, std::map<int, OverlapTransform>>& ots, const std::vector<cv::Mat>& Tin, std::vector<cv::Mat>& Tout, int until = -1);

class AffineCost {
public:
    AffineCost(const Eigen::Matrix<double, 6, 1>& rel) : rel(rel) { }

    static ceres::CostFunction* Create(const Eigen::Matrix<double, 6, 1>& rel) {
        return (new ceres::AutoDiffCostFunction<AffineCost, 7, 6, 6>(new AffineCost(rel)));
    }

	template <typename T>
	Eigen::Matrix<T, 3, 3> composeAffine(const Eigen::Matrix<T, 6, 1>& vals) const {
		Eigen::Matrix<T, 3, 3> base = Eigen::Matrix<T, 3, 3>::Identity();
		T x = vals[0]; T y = vals[1]; T r = vals[2]; 
		T sx = vals[3]; T sy = vals[4]; T m = vals[5];
		base(0, 0) = cos(r); base(0, 1) = -sin(r);
		base(1, 0) = sin(r); base(1, 1) = cos(r);
		base(0, 2) = x; base(1, 2) = y;
		Eigen::Matrix<T, 3, 3> shr = Eigen::Matrix<T, 3, 3>::Identity();
		shr(0, 1) = m;
		Eigen::Matrix<T, 3, 3> scl = Eigen::Matrix<T, 3, 3>::Identity();
		scl(0, 0) = sx;
		scl(1, 1) = sy;
		Eigen::Matrix<T, 3, 3> T = base * shr * scl;
		return T;
	}

	template <typename T>
	Eigen::Matrix<T, 6, 1> decomposeAffine(Eigen::Matrix<T, 3, 3> t) const {
		T x = t(0, 2); T y = t(1, 2);
		T a11 = t(0, 0); T a21 = t(1, 0); 
		T a12 = t(0, 1); T a22 = t(1, 1);
		T r = atan2(a21, a11);
		T sx = sqrt(a11 * a11 + a21 * a21);
		T msy = a12 * cos(r) + a22 * sin(r);
		T sy = (sin(r) != (T)0) ? ((msy * cos(r) - a12) / sin(r)) : ((a22 - msy * sin(r)) / cos(r));
		T m = msy / sy;
		Eigen::Matrix<T, 6, 1> res;
		res[0] = x; res[1] = y; res[2] = r; res[3] = sx; res[4] = sy; res[5] = m;
		return res;
	}

    template <typename T>
    bool operator()(const T* const t1_, const T* const t2_, T* residuals) const {
        
		Eigen::Matrix<T, 6, 1> t1; t1[0] = (T)t1_[0]; t1[1] = (T)t1_[1]; t1[2] = (T)t1_[2]; t1[3] = (T)t1_[3]; t1[4] = (T)t1_[4]; t1[5] = (T)t1_[5];
		Eigen::Matrix<T, 6, 1> t2; t2[0] = (T)t2_[0]; t2[1] = (T)t2_[1]; t2[2] = (T)t2_[2]; t2[3] = (T)t2_[3]; t2[4] = (T)t2_[4]; t2[5] = (T)t2_[5];
		auto T1 = composeAffine(t1);
		auto T2 = composeAffine(t2);
		Eigen::Matrix<T, 3, 3> T12 = T1.inverse() * T2;
		auto t12 = decomposeAffine(T12);

		for (int i = 0; i < 6; ++i) {
			residuals[i] = t12[i] - (T)rel[i];
			if (i < 2) residuals[i] /= (T)1000.0;
			if (i >= 3) residuals[i] *= (T)10.0;
		}
		residuals[6] = t1[2] + t2[2];

        return true;
    }

private:
    Eigen::Matrix<double, 6, 1> rel;
};

class PointCost {
public:
	PointCost(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2) : pt1(pt1), pt2(pt2) { }

	static ceres::CostFunction* Create(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2) {
		return (new ceres::AutoDiffCostFunction<PointCost, 3, 6, 6>(new PointCost(pt1, pt2)));
	}

	template <typename T>
	Eigen::Matrix<T, 3, 3> composeAffine(const Eigen::Matrix<T, 6, 1>& vals) const {
		Eigen::Matrix<T, 3, 3> base = Eigen::Matrix<T, 3, 3>::Identity();
		T x = vals[0]; T y = vals[1]; T r = vals[2];
		T sx = vals[3]; T sy = vals[4]; T m = vals[5];
		base(0, 0) = cos(r); base(0, 1) = -sin(r);
		base(1, 0) = sin(r); base(1, 1) = cos(r);
		base(0, 2) = x; base(1, 2) = y;
		Eigen::Matrix<T, 3, 3> shr = Eigen::Matrix<T, 3, 3>::Identity();
		shr(0, 1) = m;
		Eigen::Matrix<T, 3, 3> scl = Eigen::Matrix<T, 3, 3>::Identity();
		scl(0, 0) = sx;
		scl(1, 1) = sy;
		Eigen::Matrix<T, 3, 3> T = base * shr * scl;
		return T;
	}

	template <typename T>
	Eigen::Matrix<T, 2, 1> warpPoint(const Eigen::Matrix<T, 2, 1>& point, const Eigen::Matrix<T, 3, 3>& t) const {
		Eigen::Matrix<T, 3, 1> pointMat;
		pointMat(0, 0) = point.x();
		pointMat(1, 0) = point.y();
		pointMat(2, 0) = (T)1.0;
		Eigen::Matrix<T, 3, 1> warpedPoint = t * pointMat;
		Eigen::Matrix<T, 2, 1> res;
		res(0) = warpedPoint(0) / warpedPoint(2);
		res(1) = warpedPoint(1) / warpedPoint(2);
		return res;
	}

	template <typename T>
	bool operator()(const T* const t1_, const T* const t2_, T* residuals) const {

		Eigen::Matrix<T, 6, 1> t1; t1[0] = (T)t1_[0]; t1[1] = (T)t1_[1]; t1[2] = (T)t1_[2]; t1[3] = (T)t1_[3]; t1[4] = (T)t1_[4]; t1[5] = (T)t1_[5];
		Eigen::Matrix<T, 6, 1> t2; t2[0] = (T)t2_[0]; t2[1] = (T)t2_[1]; t2[2] = (T)t2_[2]; t2[3] = (T)t2_[3]; t2[4] = (T)t2_[4]; t2[5] = (T)t2_[5];
		auto T1 = composeAffine(t1);
		auto T2 = composeAffine(t2);
		Eigen::Matrix<T, 2, 1> tpt1; tpt1[0] = (T)pt1.x(); tpt1[1] = (T)pt1.y();
		Eigen::Matrix<T, 2, 1> tpt2; tpt2[0] = (T)pt2.x(); tpt2[1] = (T)pt2.y();
		auto wpt1 = warpPoint(tpt1, T1);
		auto wpt2 = warpPoint(tpt2, T2);

		residuals[0] = wpt1.x() - wpt2.x();
		residuals[1] = wpt1.y() - wpt2.y();
		residuals[2] = (t1[2] + t2[2]) * (T)100.0;

		return true;
	}

private:
	Eigen::Vector2d pt1;
	Eigen::Vector2d pt2;
};
