#include "opt.hpp"

cv::Mat composeAffine(const Eigen::Matrix<double, 6, 1>& vals) {
	cv::Mat base = cv::Mat::eye(3, 3, CV_64FC1);
	double x = vals[0]; double y = vals[1]; double r = vals[2];
	double sx = vals[3]; double sy = vals[4]; double m = vals[5];
	base.at<double>(0, 0) = cos(r); base.at<double>(0, 1) = -sin(r);
	base.at<double>(1, 0) = sin(r); base.at<double>(1, 1) = cos(r);
	base.at<double>(0, 2) = x; base.at<double>(1, 2) = y;
	cv::Mat shr = cv::Mat::eye(3, 3, CV_64FC1);
	shr.at<double>(0, 1) = m;
	cv::Mat scl = cv::Mat::eye(3, 3, CV_64FC1);
	scl.at<double>(0, 0) = sx;
	scl.at<double>(1, 1) = sy;
	cv::Mat T = base * shr * scl;
	return T;
}

Eigen::Matrix<double, 6, 1> decomposeAffine(cv::Mat t) {
	double x = t.at<double>(0, 2); double y = t.at<double>(1, 2);
	double a11 = t.at<double>(0, 0); double a21 = t.at<double>(1, 0);
	double a12 = t.at<double>(0, 1); double a22 = t.at<double>(1, 1);
	double r = atan2(a21, a11);
	double sx = sqrt(a11 * a11 + a21 * a21);
	double msy = a12 * cos(r) + a22 * sin(r);
	double sy = (sin(r) != (double)0) ? ((msy * cos(r) - a12) / sin(r)) : ((a22 - msy * sin(r)) / cos(r));
	double m = msy / sy;
	Eigen::Matrix<double, 6, 1> res;
	res[0] = x; res[1] = y; res[2] = r; res[3] = sx; res[4] = sy; res[5] = m;
	return res;
}

void optimize1(const std::map<int, std::map<int, OverlapTransform>>& ots, const std::vector<cv::Mat>& Tin, std::vector<cv::Mat>& Tout, int until) {

	if (until == -1) until = Tin.size();

	ceres::Problem p;

	std::vector<Eigen::Matrix<double, 6, 1>> trs(Tin.size(), Eigen::Matrix<double, 6, 1>::Zero());

	bool fixedOne = false;
	for (int i = 0; i < Tin.size(); ++i) {
		const auto T = Tin[i];
		if (!T.empty()) {
			trs[i] = decomposeAffine(T);
			p.AddParameterBlock(trs[i].data(), 6);
			if (!fixedOne) {
				p.SetParameterBlockConstant(trs[i].data());
				fixedOne = true;
			}
		}
	}

	for (int i = 0; i < until; ++i) {
		for (int j = i + 1; j < until; ++j) {
			if (ots.count(i) && ots.at(i).count(j) && !Tin[i].empty() && !Tin[j].empty()) {
				Eigen::Matrix<double, 6, 1> rel = decomposeAffine(ots.at(i).at(j).relT);
				p.AddResidualBlock(AffineCost::Create(rel), new ceres::HuberLoss(10.0), trs[i].data(), trs[j].data());
			}
		}
	}

	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	
	ceres::Solve(options, &p, &summary);
	std::cout << summary.FullReport();

	for (int i = 0; i < Tin.size(); ++i) {
		if (!Tin[i].empty() && i < until) {
			Tout[i] = composeAffine(trs[i]);
		}
	}
}

void optimize2(const std::map<int, std::map<int, OverlapTransform>>& ots, const std::vector<cv::Mat>& Tin, std::vector<cv::Mat>& Tout, int until) {

	if (until == -1) until = Tin.size();

	ceres::Problem p;

	std::vector<Eigen::Matrix<double, 6, 1>> trs(Tin.size(), Eigen::Matrix<double, 6, 1>::Zero());

	bool fixedOne = false;
	for (int i = 0; i < Tin.size(); ++i) {
		const auto T = Tin[i];
		if (!T.empty()) {
			trs[i] = decomposeAffine(T);
			p.AddParameterBlock(trs[i].data(), 6);
			if (!fixedOne) {
				p.SetParameterBlockConstant(trs[i].data());
				fixedOne = true;
			}
		}
	}

	for (int i = 0; i < until; ++i) {
		for (int j = i + 1; j < until; ++j) {
			if (ots.count(i) && ots.at(i).count(j) && !Tin[i].empty() && !Tin[j].empty()) {
				const auto& ot = ots.at(i).at(j);
				auto pts1 = ot.pts1;
				auto pts2 = ot.pts2;
				bool prioritized = ot.prioritized;
				for (int k = 0; k < pts1.size(); ++k) {
					if (ots.at(i).at(j).inliers[k]) {
						//std::cout << i << " " << j << std::endl;
						Eigen::Vector2d pt1; pt1(0) = pts1[k].x; pt1(1) = pts1[k].y;
						Eigen::Vector2d pt2; pt2(0) = pts2[k].x; pt2(1) = pts2[k].y;
						p.AddResidualBlock(PointCost::Create(pt1, pt2, (prioritized ? 100.0f : 1.0f)), new ceres::HuberLoss(10.0), trs[i].data(), trs[j].data());
					}
				}
			}
		}
	}

	ceres::Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;

	ceres::Solve(options, &p, &summary);
	std::cout << summary.FullReport();

	for (int i = 0; i < Tin.size(); ++i) {
		if (!Tin[i].empty() && i < until) {
			Tout[i] = composeAffine(trs[i]);
		}
	}
}