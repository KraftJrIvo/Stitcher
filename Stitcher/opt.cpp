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

void optimize2(const std::map<int, std::map<int, OverlapTransform>>& ots, const std::vector<cv::Mat>& Tin, std::vector<cv::Mat>& Tout, int until, 
	std::vector<cv::Mat>* masks) {

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

	std::map<int, bool> prioritizedVerts;
	std::map<int, cv::Rect2i> rects;
	std::map<int, std::map<int, std::vector<ceres::ResidualBlockId>>> edges;
	std::map<int, std::map<int, std::vector<int>>> edgePtIds;

	for (int i = 0; i < until; ++i) {
		for (int j = i + 1; j < until; ++j) {
			if (ots.count(i) && ots.at(i).count(j) && !Tin[i].empty() && !Tin[j].empty()) {
				const auto& ot = ots.at(i).at(j);
				auto pts1 = ot.pts1;
				auto pts2 = ot.pts2;
				bool prioritized = ot.prioritizedIdx > -1;
				for (int k = 0; k < pts1.size(); ++k) {
					if (ot.inliers[k]) {
						Eigen::Vector2d pt1; pt1(0) = pts1[k].x; pt1(1) = pts1[k].y;
						Eigen::Vector2d pt2; pt2(0) = pts2[k].x; pt2(1) = pts2[k].y;
						float weight = prioritized ? 100.0f : 1.0f;
						auto huber = new ceres::HuberLoss(10.0);		
						if (!edges.count(i) || !edges[i].count(j)) {
							edges[i][j] = {};
							edgePtIds[i][j] = {};
						}
						edges[i][j].push_back(p.AddResidualBlock(PointCost::Create(pt1, pt2, weight), huber, trs[i].data(), trs[j].data()));
						edgePtIds[i][j].push_back(k);
						if (prioritized)
							prioritizedVerts[ot.prioritizedIdx] = true;
					}
				}
				rects[i] = ot.rect1;
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

	if (masks) {
		Eigen::Vector4d residual;
		for (int i = 0; i < Tin.size(); ++i) {
			const auto T = Tin[i];
			if (!T.empty()) {
				const auto& rect = rects[i];
				if (!prioritizedVerts.count(i)) {
					cv::Mat mask = cv::Mat(rect.height, rect.width, CV_8UC1, cv::Scalar(0));
					for (int j = i + 1; j < Tin.size(); ++j) {
						if (ots.count(i) && ots.at(i).count(j) && !Tin[i].empty() && !Tin[j].empty()) {
							auto edgeIds = edges[i][j];
							std::vector<cv::Point2i> goodPoints;
							for (int k = 0; k < edgeIds.size(); ++k) {
								double cost; p.EvaluateResidualBlock(edgeIds[k], false, &cost, residual.data(), nullptr);
								cost = abs(residual[0] + residual[1]);
								if (ots.at(i).at(j).prioritizedIdx != -1)
									cost /= 100.0f;
								if (cost < 1.5) {
									auto& pts = (ots.at(i).at(j).parent == i) ? ots.at(i).at(j).pts1 : ots.at(i).at(j).pts2;
									goodPoints.push_back({ int(pts[k].x), int(pts[k].y) });
								}
							}
							std::vector<cv::Point2i> hull;
							cv::convexHull(goodPoints, hull);
							cv::fillConvexPoly(mask, hull, cv::Scalar(255));
						}
					}
					masks->push_back(mask);
				} else {
					masks->push_back(cv::Mat(rect.height, rect.width, CV_8UC1, cv::Scalar(255)));
				}
			}
		}
	}
}