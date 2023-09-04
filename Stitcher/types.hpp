#pragma once

#define CERES_MSVC_USE_UNDERSCORE_PREFIXED_BESSEL_FUNCTIONS

#include <opencv2/opencv.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>

struct OverlapTransform {
	std::vector<cv::Point2f> pts1, pts2;
	cv::Rect2i rect1, rect2;
	std::vector<uchar> inliers;
	int parent, child, nInlers;
	cv::Mat relT;
	int prioritizedIdx;
};

typedef struct vert_info {
	int idx;
	std::string fname;
};
typedef struct edge_info {
	int nInliers;
};
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vert_info, edge_info> Graph;
typedef std::pair<int, int> Edge;

template <class WeightMap, class CapacityMap>
class edge_writer {
public:
	edge_writer(WeightMap w, CapacityMap c) : wm(w), cm(c) {}
	template <class Edge>
	void operator()(std::ostream& out, const Edge& e) const {
		//out << "[label=\"" << wm[e] << "\", taillabel=\"" << cm[e] << "\"]";
		out << "[label=\"" << wm[e] << "\"]";
	}
private:
	WeightMap wm;
	CapacityMap cm;
};
template <class WeightMap, class CapacityMap>
inline edge_writer<WeightMap, CapacityMap>
make_edge_writer(WeightMap w, CapacityMap c) {
	return edge_writer<WeightMap, CapacityMap>(w, c);
}

template <class IdxMap, class FNameMap>
class vrtx_writer {
public:
	vrtx_writer(IdxMap w, FNameMap c) : idxm(w), fnm(c) {}
	template <class Vertex>
	void operator()(std::ostream& out, const Vertex& v) const {
		//out << "[label=\"" << wm[e] << "\", taillabel=\"" << cm[e] << "\"]";
		out << "[label=\"" << fnm[v] << " (" << idxm[v] << ")" << "\"]";
	}
private:
	IdxMap idxm;
	FNameMap fnm;
};
template <class IdxMap, class FNameMap>
inline vrtx_writer<IdxMap, FNameMap>
make_vrtx_writer(IdxMap w, FNameMap c) {
	return vrtx_writer<IdxMap, FNameMap>(w, c);
}