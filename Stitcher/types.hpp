#pragma once

#include <opencv2/opencv.hpp>

#include <boost/graph/graphviz.hpp>
#include <boost/graph/adjacency_list.hpp>

struct OverlapTransform {
	std::vector<cv::Point2f> pts1, pts2;
	int parent, child, nInlers;
	cv::Mat relT;
};

typedef struct vert_info {
	int idx;
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