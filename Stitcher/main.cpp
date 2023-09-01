#define NOMINMAX

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "loftr.h"

#include "types.hpp"
#include "opt.hpp"

bool getOverlapRectangle(const cv::Rect2i& rect1, const cv::Rect2i& rect2, cv::Rect2i& overlap) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    if (x2 <= x1 || y2 <= y1)
        return false;
    overlap = cv::Rect2i(x1, y1, x2 - x1, y2 - y1);
    return true;
}

std::vector<std::string> splitString(const std::string& input, const std::string& delimiters) {
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token)) {
        size_t startPos = 0;
        while (true) {
            size_t foundPos = token.find_first_of(delimiters, startPos);
            if (foundPos == std::string::npos) {
                tokens.push_back(token.substr(startPos));
                break;
            }
            tokens.push_back(token.substr(startPos, foundPos - startPos));
            startPos = foundPos + 1;
        }
    }

    return tokens;
}

std::pair<std::vector<std::pair<cv::Mat, cv::Rect2i>>, std::vector<std::string>> getImagesRects(std::string path) {
    std::vector<std::pair<cv::Mat, cv::Rect2i>> res;
    std::vector<std::string> fnames;
    std::filesystem::path dir(path);

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "Invalid directory path: " << path << std::endl;
        return { res, fnames };
    }

    int minX = INT_MAX;
    int minY = INT_MAX;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
            if (path.find("test5") != std::string::npos)
                cv::resize(img, img, {img.cols / 4, img.rows / 4});
            auto split = splitString(filePath, "\\/._");
            int x = std::stoi(split[split.size() - 3]);
            int y = std::stoi(split[split.size() - 2]);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            cv::Rect2i rect = { x, y, img.cols, img.rows };
            if (path.find("test5") != std::string::npos) {
                rect.x /= 4; rect.y /= 4;
            }

            if (!img.empty()) {
                res.push_back({ img, rect });
                auto splt = splitString(filePath, "\\/");
                fnames.push_back(splt[splt.size() - 1]);
            }
        }
    }

    for (auto& r : res) {
        r.second = { r.second.x - minX, r.second.y - minY, r.second.width, r.second.height };
    }

    return { res, fnames };
}

cv::Point2f warpPoint(const cv::Point2f& point, const cv::Mat& transform)
{
    cv::Mat pointMat(3, 1, CV_64FC1);
    pointMat.at<double>(0, 0) = point.x;
    pointMat.at<double>(1, 0) = point.y;
    pointMat.at<double>(2, 0) = 1.0;
    cv::Mat warpedPoint = transform * pointMat;
    float warpedX = warpedPoint.at<double>(0, 0);
    float warpedY = warpedPoint.at<double>(1, 0);
    return cv::Point2f(warpedX, warpedY);
}

cv::Mat drawMatches(cv::Mat img1, cv::Mat img2, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Rect2i& mskrect1, const cv::Rect2i& mskrect2, const std::vector<uchar>& inliers) {
    std::vector<cv::Mat> imgs = { img1, img2 };
    cv::Mat img; cv::vconcat(imgs, img);
    cv::Point2f offset1 = { float(-mskrect1.x), float(-mskrect1.y) };
    cv::Point2f offset2 = { float(-mskrect2.x), float(img1.rows - mskrect2.y) };
    for (int i = 0; i < pts1.size(); ++i)
        cv::line(img, pts1[i] + offset1, pts2[i] + offset2, inliers[i] ? CV_RGB(255, 255, 255) : CV_RGB(255, 0, 0));
    return img;
}

cv::Mat composeImages(const std::vector<std::pair<cv::Mat, cv::Rect2i>>& images, const std::vector<cv::Mat>& transforms, bool vis = false)
{
    // Find the maximum width and height among all images
    int maxWidth = 0;
    int maxHeight = 0;
    float maxX = -INFINITY;
    float maxY = -INFINITY;
    float minX = INFINITY;
    float minY = INFINITY;
    for (int i = 0; i < images.size(); ++i) {
        if (transforms[i].empty()) continue;
        auto t = transforms[i];
        auto p1 = warpPoint(cv::Point2f(images[i].second.x, images[i].second.y), t);
        auto p2 = warpPoint(cv::Point2f(images[i].second.x + images[i].second.width, images[i].second.y), t);
        auto p3 = warpPoint(cv::Point2f(images[i].second.x, images[i].second.y + images[i].second.height), t);
        auto p4 = warpPoint(cv::Point2f(images[i].second.x + images[i].second.width, images[i].second.y + images[i].second.height), t);
        maxX = std::max(maxX, std::max(p1.x, std::max(p2.x, std::max(p3.x, p4.x))));
        maxY = std::max(maxY, std::max(p1.y, std::max(p2.y, std::max(p3.y, p4.y))));
        minX = std::min(minX, std::min(p1.x, std::min(p2.x, std::min(p3.x, p4.x))));
        minY = std::min(minY, std::min(p1.y, std::min(p2.y, std::min(p3.y, p4.y))));
    }
    int w = ceil(maxX - minX);
    int h = ceil(maxY - minY);

    cv::Mat maskCanvas(h, w, CV_8UC1, cv::Scalar(0));
    for (size_t i = 0; i < images.size(); ++i) {
        if (transforms[i].empty()) continue;
        cv::Mat t = transforms[i]({ 0, 0, 3, 2 }).clone();
        t.at<double>(0, 2) -= minX;
        t.at<double>(1, 2) -= minY;

        cv::Mat layer; cv::cvtColor(images[i].first, layer, cv::COLOR_BGR2GRAY);
        layer *= 1000000;
        layer /= 255;
        cv::Mat warpedLayer;
        cv::warpAffine(layer, warpedLayer, t, maskCanvas.size());
        cv::add(warpedLayer, maskCanvas, maskCanvas);
    }

    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 0; i < images.size(); ++i) {
        if (transforms[i].empty()) continue;
        cv::Mat warpedImage;
        cv::Mat t = transforms[i]({ 0, 0, 3, 2 }).clone();
        t.at<double>(0, 2) -= minX;
        t.at<double>(1, 2) -= minY;
        cv::warpAffine(images[i].first, warpedImage, t, canvas.size());
        for (int x = 0; x < warpedImage.cols; ++x) {
            for (int y = 0; y < warpedImage.rows; ++y) {
                auto rgb = warpedImage.at<cv::Vec3b>(y, x);
                if (rgb[0] > 0 || rgb[1] > 0 || rgb[2] > 0) {
                    float coeff = 1.0f / float(maskCanvas.at<uchar>(y, x));
                    uchar r = floor(float(rgb[0]) * coeff);
                    uchar g = floor(float(rgb[1]) * coeff);
                    uchar b = floor(float(rgb[2]) * coeff);
                    canvas.at<cv::Vec3b>(y, x) += cv::Vec3b(r, g, b);
                }
            }
        }
        if (vis) {
            cv::imshow("c", canvas);
            cv::waitKey();
        }
    }

    return canvas;
}

inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main() {

    LOFTR loftr;

    std::string dir = "test5";
    auto imginfos = getImagesRects(dir);
    auto imrecs = imginfos.first;
    auto fnames = imginfos.second;

    Graph g;
    std::map<int, std::map<int, OverlapTransform>> ots;
    std::vector<cv::Mat> absTs0(imrecs.size(), cv::Mat());
    std::vector<cv::Mat> absTs(imrecs.size(), cv::Mat());

    for (int i = 0; i < imrecs.size(); ++i) {
        boost::add_vertex({ i, fnames[i] }, g);
    }

    for (int i = 0; i < imrecs.size(); ++i) {
        auto rect1 = imrecs[i].second;
        
        Eigen::Matrix<double, 6, 1> t0;
        t0[0] = rect1.x; t0[1] = rect1.y; t0[2] = 0; 
        t0[3] = 1.0; t0[4] = 1.0; t0[5] = 0;
        absTs0[i] = composeAffine(t0);
        
        auto img1 = imrecs[i].first;
        for (int j = i + 1; j < imrecs.size(); ++j) {
            auto img2 = imrecs[j].first;
            auto rect2 = imrecs[j].second;
            
            cv::Rect2i overlap;
            bool overlaps = getOverlapRectangle(rect1, rect2, overlap);
            if (overlaps) {
                auto mskrect1 = cv::Rect2i(overlap.x - rect1.x, overlap.y - rect1.y, overlap.width, overlap.height);
                cv::Mat msk1 = cv::Mat(img1.rows, img1.cols, CV_8UC1, cv::Scalar(0));
                cv::rectangle(msk1, mskrect1, cv::Scalar(255), -1);
                auto mskrect2 = cv::Rect2i(overlap.x - rect2.x, overlap.y - rect2.y, overlap.width, overlap.height);
                cv::Mat msk2 = cv::Mat(img2.rows, img2.cols, CV_8UC1, cv::Scalar(0));
                cv::rectangle(msk2, mskrect2, cv::Scalar(255), -1);
                
                std::vector<cv::Point2f> kpts1, kpts2;

                if (!std::filesystem::is_directory("cache") || !std::filesystem::exists("cache")) { // Check if src folder exists
                    std::filesystem::create_directory("cache");
                }
                std::string fname = "cache/" + dir + "_" + std::to_string(i) + "_" + std::to_string(j) + ".loftr";
                if (file_exists(fname)) {
                    std::ifstream in(fname);
                    int npts; in >> npts;
                    for (int i = 0; i < 2 * npts; ++i) {
                        cv::Point2f pt; in >> pt.x >> pt.y;
                        ((i >= npts) ? kpts2 : kpts1).push_back(pt);
                    }
                } else {
                    loftr.match(img1(mskrect1), img2(mskrect2), cv::Mat(), cv::Mat(), kpts1, kpts2);
                    std::ofstream out(fname);
                    out << kpts1.size() << std::endl;
                    for (auto& kpt : kpts1) {
                        kpt += cv::Point2f(overlap.x - rect1.x, overlap.y - rect1.y);
                        out << kpt.x << " " << kpt.y << " ";
                    }
                    for (auto& kpt : kpts2) {
                        kpt += cv::Point2f(overlap.x - rect2.x, overlap.y - rect2.y);
                        out << kpt.x << " " << kpt.y << " ";
                    }

                }

                if (kpts1.size()) {

                    std::vector<uchar> inliers;
                    cv::Mat T = cv::estimateAffinePartial2D(kpts2, kpts1, inliers, 8, 50.0);
                    //cv::Mat T = cv::estimateAffine2D(kpts2, kpts1, inliers);
                    int ninlers = std::count(inliers.begin(), inliers.end(), 1);
                    std::cout << i << " " << j << " " << ninlers << std::endl;

                    //if (i == 5 || j == 5) {
                    //    cv::Mat out = drawMatches(img1(mskrect1), img2(mskrect2), kpts1, kpts2, mskrect1, mskrect2, inliers);
                    //    cv::imshow("m", out);
                    //    cv::waitKey();
                    //}

                    int minarea = std::min(rect1.area(), rect2.area());
                    int maxarea = std::max(rect1.area(), rect2.area());
                    bool prioritized = ((float)minarea / (float)maxarea) < 0.5f;

                    if (ninlers >= 240 || prioritized) {
                        cv::Mat base = cv::Mat::eye(3, 3, CV_64F);
                        T.copyTo(base({ 0, 0, 3, 2 }));
                        ots[i][j] = { kpts1, kpts2, inliers, i, j, ninlers, base, prioritized };
                        ots[j][i] = { kpts2, kpts1, inliers, j, i, ninlers, base.inv(), prioritized };
                    }
                    boost::add_edge(i, j, { ninlers }, g);
                }
            }
        }
        if (i == 0) {
            absTs[i] = cv::Mat::eye(3, 3, CV_64F);
            absTs[i].at<double>(0, 2) = rect1.x;
            absTs[i].at<double>(1, 2) = rect1.y;
        }
    }

    std::ofstream dot("graph.dot");
    boost::write_graphviz(dot, g,
        make_vrtx_writer(boost::get(&vert_info::idx, g), boost::get(&vert_info::fname, g)),
        make_edge_writer(boost::get(&edge_info::nInliers, g), boost::get(&edge_info::nInliers, g)));

    bool allAbsTfound = false;
    bool added = true;
    while (!allAbsTfound && added)
    {
        std::vector<int> maxIForJ(absTs.size(), -1);
        std::vector<int> maxIForJ_idx(absTs.size(), -1);
        std::vector<int> scndMaxIForJ(absTs.size(), -1);
        std::vector<int> scndMaxIForJ_idx(absTs.size(), -1);
        int bestJ = -1, bestJ_idx1 = -1, bestJ_idx2 = -1;
        for (int i = 0; i < absTs.size(); ++i) {
            for (int j = 0; j < absTs.size(); ++j) {
                if (absTs[j].empty() && ots.count(i) && ots[i].count(j)) {
                    if (ots[i][j].nInlers > maxIForJ[j]) {
                        maxIForJ[j] = ots[i][j].nInlers;
                        maxIForJ_idx[j] = i;
                    }
                    if (!absTs[i].empty()) {
                        if (ots[i][j].nInlers > scndMaxIForJ[j]) {
                            scndMaxIForJ[j] = ots[i][j].nInlers;
                            scndMaxIForJ_idx[j] = i;
                        }
                        if (ots[i][j].nInlers > bestJ) {
                            bestJ = ots[i][j].nInlers;
                            bestJ_idx1 = i;
                            bestJ_idx2 = j;
                        }
                    }
                }
            }
        }

        added = false;
        allAbsTfound = true;
        for (int i = 0; i < absTs.size(); ++i) {
            if (maxIForJ[i] > -1) {
                if (!absTs[maxIForJ_idx[i]].empty()) {
                    absTs[i] = absTs[maxIForJ_idx[i]] * ots[maxIForJ_idx[i]][i].relT;
                    added = true;
                }           
            }
            if (absTs[i].empty()) {
                allAbsTfound = false;
            }
        }

        if (!added && bestJ > -1) {
            absTs[bestJ_idx2] = absTs[bestJ_idx1] * ots[bestJ_idx1][bestJ_idx2].relT;
            added = true;
        }
    }

    auto composed = composeImages(imrecs, absTs0);
    cv::imwrite("res0.png", composed);

    composed = composeImages(imrecs, absTs);
    cv::imwrite("res1.png", composed);

    std::vector<cv::Mat> absTs2(imrecs.size(), cv::Mat());
    optimize1(ots, absTs, absTs2);
    composed = composeImages(imrecs, absTs2);
    cv::imwrite("res2.png", composed);

    std::vector<cv::Mat> absTs3(imrecs.size(), cv::Mat());
    optimize2(ots, absTs, absTs3);
    composed = composeImages(imrecs, absTs3, true);
    cv::imwrite("res3.png", composed);

	return 0;
}