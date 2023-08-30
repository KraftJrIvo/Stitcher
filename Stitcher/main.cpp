#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "loftr.h"

#include "types.h"

bool getOverlapRectangle(const cv::Rect2i& rect1, const cv::Rect2i& rect2, cv::Rect2i& overlap) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x2 <= x1 || y2 <= y1) {
        // No overlap
        return false;
    }

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

std::vector<std::pair<cv::Mat, cv::Rect2i>> getImagesRects(std::string path) {
    std::vector<std::pair<cv::Mat, cv::Rect2i>> res;
    std::filesystem::path dir(path);

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "Invalid directory path: " << path << std::endl;
        return res;
    }

    int minX = INT_MAX;
    int minY = INT_MAX;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
            auto split = splitString(filePath, "\\/._");
            int x = std::stoi(split[split.size() - 3]);
            int y = std::stoi(split[split.size() - 2]);
            minX = std::min(minX, x);
            minY = std::min(minY, y);
            cv::Rect2i rect = { x, y, img.cols, img.rows };

            if (!img.empty()) {
                res.push_back({ img, rect });
            }
        }
    }

    for (auto& r : res) {
        r.second = { r.second.x - minX, r.second.y - minY, r.second.width, r.second.height };
    }

    return res;
}

cv::Point2f warpPoint(const cv::Point2f& point, const cv::Mat& transform)
{
    cv::Mat pointMat(3, 1, CV_64FC1);  // Homogeneous coordinate of input point
    pointMat.at<double>(0, 0) = point.x;
    pointMat.at<double>(1, 0) = point.y;
    pointMat.at<double>(2, 0) = 1.0;

    // Apply the affine transform to the point
    cv::Mat warpedPoint = transform * pointMat;

    // Retrieve the warped point coordinates
    float warpedX = warpedPoint.at<double>(0, 0);
    float warpedY = warpedPoint.at<double>(1, 0);

    return cv::Point2f(warpedX, warpedY);
}

cv::Mat composeImages(const std::vector<std::pair<cv::Mat, cv::Rect2i>>& images, const std::vector<cv::Mat>& transforms)
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

        cv::Mat layer = cv::Mat(images[i].first.rows, images[i].first.cols, CV_8UC1, cv::Scalar(1));
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
                float coeff = 1.0f / float(maskCanvas.at<uchar>(y, x));
                uchar r = floor(float(rgb[0]) * coeff);
                uchar g = floor(float(rgb[1]) * coeff);
                uchar b = floor(float(rgb[2]) * coeff);
                canvas.at<cv::Vec3b>(y, x) += cv::Vec3b(r, g, b);
            }
        }
        cv::imshow("c", canvas);
        cv::waitKey();
    }

    return canvas;
}

int main() {

    LOFTR loftr;

    //auto imrecs = getImagesRects("C:/cool/dioram/yandex/facades/butman2/walls/3");
    //auto imrecs = getImagesRects("test");
    auto imrecs = getImagesRects("test4");

    std::map<int, std::map<int, OverlapTransform>> ots;
    std::vector<cv::Mat> absTs(imrecs.size(), cv::Mat());

    for (int i = 0; i < imrecs.size(); ++i) {
        auto img1 = imrecs[i].first;
        auto rct1 = imrecs[i].second;
        for (int j = i + 1; j < imrecs.size(); ++j) {
            auto img2 = imrecs[j].first;
            auto rct2 = imrecs[j].second;
            
            cv::Rect2i overlap;
            bool overlaps = getOverlapRectangle(rct1, rct2, overlap);
            if (overlaps) {
                std::cout << i << " " << j << " | " << rct1.x << " " << rct1.y << " " << rct1.width << " " << rct1.height << " | " << rct2.x << " " << rct2.y << " " << rct2.width << " " << rct2.height << " | " << overlap.area() << std::endl;
                auto mskrect1 = cv::Rect2i(overlap.x - rct1.x, overlap.y - rct1.y, overlap.width, overlap.height);
                cv::Mat msk1 = cv::Mat(img1.rows, img1.cols, CV_8UC1, cv::Scalar(0));
                cv::rectangle(msk1, mskrect1, cv::Scalar(255), -1);
                auto mskrect2 = cv::Rect2i(overlap.x - rct2.x, overlap.y - rct2.y, overlap.width, overlap.height);
                cv::Mat msk2 = cv::Mat(img2.rows, img2.cols, CV_8UC1, cv::Scalar(0));
                cv::rectangle(msk2, mskrect2, cv::Scalar(255), -1);
                
                std::vector<cv::Point2f> kpts1, kpts2;
                loftr.match(img1, img2, msk1, msk2, kpts1, kpts2);

                if (kpts1.size()) {
                    //std::vector<cv::KeyPoint> kpts1_, kpts2_;
                    //std::vector<std::vector<cv::DMatch>> mtchs(1);
                    //for (int i = 0; i < kpts1.size(); ++i) {
                    //    kpts1_.push_back(cv::KeyPoint(kpts1[i], 2));
                    //    mtchs[0].push_back(cv::DMatch(i, i, 0.0f));
                    //}
                    //for (auto& pt : kpts2)
                    //    kpts2_.push_back(cv::KeyPoint(pt, 2));
                    //cv::Mat out; cv::drawMatches(img1, kpts1_, img2, kpts2_, mtchs, out);
                    //cv::imshow("m1", msk1);
                    //cv::imshow("m2", msk2);
                    //cv::imshow("w", out);
                    //cv::waitKey();

                    std::vector<uchar> inliers;
                    cv::Mat T = cv::estimateAffinePartial2D(kpts2, kpts1, inliers);
                    cv::Mat base = cv::Mat::eye(3, 3, CV_64F);
                    T.copyTo(base({ 0, 0, 3, 2 }));
                    int ninlers = std::count(inliers.begin(), inliers.end(), 1);

                    ots[i][j] = { kpts1, kpts2, i, j, ninlers, base };
                    ots[j][i] = { kpts2, kpts1, j, i, ninlers, base.inv() };
                }
            }
        }
        if (i == 0) {
            absTs[i] = cv::Mat::eye(3, 3, CV_64F);
            absTs[i].at<double>(0, 2) = rct1.x;
            absTs[i].at<double>(1, 2) = rct1.y;
        }
    }

    bool allAbsTfound = false;
    bool added = true;
    while (!allAbsTfound && added)
    {
        std::vector<int> maxIForJ(absTs.size(), -1);
        std::vector<int> maxIForJ_idx(absTs.size(), -1);
        std::vector<int> scndMaxIForJ(absTs.size(), -1);
        std::vector<int> scndMaxIForJ_idx(absTs.size(), -1);
        for (int i = 0; i < absTs.size(); ++i) {
            for (int j = 0; j < absTs.size(); ++j) {
                if (absTs[j].empty() && ots.count(i) && ots[i].count(j)) {
                    if (ots[i][j].nInlers > maxIForJ[j]) {
                        maxIForJ[j] = ots[i][j].nInlers;
                        maxIForJ_idx[j] = i;
                    }
                    if (!absTs[i].empty() && ots[i][j].nInlers > scndMaxIForJ[j]) {
                        scndMaxIForJ[j] = ots[i][j].nInlers;
                        scndMaxIForJ_idx[j] = i;
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
                } else if (i == maxIForJ_idx[maxIForJ_idx[i]] && scndMaxIForJ_idx[i] > -1) {
                    absTs[i] = absTs[scndMaxIForJ_idx[i]] * ots[scndMaxIForJ_idx[i]][i].relT;
                    maxIForJ_idx[i] = -1;
                    added = true;
                    allAbsTfound = false;
                    break;
                }                
            }
            if (absTs[i].empty()) {
                allAbsTfound = false;
            }
        }
    }

    auto composed = composeImages(imrecs, absTs);
    cv::imshow("w", composed);
    cv::waitKey();

	return 0;
}