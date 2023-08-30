#include "loftr.h"
#include "util.h"

#include <boost/asio.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define IPV4 "127.0.0.1"
#define PORT 8867


LOFTR::LOFTR()
{
    if (!util::port_in_use(PORT)) {
        _popen("python loftr-server.py", "r");
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
}

void LOFTR::_request_matches(cv::Mat img1, cv::Mat img2, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>& matches)
{
    try
    {
        using namespace std;
        using boost::asio::ip::tcp;
        std::vector<uchar> buf;
        std::vector<cv::Mat> imgs = { img1, img2 };
        cv::Mat img; cv::hconcat(imgs, img);
        cv::imencode(".jpg", img, buf);
        auto* enc_msg = reinterpret_cast<unsigned char*>(buf.data());
        std::string js = util::base64_encode(enc_msg, buf.size());
        boost::asio::io_service io_service;
        string ipAddress = IPV4;
        string portNum = std::to_string(PORT);
        string hostAddress = ipAddress + ":" + portNum;
        string wordToQuery = "aha";
        tcp::resolver resolver(io_service);
        tcp::resolver::query query(ipAddress, portNum);
        tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
        tcp::socket socket(io_service);
        boost::asio::connect(socket, endpoint_iterator);
        boost::asio::streambuf request;
        std::ostream request_stream(&request);
        request_stream << "POST /match" << " HTTP/1.1\r\n";
        request_stream << "Host: " << hostAddress << "\r\n";
        request_stream << "Accept: */*\r\n";
        request_stream << "User-Agent: glOpt\r\n";
        request_stream << "Content-Type: application/json; charset=utf-8 \r\n";
        request_stream << "Content-Length: " << js.length() << "\r\n";
        request_stream << "Connection: close\r\n\r\n";  //NOTE THE Double line feed
        request_stream << js;
        boost::asio::write(socket, request);
        boost::asio::streambuf response;
        boost::asio::read_until(socket, response, "\r\n");
        std::istream response_stream(&response);
        std::string http_version;
        response_stream >> http_version;
        unsigned int status_code;
        response_stream >> status_code;
        std::string status_message;
        std::getline(response_stream, status_message);
        if (!response_stream || (http_version.substr(0, 5) != "HTTP/") || status_code != 200)
            return;
        boost::asio::read_until(socket, response, "\r\n\r\n");
        std::string header;
        while (std::getline(response_stream, header) && header != "\r")
        {
        }
        boost::system::error_code error;
        while (boost::asio::read(socket, response, boost::asio::transfer_at_least(1), error))
        {
        }
        std::string resp((std::istreambuf_iterator<char>(&response)), std::istreambuf_iterator<char>());
        json j_complete = json::parse(resp);
        auto sz = j_complete.size();
        matches.first.resize(sz);
        matches.second.resize(sz);
        for (int i = 0; i < sz; ++i) {
            matches.first[i] = { j_complete[i][0], j_complete[i][1] };
            matches.second[i] = { j_complete[i][2], j_complete[i][3] };
        }
    }
    catch (std::exception& e)
    {
        std::cout << "Exception: " << e.what() << "\n";
    }

    //return cv::Mat();
}

void LOFTR::match(cv::Mat img1, cv::Mat img2, cv::Mat msk1, cv::Mat msk2,
    std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2)
{
    cv::Mat paddedImg1 = img1;
    cv::Mat paddedImg2 = img2;
    cv::Mat paddedMsk1 = msk1;
    cv::Mat paddedMsk2 = msk2;
    if (img1.cols != img2.cols || img1.rows != img2.rows) {
        int maxw = std::max(img1.cols, img2.cols);
        int maxh = std::max(img1.rows, img2.rows);
        paddedImg1 = cv::Mat(maxh, maxw, CV_8UC3, cv::Scalar(0, 0, 0));
        paddedImg2 = cv::Mat(maxh, maxw, CV_8UC3, cv::Scalar(0, 0, 0));
        img1.copyTo(paddedImg1({ 0, 0, img1.cols, img1.rows }));
        img2.copyTo(paddedImg2({ 0, 0, img2.cols, img2.rows }));
        if (!msk1.empty()) {
            paddedMsk1 = cv::Mat(maxh, maxw, CV_8UC1, cv::Scalar(0));
            paddedMsk2 = cv::Mat(maxh, maxw, CV_8UC1, cv::Scalar(0));
            msk1.copyTo(paddedMsk1({ 0, 0, msk1.cols, msk1.rows }));
            msk2.copyTo(paddedMsk2({ 0, 0, msk2.cols, msk2.rows }));
        }
    }

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> m;
    _request_matches(paddedImg1, paddedImg2, m);

    for (int i = 0; i < m.first.size(); ++i) {
        bool msk1ok = msk1.empty() || (paddedMsk1.at<uchar>(m.first[i].y, m.first[i].x) > 0);
        bool msk2ok = msk2.empty() || (paddedMsk2.at<uchar>(m.second[i].y, m.second[i].x) > 0);
        if (msk1ok && msk2ok) {
            pts1.push_back(m.first[i]);
            pts2.push_back(m.second[i]);
        }
    }
}
