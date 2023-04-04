#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat img = imread("../example/Desk1.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cout << "无法打开或找到图像" << endl;
        return -1;
    }

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Mat blurred;
    // GaussianBlur(gray, blurred, Size(25, 25), 0);

    // Mat binary;
    // adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

    Mat edges;
    Canny(gray, edges, 0, 50);
    imwrite("../example/Desk1_edges.jpg", edges);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    imwrite("../example/Desk1_lines.jpg", edges);
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1, false) > contourArea(c2, false);
    });

    vector<Point> screenCnt;
    for (const auto& c : contours) {
        double peri = arcLength(c, true);
        vector<Point> approx;
        approxPolyDP(c, approx, 0.02 * peri, true);

        if (approx.size() == 4) {
            screenCnt = approx;
            break;
        }
    }

    if (!screenCnt.empty()) {
        for (size_t i = 0; i < screenCnt.size(); i++) {
            line(img, screenCnt[i], screenCnt[(i + 1) % 4], Scalar(0, 0, 255), 2);
        }
    }

    imwrite("../example/Desk1_result.jpg", img);
    waitKey(0);

    return 0;
}
