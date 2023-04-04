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

    Mat blurred;
    GaussianBlur(gray, blurred, Size(31, 31), 0);

    Mat binary;
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);

    Mat edges;
    Canny(binary, edges, 1, 10);
    imwrite("../example/Desk1_edges.jpg", edges);
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 100, 10);
    imwrite("../example/Desk1_lines.jpg", edges);
    vector<Point> points;
    for (size_t i = 0; i < lines.size(); i++) {
        points.push_back(Point(lines[i][0], lines[i][1]));
        points.push_back(Point(lines[i][2], lines[i][3]));
    }

    Vec4f fitted_line;
    fitLine(points, fitted_line, DIST_L2, 0, 0.01, 0.01);

    Point point1(fitted_line[2] - 1000 * fitted_line[0], fitted_line[3] - 1000 * fitted_line[1]);
    Point point2(fitted_line[2] + 1000 * fitted_line[0], fitted_line[3] + 1000 * fitted_line[1]);

    line(img, point1, point2, Scalar(0, 0, 255), 2);

    //imshow("Result", img);
    imwrite("../example/Desk1_result.jpg", img);
    waitKey(0);

    return 0;
}
