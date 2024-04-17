#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/core_c.h>

#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>
#include <fstream>

 
using namespace cv;


void filterCriterion(Mat img, std::string filter) {

    // counting difference between image and it's noised copy // 

    Mat filtered = Mat(img.size(), CV_32F), diff, combined;
    int d = sqrt(img.total()) / 100;
    d += 1 - d % 2;

    if (filter == "biliteral")
        bilateralFilter(img, filtered, d, 90, 90);
    else if (filter == "blur")
        GaussianBlur(img, filtered, Size(d, d), 0, 0, BORDER_DEFAULT);
    else {
        std::cout << "incorrect filter" << std::endl;
        exit(-1);
    }

    normalize(filtered, filtered, 0, 1, NORM_MINMAX, CV_32F);
    absdiff(img, filtered, diff);
    normalize(diff, diff, 0, 1, NORM_MINMAX, CV_32F);

    hconcat(img, filtered, combined);
    hconcat(combined, diff, combined);
    //imshow("img + " + filter + "+ diff", combined);
    //waitKey(0);

    Scalar diffCoefs = mean(diff);
    //std::cout << filter << " coef: " << diffCoefs[0] << std::endl;

    //normalize(img, img, 0, 255, NORM_MINMAX, CV_8U);
    //normalize(filtered, filtered, 0, 255, NORM_MINMAX, CV_8U);
    //normalize(diff, diff, 0, 255, NORM_MINMAX, CV_8U);
    //imwrite("image.jpg", img);
    //imwrite("filtered.jpg", filtered);
    //imwrite("diff.jpg", diff);


    //imshow("img + " + filter + "+ diff", combined);
    //waitKey(0);

    // finding countours // 
/*
    std::vector<std::vector<Vec2i>> contours, cc1, cc2, cc3;
    Mat res, a, b, c;
    Mat diff8U, diffThresh;

    normalize(img, diff8U, 0, 255, NORM_MINMAX, CV_8U);
    //normalize(img, diffThresh, 0, 255, NORM_MINMAX, CV_8U);
    threshold(diff8U, diffThresh, 80, 255, THRESH_BINARY); 
    findContours(diffThresh, contours, RETR_LIST, CHAIN_APPROX_NONE);

    std::vector<int> sizes;
    sizes.reserve(contours.size());
    std::sort(contours.begin(), contours.end(),
        [](std::vector<Vec2i> const &x, std::vector<Vec2i> const &y) -> bool { return x.size() > y.size(); });
    for (auto const& it : contours) {
        //std::cout << it.size() << " ";
        sizes.push_back(it.size());
    }

    // if long contours are mean of top 10%

    int thresh1 = std::accumulate(sizes.begin(), sizes.begin() + sizes.size()/10, 0) / sizes.size() * 10;
    int count1 = 0;

    for (int x : sizes)
        if (x > thresh1)
            count1++;

    for (auto cont : contours)
        if (cont.size() > thresh1)
            cc1.push_back(cont);

    cvtColor(img, a, COLOR_GRAY2BGR);
    drawContours(a, cc1, -1, Scalar(0, 255, 0), 1);
    normalize(a, a, 0, 255, NORM_MINMAX, CV_8U);
    //imwrite("a.jpg", a);

    std::cout << "above thresh1: " << count1 << ", part1 = " << (double)count1 / sizes.size() << std::endl;
    
    // if long contours are median and above

    int thresh2 = sizes[sizes.size() / 2];
    int count2 = 0;

    for (int x : sizes)
        if (x > thresh2)
            count2++;

    for (auto cont : contours)
        if (cont.size() > thresh2)
            cc2.push_back(cont);

    cvtColor(img, b, COLOR_GRAY2BGR);
    drawContours(b, cc2, -1, Scalar(0, 255, 0), 1);
    normalize(b, b, 0, 255, NORM_MINMAX, CV_8U);
    //imwrite("b.jpg", b);

    std::cout << "above thresh2: " << count2 << ", part2 = " << (double)count2 / sizes.size() << std::endl;

    // if long contours are mean and above

    int thresh3 = std::accumulate(sizes.begin(), sizes.end(), 0) / sizes.size();
    int count3 = 0;

    for (int x : sizes)
        if (x > thresh3)
            count3++;

    for (auto cont : contours)
        if (cont.size() > thresh3)
            cc3.push_back(cont);

    cvtColor(img, c, COLOR_GRAY2BGR);
    drawContours(c, cc3, -1, Scalar(0, 255, 0), 1);
    normalize(c, c, 0, 255, NORM_MINMAX, CV_8U);
    //imwrite("c.jpg", c);

    std::cout << "above thresh3: " << count3 << ", part3 = " << (double)count3 / sizes.size() << std::endl;
    std::cout << std::endl;
    
    cvtColor(img, res, COLOR_GRAY2BGR);
    drawContours(res, contours, -1, Scalar(0, 255, 0), 1);

    //imshow("input image with contours", res);
    //waitKey(0);
    //*/
}


double clusterization(Mat img, int K) {

    normalize(img, img, 0, 255, NORM_MINMAX, CV_8U);

    Mat samples(img.total(), 1, CV_32F);
    for (int x = 0; x < img.cols; x++)
        for (int y = 0; y < img.rows; y++)
            samples.at<float>(y + x * img.rows, 0) = img.at<uchar>(y, x);

    Mat labels, centers;
    int attempts = 10;
    kmeans(samples, K, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    Mat clustered(img.size(), img.type());
    for (int x = 0; x < img.cols; x++)
        for (int y = 0; y < img.rows; y++) {
            int clusterIndex = labels.at<int>(y + x * img.rows, 0);
            clustered.at<uchar>(y, x) = centers.at<float>(clusterIndex, 0);
        }


    std::vector<std::vector<double>> densityX(K, std::vector<double>(img.cols, 1E-15)),
                                     densityY(K, std::vector<double>(img.rows, 1E-15));
    std::vector<int> sizesOfClusters(K, 0);

    for (int x = 0; x < img.cols; x++)
        for (int y = 0; y < img.rows; y++) {

            int clusterIndex = labels.at<int>(y + x * img.rows, 0);

            densityX[clusterIndex][x]++;
            densityY[clusterIndex][y]++;

            sizesOfClusters[clusterIndex]++;
        }

    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        for (int x = 0; x < img.cols; x++)
            densityX[clusterIndex][x] /= sizesOfClusters[clusterIndex];
        for (int y = 0; y < img.rows; y++)
            densityY[clusterIndex][y] /= sizesOfClusters[clusterIndex];
    }


    // calculating the entropy //

    /*

    std::cout << "entropy:" << std::endl << std::endl;
    std::cout << "# | x     | y" << std::endl;
    std::vector<double> Hx(K), Hy(K);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {

        for (int x = 0; x < img.cols; x++) {
            double p = densityX[clusterIndex][x];
            Hx[clusterIndex] -= p * log(p);
        }

        for (int y = 0; y < img.rows; y++) {
            double p = densityY[clusterIndex][y];
            Hy[clusterIndex] -= p * log(p);
        }

        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << clusterIndex + 1 << " | " << Hx[clusterIndex] << " | " << Hy[clusterIndex] << std::endl;
    }

    //*/

    // calculating means and vars //

    /*
    int stepsCount = 10;
    int stepX = img.cols / stepsCount, stepY = img.rows / stepsCount;
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {

        std::cout << "Cluster number: " << clusterIndex << std::endl;
        std::cout << "     x      y" << std::endl;

        std::vector<double> meanDensityX(stepsCount, 0), varDensityX(stepsCount, 0),
                            meanDensityY(stepsCount, 0), varDensityY(stepsCount, 0);

        for (int stepNumber = 0; stepNumber < stepsCount; stepNumber++) {

            double meanDensityOnStepX = 0, meanDensityOnStepY = 0;
            for (int x = stepNumber * stepX; x < (stepNumber + 1) * stepX; x++) 
                meanDensityOnStepX += densityX[x][clusterIndex];
            for (int y = stepNumber * stepY; y < (stepNumber + 1) * stepY; y++)
                meanDensityOnStepY += densityY[y][clusterIndex];
            
            meanDensityOnStepX /= stepX;
            meanDensityOnStepY /= stepY;

            meanDensityX[stepNumber] = meanDensityOnStepX * 1000;
            meanDensityY[stepNumber] = meanDensityOnStepY * 1000;

            std::cout << std::fixed;
            std::cout << std::setprecision(2);
            std::cout << stepNumber + 1 << " | " << meanDensityX[stepNumber] << "   " << meanDensityY[stepNumber] << std::endl;
        }

        double meanX = std::accumulate(meanDensityX.begin(), meanDensityX.end(), 0.0) / stepsCount;
        double meanY = std::accumulate(meanDensityY.begin(), meanDensityY.end(), 0.0) / stepsCount;
        
        double varX = 0, varY = 0;
        for (int stepNumber = 0; stepNumber < stepsCount; stepNumber++) {
            varX += pow(meanX - meanDensityX[stepNumber], 2);
            varY += pow(meanY - meanDensityY[stepNumber], 2);
        }
        varX /= stepsCount, varY /= stepsCount;

        for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
            std::cout << std::setprecision(4);
            std::cout << "means: " << meanX << "   " << meanY << std::endl;
            std::cout << "devs:  " << sqrt(varX) << "   " << sqrt(varY) << std::endl;
        }

        std::cout << "RESULT: " << sqrt(varX + varY) << std::endl;
        std::cout << std::endl;
    }

    std::vector<double> meanX(K, 0), meanY(K, 0);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        for (int x = 0; x < img.cols; x++)
            meanX[clusterIndex] += densityX[x][clusterIndex];
        for (int y = 0; y < img.rows; y++)
            meanY[clusterIndex] += densityY[y][clusterIndex];

        meanX[clusterIndex] /= img.cols;
        meanY[clusterIndex] /= img.rows;
    }

    std::vector<double> varX(K, 0), varY(K, 0);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        for (int x = 0; x < img.cols; x++)
            varX[clusterIndex] += pow(meanX[clusterIndex] - densityX[x][clusterIndex], 2);
        for (int y = 0; y < img.rows; y++)
            varY[clusterIndex] += pow(meanY[clusterIndex] - densityY[y][clusterIndex], 2);

        varX[clusterIndex] /= img.cols;
        varY[clusterIndex] /= img.rows;

        varX[clusterIndex] = sqrt(varX[clusterIndex]);
        varY[clusterIndex] = sqrt(varY[clusterIndex]);
    }

    std::cout << "    divsX   divsY" << std::endl;
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << clusterIndex + 1 << " | " << varX[clusterIndex] * 1000 << " | " << varY[clusterIndex] * 1000 << std::endl;
    }
    */

    // counting distances between centers //

    /*
    double perimeter = 0;
    for (int i = 0; i < K - 1; i++)
        for (int j = i + 1; j < K; j++)
            perimeter += sqrt(pow(centers.at<float>(i, 0) - centers.at<float>(j, 0), 2) +
                pow(centers.at<float>(i, 1) - centers.at<float>(j, 1), 2));


    Mat clusteredWithCenters;
    cvtColor(clustered, clusteredWithCenters, COLOR_GRAY2BGR);
    for (int i = 0; i < K; i++) {
        //std::cout << centers.at<float>(i, 0) << " " << centers.at<float>(i, 1) << "\n";
        circle(clusteredWithCenters, Point(centers.at<float>(i, 0), centers.at<float>(i, 1)), 20, Scalar(0, 0, 255), 10);
    }

    std::cout << "k = " << K << " p / sqrt(total) = " << perimeter / sqrt(img.total()) << std::endl;
    //imshow("original", img);
    imshow("clustered image with centers", clusteredWithCenters);
    normalize(clusteredWithCenters, clusteredWithCenters, 0, 255, NORM_MINMAX, CV_8U);
    imwrite("clusteredWithCenters.jpg", clusteredWithCenters);
    waitKey(0);
//*/   

    // means and vers v.2 //

    std::vector<std::vector<double>> densityX2(K, std::vector<double>(img.cols, 1.0E-15)),
                                     densityY2(K, std::vector<double>(img.rows, 1.0E-15));

    for (int x = 0; x < img.cols; x++)
        for (int y = 0; y < img.rows; y++) {
            int clusterIndex = labels.at<int>(y + x * img.rows, 0);
            densityX2[clusterIndex][x]++;
            densityY2[clusterIndex][y]++;
        }

    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        for (int x = 0; x < img.cols; x++)
            densityX2[clusterIndex][x] /= img.rows;
        for (int y = 0; y < img.rows; y++)
            densityY2[clusterIndex][y] /= img.cols;
    }


    std::vector<double> meanX2(K), meanY2(K);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        meanX2[clusterIndex] = (double) sizesOfClusters[clusterIndex] / img.total();
        meanY2[clusterIndex] = (double) sizesOfClusters[clusterIndex] / img.total();
    }
    
    std::vector<double> varX2(K, 0), varY2(K, 0);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        for (int x = 0; x < img.cols; x++)
            varX2[clusterIndex] += pow(meanX2[clusterIndex] - densityX2[clusterIndex][x], 2);
        for (int y = 0; y < img.rows; y++)
            varY2[clusterIndex] += pow(meanY2[clusterIndex] - densityY2[clusterIndex][y], 2);

        varX2[clusterIndex] = sqrt(varX2[clusterIndex] / img.cols);
        varY2[clusterIndex] = sqrt(varY2[clusterIndex] / img.rows);
    }

    double RMS = 0, AVR = 0;
    //std::cout << "    divsX   divsY" << std::endl;
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
        //std::cout << std::fixed;
        //std::cout << std::setprecision(3);
        //std::cout << clusterIndex + 1 << " | " << varX2[clusterIndex] << " | " << varY2[clusterIndex] << std::endl;
        RMS += pow(varX2[clusterIndex], 2) + pow(varY2[clusterIndex], 2);
        AVR += varX2[clusterIndex] + varY2[clusterIndex];
    }


    // entropy for densuty2 //

    /*
    std::cout << "entropy:" << std::endl;
    std::cout << "# | x     | y" << std::endl;
    std::vector<double> Hx(K), Hy(K);
    for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {

        for (int x = 0; x < img.cols; x++) {
            double p = densityX2[clusterIndex][x];
            Hx[clusterIndex] -= p * log(p);
        }

        for (int y = 0; y < img.rows; y++) {
            double p = densityY2[clusterIndex][y];
            Hy[clusterIndex] -= p * log(p);
        }

        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << clusterIndex + 1 << " | " << Hx[clusterIndex] << " | " << Hy[clusterIndex] << std::endl;

    }
    */

    // std::cout << "RMS = " << sqrt(RMS) << " AVR = " << AVR / (2 * K) << std::endl;
    return sqrt(RMS);
//*/
}


int main(int argc, char** argv) {

    Mat src, noise, black, srcGray, noiseGray, blackGray;
    std::string folderPath = "D:/Downloads/images/images"; 
    //std::string folderPath = "C:/Users/User/source/repos/opencv_test2/images";
    std::vector<String> fileNames;
    glob(folderPath, fileNames);

    for (const auto& fileName : fileNames) {
        src = imread(fileName);

        if (src.empty()) {
            std::cout << "Error opening image " << fileName << std::endl;
            return -1;
        }

        cvtColor(src, srcGray, COLOR_BGR2GRAY);
        normalize(srcGray, srcGray, 0, 1, NORM_MINMAX, CV_32F);

        noise = Mat(src.size(), CV_32F);
        randn(noise, 1, 0.1);
        normalize(noise, noise, 0, 1, NORM_MINMAX, CV_32F);
        noise.convertTo(noiseGray, CV_32F);


        int maxClusters = 4;
        for (int i = 2; i <= maxClusters; i++) {
            double imgCoef = clusterization(srcGray, i);
            double noiseCoef = clusterization(noiseGray, i);
            double SNR = imgCoef / noiseCoef;
            std::cout << i << ": " << SNR / 10 << std::endl;
        }

        //filterCriterion(srcGray, "blur");
        //filterCriterion(srcGray, "biliteral");
        
        imshow("input image", srcGray);
        waitKey(0);

        /*
        noise = Mat(srcGray.size(), CV_32F);
        randn(noise, 5, 5);
        normalize(noise, noise, 0, 1, NORM_MINMAX, CV_32F);
        Mat srcNoised = srcGray + noise;
        
        for (int i = 2; i <= maxClusters; i++) {
            std::cout << "(NOISE)\nAmount of clusters: " << i << std::endl;
            clusterization(srcNoised, i);
            std::cout << std::endl;
        }
        //*/

        destroyAllWindows();
    }
    return 0;
}