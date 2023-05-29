#include "mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
using namespace std;
using namespace cv;

int main()
{
   Mat image;
   VideoCapture capture(0);
   if (capture.isOpened())
   {
       qDebug("camera open successfully!");
   } else
   {
       qDebug("camera open failed!");
       return 0;
   }

   while (true)
   {
       capture.read(OutputArray(image));
       imshow("video", InputArray(image));
       int key = waitKey(1);
       if (key == 27)
       {
           qDebug("exitting...");
           break;
       }
   }
   return 0;
}
