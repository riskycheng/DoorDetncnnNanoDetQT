# DoorDetncnnNanoDetQT
Qt：5.12.8
opencv：3.4.14

## qt安装----------------------------------------------------------------------------------
因为使用官网下载的版本始终安装失败，所以我可以直接使用命令行进行下载安装：

- 1：首先先将ubuntu的软件更新,并更新镜像源
```shell
  sudo apt-get update
  sudo apt-get upgrade
```

- 2：使用如下步骤安装Qt
```shell
  sudo apt-get install build-essential
  sudo apt-get install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
  sudo apt-get install qtcreator
  sudo apt-get install qt5*
```

## opencv安装----------------------------------------------------
这里选用的是opencv3.4.14

- 1：首先安装依赖项
```shell
  sudo apt-get install build-essential
  sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libatlas-base-dev gfortran libgtk2.0-dev libjpeg-dev libpng-dev
```

- 2：下载编译安装opencv--> Releases - OpenCV

- 3：进入opencv文件夹新建一个build目录
```shell  
  mdir build
  cd build
  cmake ../
  sudo make
```

- 4：make完毕后执行
```shell
  sudo make install
```

- 5：至此opencv编译安装完成

- 6：接着我们需要配置动态库环境
```shell
  sudo vim /etc/ld.so.conf
```
在文末加入:
```shell
  include /etc/ld.so.conf.d/*.conf
  /usr/local/lib
```
接着使之生效:
```shell
sudo /sbin/ldconfig -v  
```

- 7：可以查看opencv安装后的库
```shell
  ls -l /usr/local/include/opencv2
  ls -l /usr/local/lib/libopencv*
```

- 8：至此 opencv安装完成


## Qt配置opencv------------------------------------------------------------
1：首先创建Qt项目，使用qmake，创建完后打开.pro文件加入下图所示opencv库的路径
```shell
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui


INCLUDEPATH += /usr/local/include/ \
               /usr/local/include/opencv \
               /usr/local/include/opencv2
LIBS += /usr/local/lib/lib*
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
```

2 : 我们可以使用v4l2-ctl --list-devices来获取usb摄像头的节点:
```shell
root@teamhd:/home/teamhd/opencvTest_QT# sudo v4l2-ctl --list-devices
C670i FHD Webcam (usb-fc800000.usb-1):
	/dev/video0
	/dev/video1
	/dev/media0

HIK 2K Camera: HIK 2K Camera (usb-xhci-hcd.3.auto-1):
	/dev/video2
	/dev/video3
	/dev/media1
```

3 : 打开main.cpp文件，输入以下代码
```c++
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
```

4 : 完成
