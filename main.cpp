#include "mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <ncnn/include/net.h>
#include "nanodet.h"
#include <vector>
#include <json.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Json;
using namespace cv;

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


struct DoorDet_config {
    float det_threshold;
};

bool parseConfig(const char* filename, DoorDet_config& config)
{
    std::ifstream ifs;
    ifs.open(filename);
    if(!ifs.is_open()){
        printf("failed to open this config file >>> %s\n", filename);
        return false;
    }

    bool res;
    Json::CharReaderBuilder readerBuilder;
    string err_json;
    Json::Value json_obj, lang, mail;
    try{
        bool ret = Json::parseFromStream(readerBuilder, ifs, &json_obj, &err_json);
        if(!ret){
            printf("invalid json file ! \n");
            return false;
        }
    } catch(exception &e){
        printf("exception while parse json file %s due to %s \n", filename, e.what());
        return false;
    }

    // start parsing
    float det_threshold = json_obj["det_threshold"].asFloat();


    // check the configs
    printf("parsed Configs STARTED\n");
    printf("det_threshold:%.2f\n", det_threshold);
    printf("parsed Configs ENDED\n");

    return true;

}

int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(cv::InputArray(src), cv::OutputArray(dst), dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;
    cv::resize(cv::InputArray(src), cv::OutputArray(tmp), cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}

const int color_list[2][3] =
{
    {216 , 82 , 24},
    {236 ,176 , 31}
};

void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi, char* winName)
{
    static const char* class_names[] = { "box_close", "box_open"  };

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);

        cv::putText(cv::InputOutputArray(image), text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow(winName, cv::InputArray(image));
}

// dual cameras mode
int webcam_demo(NanoDet& detector, int cam_id_1, int cam_id_2)
{
    cv::Mat image1, image2;
    cv::VideoCapture cap1(cam_id_1), cap2(cam_id_2);
    int height = detector.input_size[0];
    int width = detector.input_size[1];
    if (!cap1.isOpened())
    {
        printf("failed to open camera %d", cam_id_1);
        return -1;
    }

    if (!cap2.isOpened())
    {
        printf("failed to open camera %d", cam_id_2);
        return -1;
    }

    char* winName1 = new char[10]();
    sprintf(winName1, "WIN_%d", cam_id_1);

    char* winName2 = new char[10]();
    sprintf(winName2, "WIN_%d", cam_id_2);

    while (true)
    {
        cap1 >> image1;
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image1, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        draw_bboxes(image1, results, effect_roi, winName1);
        cv::waitKey(1);

        cap2 >> image2;
        resize_uniform(image2, resized_img, cv::Size(width, height), effect_roi);
        results = detector.detect(resized_img, 0.4, 0.5);
        draw_bboxes(image2, results, effect_roi, winName2);
        cv::waitKey(1);
    }
    return 0;
}

// single camera mode
int webcam_demo(NanoDet& detector, int cam_id)
{
    cv::Mat image;
    cv::VideoCapture cap(cam_id);
    int height = detector.input_size[0];
    int width = detector.input_size[1];
    if (!cap.isOpened())
    {
        printf("failed to open camera %d", cam_id);
        return -1;
    }

    char* winName = new char[10]();
    sprintf(winName, "WIN_%d", cam_id);

    while (true)
    {
        cap >> image;
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        draw_bboxes(image, results, effect_roi, winName);
        cv::waitKey(1);
    }
    return 0;
}



int video_demo(NanoDet& detector, const char* path)
{
    cv::Mat image;
    cv::VideoCapture cap(path);
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    while (true)
    {
        cap >> image;
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, 0.4, 0.5);
        draw_bboxes(image, results, effect_roi, "video");
        cv::waitKey(1);
    }
    return 0;
}


int main(int argc, char** argv)
{
   if (argc != 3 && argc != 4)
   {
       fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam ids (single cam: e.g. 0, dual cams: e.g. 0 2); \n For video, mode=1 path=video.mp4.\n", argv[0]);
       return -1;
   }

   NanoDet detector = NanoDet("/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.param", "/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.bin", true);
   int mode = atoi(argv[1]);

   DoorDet_config config;
   bool ret = parseConfig("./config.json", config);
   if (ret)
   {
       printf("config parsing successfully! \n");
   } else
   {
       printf("warning : config parsing failed! \n");
   }

   switch (mode)
   {
     case 0:
     {
        if (argc == 3)
        {
          int cam_id = atoi(argv[2]);
          webcam_demo(detector, cam_id);
        } else if (argc == 4)
        {
           int cam_id_1 = atoi(argv[2]);
           int cam_id_2 = atoi(argv[3]);
           webcam_demo(detector, cam_id_1, cam_id_2);
        }
        break;
     }

     case 1:
     {
        const char* path = argv[2];
        video_demo(detector, path);
        break;
     }

     default:
     {
         fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam ids (single cam: e.g. 0, dual cams: e.g. 0 2); \n For video, mode=1 path=video.mp4.\n", argv[0]);
         break;
     }
   }
   return 0;
}


int main_()
{
    NanoDet detector = NanoDet("/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.param", "/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.bin", true);


    webcam_demo(detector, 0, 3);

    //video_demo(detector, "/home/teamhd/Downloads/video_09_02_230317_nightOpen_reserved_TEST.mp4");
    return 0;
}


