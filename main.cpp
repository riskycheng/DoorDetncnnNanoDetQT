#include "mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QDebug>
#include <ncnn/include/net.h>
#include <nanodet.h>
#include <vector>
#include <json.h>
#include <iostream>
#include <fstream>
#include <sys/shm.h>
#include <sys/sem.h>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace Json;
using namespace cv;
#define NMS_THRESHOLD 0.5F

// 结构体，用于在共享内存中存储消息
struct Message {
    bool isWritten;
    char content[1024 * 10];
};

int M_SHARED_MEMORY_ID;
int M_SHARED_SEM_ID;
Message * M_MESSAGE_ID;

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


struct DoorDet_config {
    float det_threshold;
    int compute_every_frames;
    bool sync_results_frame;
    int sharedMemID;
    int sharedSemID;
    bool sync_waiting_sharedMemory_consumed;
};

struct DoorDetResultInfo {
    object_rect boundingBox;
    int label; // 0 for close, 1 for open
    float conf; // the detected confidence
};

struct FusedResultInfo {
    std::vector<DoorDetResultInfo> doorInfoArray;
    int camera_idx;
    uint64_t timeStamp;
};


bool initSharedMemory(DoorDet_config config)
{
    printf("creating sharedMemory at ID:%d \n", config.sharedMemID);
    bool resultCode = true;
    M_SHARED_MEMORY_ID = shmget(config.sharedMemID, sizeof(Message), IPC_CREAT | 0666);
    if (M_SHARED_MEMORY_ID == -1)
    {
        printf("error: failed to create shared memory! \n");
        resultCode = false;
    }

    // connect to the shared memory
    M_MESSAGE_ID = (Message *)shmat(M_SHARED_MEMORY_ID, nullptr, 0);
    if (M_MESSAGE_ID == (Message *) -1)
    {
        printf("error: failed to attach the shared memory! \n");
        resultCode = false;
    }

    // create the shared sem
    M_SHARED_SEM_ID = semget(config.sharedSemID, 1, IPC_CREAT | 0666);
    if (M_SHARED_SEM_ID == -1)
    {
        printf("error: failed to create semaphore! \n");
        resultCode = false;
    }

    // init the semaphore
    semctl(M_SHARED_SEM_ID, 0, SETVAL, 0);

    // started the sharedMemory processing
    if (resultCode)
    {
        printf("Congrat: the shared memory stuff prepared successfully! \n");
    } else
    {
        printf("ERROR: the shared memory stuff preparation failed! \n");
    }

    return resultCode;
}

void writeToSharedMemory(string content, DoorDet_config config)
{
    // 获取信号量的当前值
    struct sembuf sops;
    sops.sem_num = 0;
    sops.sem_op = 0;
    sops.sem_flg = 0;
    if(config.sync_waiting_sharedMemory_consumed)
        semop(M_SHARED_SEM_ID, &sops, 1);

    // 写入消息到共享内存
    strncpy(M_MESSAGE_ID->content, content.c_str(), sizeof(M_MESSAGE_ID->content));
    M_MESSAGE_ID->isWritten = true;

    // 释放信号量
    sops.sem_op = 1;
    semop(M_SHARED_SEM_ID, &sops, 1);
}

void releaseSharedMemory() {
    shmdt(M_MESSAGE_ID);

    // 删除共享内存
    shmctl(M_SHARED_MEMORY_ID, IPC_RMID, nullptr);

    // 删除信号量
    semctl(M_SHARED_SEM_ID, 0, IPC_RMID);
}

void draw_bboxes(const cv::Mat& bgr, DoorDet_config config, const std::vector<BoxInfo>& bboxes, object_rect effect_roi, char* winName, int camera_id = 0, bool savingLogs = true, char* logPath = nullptr, uint64_t timeStamp = 0, bool append = false);
//声明
string WriteFileJson(char* filePath, FusedResultInfo info, bool append);

//定义
string WriteFileJson(char* filePath, FusedResultInfo info, bool append)
{
    Json::Value root;
    root["camera_idx"] = Json::Value(info.camera_idx);
    root["timeStamp"] = Json::Value(info.timeStamp);

    bool anyDoorOpen = false;
    //数组形式
    root["doors"] = Json::arrayValue;
    for (auto& item : info.doorInfoArray)
    {
        Json::Value box;
        box["x"] = Json::Value(item.boundingBox.x);
        box["y"] = Json::Value(item.boundingBox.y);
        box["width"] = Json::Value(item.boundingBox.width);
        box["height"] = Json::Value(item.boundingBox.height);
        box["status"] = Json::Value(item.label);
        box["confidence"] = Json::Value(item.conf);
        root["doors"].append(box);
        if (item.label > 0)
            anyDoorOpen = true;
    }

    // write out
    root["anyDoorOpen"] = Json::Value(anyDoorOpen);

    /* 测试内容：会在屏幕输出 */
    Json::StyledWriter sw;
    //将内容输入到指定的文件
    ofstream os;
    if (append)
    {
        os.open(filePath, std::ios::app);
    } else
    {
        os.open(filePath, std::ios::binary);
    }

    if(!os.is_open())
    {
        printf("Error: can not find or create the file which named : %s \n", filePath);
    }

    os << sw.write(root);

    // write to string
    string dataStr = root.toStyledString();
    os.close();
    return dataStr;
}

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
    config.det_threshold = det_threshold;
    int compute_every_frames = json_obj["compute_every_frames"].asInt();
    config.compute_every_frames = compute_every_frames;
    bool sync_results_frame = json_obj["sync_results_frame"].asBool();
    config.sync_results_frame = sync_results_frame;
    int shared_memory_key = json_obj["shared_memory_key"].asInt();
    config.sharedMemID = shared_memory_key;
    int shared_sem_key = json_obj["shared_sem_key"].asInt();
    config.sharedSemID = shared_sem_key;
    bool sync_waiting_sharedMemory_consumed = json_obj["sync_waiting_sharedMemory_consumed"].asBool();
    config.sync_waiting_sharedMemory_consumed = sync_waiting_sharedMemory_consumed;


    // check the configs
    printf("parsed Configs STARTED\n");
    printf("det_threshold:%.2f\n", config.det_threshold);
    printf("compute_every_frames:%d\n", config.compute_every_frames);
    printf("sync_results_frame:%s\n", config.sync_results_frame ? "true" : "false");
    printf("sharedMemID:%d\n", config.sharedMemID);
    printf("sharedSemID:%d\n", config.sharedSemID);
    printf("sync_waiting_sharedMemory_consumed:%s\n", config.sync_waiting_sharedMemory_consumed ? "true" : "false");
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
    {0 ,0 , 255}
};

void draw_bboxes(const cv::Mat& bgr, DoorDet_config config, const std::vector<BoxInfo>& bboxes, object_rect effect_roi, char* winName, int camera_id, bool savingLogs, char* logPath, uint64_t timeStamp, bool append)
{
    static const char* class_names[] = {"box_close", "box_open"};

    cv::Mat image = bgr.clone();
    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    bool anyDoorOpen = false;

    FusedResultInfo results;
    results.camera_idx = camera_id;
    results.timeStamp = timeStamp;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        if (bbox.label > 0)
        {
            // indicates it is open status
            anyDoorOpen = true;
        }
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);
        auto obj_rect = cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                 cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio));
        cv::rectangle(image, obj_rect, color);

        // put it to the fused structure
        DoorDetResultInfo doorInfo;
        doorInfo.label = bbox.label;
        doorInfo.conf = bbox.score;
        doorInfo.boundingBox.x = obj_rect.x;
        doorInfo.boundingBox.y = obj_rect.y;
        doorInfo.boundingBox.width = obj_rect.width;
        doorInfo.boundingBox.height = obj_rect.height;
        results.doorInfoArray.push_back(doorInfo);

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

    // render the warning box if needed
    if (anyDoorOpen)
    {
        int baseLine = 0;
        float fontScale = 1.f;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        int fontThickness = 6;
        Scalar fontBorderColor(0, 255, 255);
        Scalar fontColor(255, 255, 255);
        Scalar fontBackground(0xf9, 0x8d, 0x85);
        const char* warningTxt = "Door Open!!";
        cv::Size txtSize = cv::getTextSize(warningTxt, fontFace, fontScale, fontThickness, &baseLine);

        int x = (image.cols - txtSize.width) / 2;
        int y = 100;
        // render the outer rect
        int margin = 2;
        cv::rectangle(image, cv::Rect(cv::Point(x - margin, y - margin), cv::Size(txtSize.width + 2 * margin, txtSize.height + baseLine + 2 * margin)), fontBorderColor, 4);
        // render the inner rect
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(txtSize.width, txtSize.height + baseLine)), fontBackground, -1);
        // render the inner text
        cv::putText(cv::InputOutputArray(image), warningTxt, cv::Point(x, y + txtSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor);
    }

    cv::imshow(winName, cv::InputArray(image));

    // saving out results
    if (!savingLogs) return;
    // construct the log path
    string result = WriteFileJson(logPath, results, true);
    writeToSharedMemory(result, config);
}


// dual cameras mode
int webcam_demo(NanoDet& detector, DoorDet_config config, int cam_id_1, int cam_id_2)
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

    std::time_t result = std::time(nullptr);
    char* logPath = new char[100]();
    sprintf(logPath, "./log_%d.txt", result);

    int64_t frameIndex = -1;
    std::vector<BoxInfo> results_cam1, results_cam2;
    while (true)
    {
        // the flag whether the abnormal status detected
        bool isAnyDoorOpen = false;

        cap1 >> image1;

        frameIndex++;
        if (frameIndex > 10000)
            frameIndex = 0;

        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image1, resized_img, cv::Size(width, height), effect_roi);
        if (frameIndex % config.compute_every_frames == 0)
        {
            results_cam1.clear();
            results_cam1 = detector.detect(resized_img, config.det_threshold, NMS_THRESHOLD);
        } else
        {
            if (config.sync_results_frame)
                continue;
        }

        uint64_t timeStamp_ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        draw_bboxes(image1, config, results_cam1, effect_roi, winName1, cam_id_1, true, logPath, timeStamp_ms, true);

        for (auto box : results_cam1)
        {
            if (box.label > 0)
            {
                isAnyDoorOpen = true;
            }
        }
        cv::waitKey(1);

        cap2 >> image2;
        resize_uniform(image2, resized_img, cv::Size(width, height), effect_roi);
        if (frameIndex % config.compute_every_frames == 0)
        {
            results_cam2.clear();
            results_cam2 = detector.detect(resized_img, config.det_threshold, NMS_THRESHOLD);
        }

        draw_bboxes(image2, config, results_cam2, effect_roi, winName2, cam_id_2, true, logPath, timeStamp_ms, true);

        for (auto box : results_cam2)
        {
            if (box.label > 0)
            {
                isAnyDoorOpen = true;
            }
        }

        cv::waitKey(1);

        // the summarized info
        if (isAnyDoorOpen)
        {
            printf("WARNING: detected open door via the 2-ways-camera!!\n");
        }
    }
    delete[] winName1;
    delete[] winName2;
    delete[] logPath;
    return 0;
}

// single camera mode
int webcam_demo(NanoDet& detector, DoorDet_config config, int cam_id)
{
    cv::Mat image;
    cv::VideoCapture cap(cam_id);
    int height = detector.input_size[0];
    int width = detector.input_size[1];
    if (!cap.isOpened())
    {
        printf("failed to open camera %d\n", cam_id);
        return -1;
    }
    std::time_t result = std::time(nullptr);
    char* logPath = new char[100]();
    sprintf(logPath, "./log_%d.txt", result);
    char* winName = new char[10]();
    sprintf(winName, "WIN_%d", cam_id);

    std::vector<BoxInfo> results;
    int frameIndex = -1;
    while (true)
    {
        cap >> image;
        frameIndex++;
        if (frameIndex > 10000)
            frameIndex = 0;
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        if (frameIndex % config.compute_every_frames == 0 || frameIndex < 0)
        {
            printf("this frame %d is computed\n", frameIndex);
            results = detector.detect(resized_img, config.det_threshold, NMS_THRESHOLD);
        } else
        {
            if (config.sync_results_frame)
                continue;
        }
        uint64_t timeStamp_ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        draw_bboxes(image, config, results, effect_roi, winName, cam_id, true, logPath, timeStamp_ms, true);
        cv::waitKey(1);
    }
    delete[] winName;
    delete[] logPath;
    return 0;
}



int video_demo(NanoDet& detector, const DoorDet_config config, const char* path)
{
    cv::Mat image;
    cv::VideoCapture cap(path);
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    std::time_t result = std::time(nullptr);

    char* logPath = new char[100]();
    sprintf(logPath, "./log_%d.txt", result);

    printf("config.thresh:%.2f\n", config.det_threshold);
    std::vector<BoxInfo> results;
    int frameIndex = -1;
    while (true)
    {
        cap >> image;

        frameIndex++;
        if (frameIndex > 10000)
            frameIndex = 0;

        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);

        if (frameIndex % config.compute_every_frames == 0)
        {
            results = detector.detect(resized_img, config.det_threshold, NMS_THRESHOLD);
        } else
        {
            if (config.sync_results_frame)
                continue;
        }

        uint64_t timeStamp_ms = duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        draw_bboxes(image, config, results, effect_roi, "video", 0, true, logPath, timeStamp_ms, true);
        cv::waitKey(1);
    }
       delete[] logPath;
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

   initSharedMemory(config);

   switch (mode)
   {
     case 0:
     {
        if (argc == 3)
        {
          int cam_id = atoi(argv[2]);
          webcam_demo(detector, config, cam_id);
        } else if (argc == 4)
        {
           int cam_id_1 = atoi(argv[2]);
           int cam_id_2 = atoi(argv[3]);
           webcam_demo(detector, config, cam_id_1, cam_id_2);
        }
        break;
     }

     case 1:
     {
        const char* path = argv[2];
        video_demo(detector, config, path);
        break;
     }

     default:
     {
         fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam ids (single cam: e.g. 0, dual cams: e.g. 0 2); \n For video, mode=1 path=video.mp4.\n", argv[0]);
         break;
     }
   }

   releaseSharedMemory();
   return 0;
}





int main_()
{
    NanoDet detector = NanoDet("/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.param", "/home/teamhd/opencvTest_QT/ncnn_models/nanodet_door.bin", true);

    DoorDet_config config;
    bool ret = parseConfig("./config.json", config);
    if (ret)
    {
        printf("config parsing successfully! \n");
    } else
    {
        printf("warning : config parsing failed! \n");
    }

    initSharedMemory(config);

    webcam_demo(detector, config, 0);

    //video_demo(detector, "/home/teamhd/Downloads/video_09_02_230317_nightOpen_reserved_TEST.mp4");

    releaseSharedMemory();
    return 0;
}


