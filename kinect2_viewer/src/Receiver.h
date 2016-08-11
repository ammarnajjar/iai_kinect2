#ifndef RECIEVER_H
#define RECIEVER_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iomanip>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/octree/octree.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>

#include "kinect2_viewer/Merosy_Obj.h"
#include "kinect2_viewer/Vec_Obj.h"
#include <kinect2_bridge/kinect2_definitions.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>


class Receiver
{
    public:
        enum Mode
        {
            IMAGE = 0,
            CLOUD,
            BOTH
        };
        int callback_counter;
        Eigen::Matrix4f Transform;

        Receiver(const std::string &topicColor, const std::string &topicDepth, const bool useExact, const bool useCompressed);
        void run(const Mode mode);

    private:
        std::mutex lock;

        const std::string topicColor, topicDepth;
        const bool useExact, useCompressed;

        bool updateImage, updateCloud;
        bool save;
        bool running;
        size_t frame;
        const size_t queueSize;

        cv::Mat color, depth, canny, resultImage;
        cv::Mat cameraMatrixColor, cameraMatrixDepth;
        cv::Mat lookupX, lookupY;

        std::vector<cv::Point2f> corner;
        std::vector<cv::Point2f> corner_opt;
        std::vector<cv::Vec4i> lines;

        struct edgeline
        {
            cv::Vec4i line_endpoints;
            int line_index;
            float line_slope;
            float line_intercept;
        } ;

        std::vector<edgeline> vec_line0;
        std::vector<edgeline> vec_line1;
        std::vector<edgeline> vec_line2;
        std::vector<edgeline> vec_line3;

        float CX,CY,FX,FY;
        float Center_u,Center_v,Center_u2,Center_v2;
        unsigned int Center_change;

        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

        ros::NodeHandle nh;
        ros::AsyncSpinner spinner;
        ros::Publisher pub_objvec;
        ros::Publisher pub_desk;

        ros::Subscriber sub_modulestate;

        bool bIsModuleRunning;

        image_transport::ImageTransport it;
        image_transport::SubscriberFilter *subImageColor, *subImageDepth;
        message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;
        message_filters::Synchronizer<ExactSyncPolicy> *syncExact;
        message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

        std::thread imageViewerThread;
        Mode mode;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clone;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_p;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_model;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_desk;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_desk2;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_sum;

        pcl::PointIndices::Ptr inliers;
        pcl::ModelCoefficients::Ptr coefficients;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane;

        struct desk_object
        {
            int obj_index;
            Eigen::Vector4f obj_centroid;
            bool obj_state;
            int obj_onobj;
            pcl::PointXYZRGB obj_minpt;
            pcl::PointXYZRGB obj_maxpt;
            pcl::PointXYZ obj_p1;
            pcl::PointXYZ obj_p2;
            float obj_dx;
            float obj_dy;
            Eigen::Quaternionf obj_qr;
            float obj_angle;
            pcl::PointCloud<pcl::PointXYZRGB> obj_ecke;
        };

        pcl::PointXYZRGB bb_minpt;
        pcl::PointXYZRGB bb_maxpt;

        int object_counter;
        std::vector<desk_object> vec_obj;

        std::vector<int> octqueue;
        std::vector<int> octqueue2;

        bool b_mode=1;
        int tenframecheck;
        pcl::PointXYZ p1;
        pcl::PointXYZ p2;
        int frame_counter;

        std::vector<int> newPointIdxVector;
        std::string s_mode;
        std::string s_mode2;
        std::vector<std::string> v_cluster_c;
        std::vector<pcl::PointXYZ> v_p1;
        std::vector<pcl::PointXYZ> v_p2;
        std::vector<Eigen::Quaternionf> v_qr;
        std::vector<Eigen::Vector3f> v_tr;
        std::vector<float> v_dx;
        std::vector<float> v_dy;
        std::vector<float> v_dz;

        std::stringstream s_cluster;
        std::stringstream s_cluster_c;
        std::stringstream s_cluster_o;
        pcl::PCDWriter writer;
        std::ostringstream oss;
        std::vector<int> params;
        pcl::PointCloud<pcl::Normal>::Ptr floor_normals;

        geometry_msgs::Pose merosy_desk;

        kinect2_viewer::Merosy_Obj merosy_object;
        kinect2_viewer::Vec_Obj merosy_objvec;

        // Spatial change detection
        pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB> *octree;
        int noise_filter_;
        pcl::PassThrough<pcl::PointXYZRGB> pass;

        Eigen::Vector4f centroid,max_pt,min_pt;
        Eigen::Vector3f origin_coeff;
        Eigen::Vector3f desire_coeff;
        Eigen::Vector3f modelVectorAxisPoint;

        ros::Time timeStamp;

        void start(const Mode mode);

        void stop();

        void TransformPoint(float &x, float &y, float &z);

        cv::Point2f computeIntersect(cv::Vec4f a, cv::Vec4f b);

        void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth);

        void imageViewer();

        void imageProcesser(cv::Mat &color, cv::Mat &depth);

        //======================================================================================================================
        //==================================================CLOUDPROCESSER======================================================
        //======================================================================================================================

        void cloudProcesser(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

        void cloudViewer();

        void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *);

        void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const;

        void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const;

        void dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue);

        void combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out);

        void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const;

        void saveCloudAndImages(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored);

        void createLookup(size_t width, size_t height);
};

#endif /* end of include guard: RECIEVER_H */

/* vim: set ft=cpp ts=4 sw=4 et ai : */
