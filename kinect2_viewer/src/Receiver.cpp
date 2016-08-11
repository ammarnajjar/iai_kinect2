#include "Receiver.h"

Receiver::Receiver(const std::string &topicColor, const std::string &topicDepth, const bool useExact, const bool useCompressed)
    : topicColor(topicColor), topicDepth(topicDepth), useExact(useExact), useCompressed(useCompressed),
    updateImage(false), updateCloud(false), save(false), running(false), frame(0), queueSize(5),
    nh("~"), spinner(0), it(nh), mode(CLOUD)
{
    cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
    cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(100);
    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    params.push_back(1);
    params.push_back(cv::IMWRITE_PNG_STRATEGY);
    params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
    params.push_back(0);
}

void Receiver::run(const Mode mode)
{
    start(mode);
    stop();
}


void Receiver::start(const Mode mode)
{
    this->mode = mode;
    running = true;

    std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
    std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

    image_transport::TransportHints hints(useCompressed ? "compressed" : "raw");
    subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
    subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
    subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
    subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

    pub_desk= nh.advertise<geometry_msgs::Pose>("/merosy_desk",1);
    pub_objvec = nh.advertise<kinect2_viewer::Vec_Obj>("/merosy_objects",1);

    bIsModuleRunning = true;

    if(useExact)
    {
        syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncExact->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
    }
    else
    {
        syncApproximate = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncApproximate->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));
    }

    spinner.start();

    std::chrono::milliseconds duration(1);
    while(!updateImage || !updateCloud)
    {
        if(!ros::ok())
        {
            return;
        }
        std::this_thread::sleep_for(duration);
    }
    cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->height = color.rows;
    cloud->width = color.cols;
    cloud->is_dense = false;
    cloud->points.resize(cloud->height * cloud->width);
    createLookup(this->color.cols, this->color.rows);

    float resolution = 0.05f;
    s_cluster_c<<".";
    s_cluster_o<<".";
    s_cluster<<".";
    tenframecheck=0;
    frame_counter=0,
        inliers=pcl::PointIndices::Ptr(new pcl::PointIndices);
    coefficients = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients);
    cloud_plane = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB> ());

    cloud_p = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_filtered = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_clone = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_b = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_model = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_desk = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_desk2 = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_f = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_sum = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());

    octree = new pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZRGB>(resolution); // resolution 0.1f
    Center_u=0.0;Center_v=0.0;Center_change=100;Center_u2=0.0;Center_v2=0.0;

    switch(mode)
    {
        case CLOUD:
            cloudViewer();
            break;
        case IMAGE:
            imageViewer();
            break;
        case BOTH:
            imageViewerThread = std::thread(&Receiver::imageViewer, this);
            cloudViewer();
            break;
    }
}


void Receiver::stop()
{
    spinner.stop();
    if(useExact) {
        delete syncExact;
    } else {
        delete syncApproximate;
    }

    delete subImageColor;
    delete subImageDepth;
    delete subCameraInfoColor;
    delete subCameraInfoDepth;

    running = false;
    if(mode == BOTH) {
        imageViewerThread.join();
    }
}


void Receiver::TransformPoint(float &x, float &y, float &z)
{
    Eigen::Matrix4f MatInverseTransform = Transform.inverse();
    Eigen::Vector4f Point3D, PointTransformed;
    Point3D << x,y,z,1;

    PointTransformed = MatInverseTransform*Point3D;

    x = PointTransformed(0);
    y = PointTransformed(1);
    z = PointTransformed(2);

}


cv::Point2f Receiver::computeIntersect(cv::Vec4f a, cv::Vec4f b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

    if (float d = ((float)(x1-x2) * (float)(y3-y4)) - ((y1-y2) * (x3-x4))) {
        cv::Point2f pt;
        pt.x = (float)((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = (float)((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        return pt;
    } else {
        return cv::Point2f(-1, -1);
    }
}


void Receiver::callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
        const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
{
    if (imageColor->header.stamp < timeStamp) {
        std::cout<<"====================================================="<<std::endl;
        std::cout << "ROSBAG RESTARTED" << std::endl;
        vec_obj.clear();
        v_cluster_c.clear();
        v_p1.clear();
        callback_counter=0;
        frame_counter=0;
        s_cluster_c.str("");
        s_cluster_c.clear();

        s_cluster_o.str("");
        s_cluster_o.clear();

        s_cluster.str("");
        s_cluster.clear();

        s_cluster_c<<".";
        s_cluster_o<<".";
        s_cluster<<".";

        b_mode=0;
        std::cout<<"====================================================="<<std::endl;
    }

    timeStamp = imageColor->header.stamp;
    cv::Mat color, depth;

    readCameraInfo(cameraInfoColor, cameraMatrixColor);
    readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
    readImage(imageColor, color);
    readImage(imageDepth, depth);

    // IR image input
    if(color.type() == CV_16U) {
        cv::Mat tmp;
        color.convertTo(tmp, CV_8U, 0.02);
        cv::cvtColor(tmp, color, CV_GRAY2BGR);
    }

    lock.lock();
    this->color = color;
    this->depth = depth;
    updateImage = true;
    updateCloud = true;
    lock.unlock();
}


void Receiver::imageViewer()
{
    std::cout<<"ImageViewer started... callback counter is "<<callback_counter<<endl;
    cv::Mat color, depth, depthDisp, combined;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, now;
    double fps = 0;
    size_t frameCount = 0;
    std::ostringstream oss;
    const cv::Point pos(5, 15);
    const cv::Scalar colorText = CV_RGB(255, 255, 255);
    const double sizeText = 0.5;
    const int lineText = 1;
    const int font = cv::FONT_HERSHEY_SIMPLEX;

    cv::namedWindow("Image Viewer");
    oss << "starting...";

    start = std::chrono::high_resolution_clock::now();
    for(; running && ros::ok();)
    {
        if(updateImage)
        {
            std::cout<<"Image updated... callback counter is "<<callback_counter<<endl;
            lock.lock();
            color = this->color;
            depth = this->depth;

            updateImage = false;
            lock.unlock();

            ++frameCount;
            now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() / 1000.0;
            if(elapsed >= 1.0)
            {
                fps = frameCount / elapsed;
                oss.str("");
                oss << "fps: " << fps << " ( " << elapsed / frameCount * 1000.0 << " ms)";
                start = now;
                frameCount = 0;
            }

            dispDepth(depth, depthDisp, 12000.0f);
            combine(color, depthDisp, combined);
            // combined = color;
            cv::putText(combined, oss.str(), pos, font, sizeText, colorText, lineText, CV_AA);
            imageProcesser(color,depth);
            // combined=resultImage;
            cv::imshow("Image Viewer", resultImage);
        }

        int key = cv::waitKey(1);
        switch(key & 0xFF)
        {
            case 27:
            case 'q':
                running = false;
                break;
            case ' ':
            case 's':
                if(mode == IMAGE)
                {
                    createCloud(depth, color, cloud);
                    saveCloudAndImages(cloud, color, depth, depthDisp);
                }
                else
                {
                    save = true;
                }
                break;
        }
    }
    cv::destroyAllWindows();
    cv::waitKey(100);
}


void Receiver::imageProcesser(cv::Mat &color, cv::Mat &depth)
{
    resultImage=color.clone();
    if (corner.size() == 0) return;

    int iROIXMin = corner[2].x-10;
    int iROIYMin = corner[0].y-20;
    int iROIWidth = corner[3].x-corner[2].x+30;
    int iROIHeight = corner[3].y-corner[0].y+50;

    if (Center_change < 10) {
        cv::cvtColor(resultImage, canny, CV_BGR2GRAY);
        cv::GaussianBlur(canny, canny, cv::Size( 9, 9 ), 0);
        cv::Canny(canny,canny,250,550,5);
        //cv::imshow("Win", canny);
        //cv::waitKey(10);
        cv::Mat roi(canny, cv::Rect(iROIXMin, iROIYMin, iROIWidth, iROIHeight));
        cv::HoughLinesP(roi, lines, 1, CV_PI/180, 80, 0, 0 );

        for( size_t i = 0; i < lines.size(); i++ )
        {
            cv::Vec4i l = lines[i];
            cv::line( resultImage, cv::Point(l[0]+iROIXMin, l[1]+iROIYMin), cv::Point(l[2]+iROIXMin, l[3]+iROIYMin), cv::Scalar(0,0,255), 3, CV_AA);
        }

        std::cout<<"The no. of lines is "<<lines.size()<<endl;

        cv::imshow("Canny ROI", roi);
        cv::imshow("Win", resultImage);
        cv::waitKey(500);

    }

    if(Center_change == 0) {
        for( size_t i = 0; i < lines.size(); i++ )
        {
            lines[i][0]=lines[i][0]+iROIXMin;
            lines[i][2]=lines[i][2]+iROIXMin;
            lines[i][1]=lines[i][1]+iROIYMin;
            lines[i][3]=lines[i][3]+iROIYMin;

            edgeline new_line;
            new_line.line_endpoints=lines[i];
            new_line.line_slope=(float)(lines[i][1]-lines[i][3])/(lines[i][0]-lines[i][2]);

            cv::Vec4i l = lines[i];
            float tanangle=0.0;
            tanangle=fabs(l[1]-l[3])/fabs(l[0]-l[2]);
            float arctan=0.0;
            arctan=180*atan(tanangle)/3.1415;

            std::cout<<"== line l = "<<l<<endl;
            std::cout<<"Angle is "<<arctan<<endl;

            if(arctan<10&&fabs(l[1]-corner[0].y)<=50&&fabs(l[3]-corner[0].y)<=50)
            {
                new_line.line_intercept=(float)(-1)*new_line.line_slope*l[0]+(float)l[1];
                if(vec_line0.size()==0) {
                    new_line.line_index=0;
                    vec_line0.push_back(new_line);
                } else if(new_line.line_intercept>=vec_line0[0].line_intercept) {
                    vec_line0.clear();
                    vec_line0.push_back(new_line);
                }
            } else if (arctan<10&&fabs(l[1]-corner[2].y)<=50&&fabs(l[3]-corner[2].y)<=50) {
                new_line.line_intercept=(float)(-1)*new_line.line_slope*l[0]+(float)l[1];
                if(vec_line1.size()==0) {
                    new_line.line_index=1;
                    vec_line1.push_back(new_line);
                } else if(new_line.line_intercept<=vec_line1[0].line_intercept) {
                    vec_line1.clear();
                    vec_line1.push_back(new_line);
                }
            } else if ( arctan>70&&fabs(l[0]-corner[2].x)<=80&&fabs(l[2]-corner[2].x)<=80) {
                new_line.line_intercept=(float)(-1)*l[1]/new_line.line_slope+l[0];
                if(vec_line2.size()==0) {
                    new_line.line_index=2;
                    vec_line2.push_back(new_line);
                } else if(new_line.line_intercept>=vec_line2[0].line_intercept) {
                    vec_line2.clear();
                    vec_line2.push_back(new_line);
                }
            } else if ( arctan>70&&fabs(l[0]-corner[3].x)<=80&&fabs(l[2]-corner[3].x)<=80) {
                new_line.line_intercept=(float)(-1)*l[1]/new_line.line_slope+l[0];
                if(vec_line3.size()==0) {
                    new_line.line_index=3;
                    vec_line3.push_back(new_line);
                } else if(new_line.line_intercept<=vec_line3[0].line_intercept) {
                    vec_line3.clear();
                    vec_line3.push_back(new_line);
                }
            } else {
                lines.erase(lines.begin()+i);
                i--;
            }
        }

        std::cout<<"size of 4 vectors : " << vec_line0.size() << "" << vec_line1.size()
            <<""<<vec_line2.size()<<""<<vec_line3.size()<<""<<endl;

        Center_change++;
    } else if(Center_change>0&&Center_change<10) {
        for( size_t i = 0; i < lines.size(); i++ ) {
            lines[i][0]=lines[i][0]+iROIXMin;
            lines[i][2]=lines[i][2]+iROIXMin;
            lines[i][1]=lines[i][1]+iROIYMin;
            lines[i][3]=lines[i][3]+iROIYMin;

            edgeline new_line;
            new_line.line_endpoints=lines[i];
            new_line.line_slope=(float)(lines[i][1]-lines[i][3])/(lines[i][0]-lines[i][2]);

            cv::Vec4i l = lines[i];
            float tanangle=0.0;
            tanangle=fabs(l[1]-l[3])/fabs(l[0]-l[2]);
            float arctan=0.0;
            arctan=180*atan(tanangle)/3.1415;

            if(arctan<10&&fabs(l[1]-corner[0].y)<=50&&fabs(l[3]-corner[0].y)<=50) {
                new_line.line_intercept=(float)(-1)*new_line.line_slope*l[0]+l[1];
                if(vec_line0.size()==Center_change) {
                    new_line.line_index=0;
                    vec_line0.push_back(new_line);
                } else if(vec_line0.size() > Center_change && new_line.line_intercept>=vec_line0[Center_change].line_intercept) {
                    vec_line0.erase(vec_line0.begin()+Center_change);
                    vec_line0.push_back(new_line);
                }
            } else if (arctan<10&&(fabs(l[1]-corner[2].y)<=50)&&(fabs(l[3]-corner[2].y)<=50)) {
                new_line.line_intercept=(float)(-1)*new_line.line_slope*l[0]+l[1];
                std::cout<<"the new_line.line_intercept "<<new_line.line_intercept<<endl;
                if(vec_line1.size()==Center_change) {
                    std::cout<<"vec_line1.size()==Center_change, pushback "<<new_line.line_intercept<<endl;
                    new_line.line_index=1;
                    vec_line1.push_back(new_line);
                } else if(vec_line1.size() > Center_change && new_line.line_intercept<=vec_line1[Center_change].line_intercept) {
                    std::cout<<"is smaller, pushback and delete last "<<new_line.line_intercept<<endl;
                    vec_line1.erase(vec_line1.begin()+Center_change);
                    vec_line1.push_back(new_line);
                }
            } else if ( arctan>70&&fabs(l[0]-corner[2].x)<=80&&fabs(l[2]-corner[2].x)<=80) {
                new_line.line_intercept=(float)(-1)*l[1]/new_line.line_slope+l[0];
                if(vec_line2.size()==Center_change) {
                    new_line.line_index=2;
                    vec_line2.push_back(new_line);
                } else if(vec_line2.size() > Center_change && new_line.line_intercept>=vec_line2[Center_change].line_intercept) {
                    vec_line2.erase(vec_line2.begin()+Center_change);
                    vec_line2.push_back(new_line);
                }
            } else if ( arctan>70&&fabs(l[0]-corner[3].x)<=80&&fabs(l[2]-corner[3].x)<=80) {
                new_line.line_intercept=(float)(-1)*l[1]/new_line.line_slope+l[0];
                if(vec_line3.size()==Center_change) {
                    new_line.line_index=3;
                    vec_line3.push_back(new_line);
                } else if(vec_line3.size() > Center_change && new_line.line_intercept<=vec_line3[Center_change].line_intercept) {
                    vec_line3.erase(vec_line3.begin()+Center_change);
                    vec_line3.push_back(new_line);
                }
            } else {
                if (lines.size()> 0 && i<lines.size()) {
                    lines.erase(lines.begin()+i);
                    i--;
                }
            }
        }

        std::cout<<"size of 4 vectors : "<<vec_line0.size()<<" "<<vec_line1.size()
            <<" "<<vec_line2.size()<<" "<<vec_line3.size()<<""<<endl;

        if(Center_change==9) {
            //calculate the average
            if (vec_line0.size() == 0 || vec_line1.size() == 0 || vec_line2.size() == 0 || vec_line3.size() == 0) {
                ROS_ERROR("Line vectors: %d %d %d %d. Aborting!!!", (int)vec_line0.size(), (int)vec_line1.size(), (int)vec_line2.size(), (int)vec_line3.size());
                exit(1);
            }

            float average_slope0=0.0;
            float average_intercept0=0.0;
            for(size_t i=0;i<vec_line0.size();i++) {
                std::cout<<"slope0s are ..... "<<vec_line0[i].line_slope<<endl;
                std::cout<<"intercept0s are ..... "<<vec_line0[i].line_intercept<<endl;
                average_slope0= average_slope0+ vec_line0[i].line_slope;
                average_intercept0= average_intercept0+ vec_line0[i].line_intercept;
            }
            average_slope0=(float)average_slope0/vec_line0.size();
            average_intercept0=(float)average_intercept0/vec_line0.size();

            float average_slope1=0.0;
            float average_intercept1=0.0;
            for(size_t i=0;i<vec_line1.size();i++) {
                std::cout<<"slope1s are ..... "<<vec_line1[i].line_slope<<endl;
                std::cout<<"intercept1s are ..... "<<vec_line1[i].line_intercept<<endl;

                average_slope1= average_slope1+ vec_line1[i].line_slope;
                average_intercept1= average_intercept1+ vec_line1[i].line_intercept;

                cv::line( resultImage, cv::Point(vec_line1[i].line_endpoints[0],vec_line1[i].line_endpoints[1]), cv::Point(vec_line1[i].line_endpoints[2],vec_line1[i].line_endpoints[3]), cv::Scalar(255,0,0), 1, CV_AA);
            }
            average_slope1=(float)average_slope1/vec_line1.size();
            average_intercept1=(float)average_intercept1/vec_line1.size();
            std::cout<<"average slope1 = "<<average_slope1<<endl;

            float average_slope2=0.0;
            float average_intercept2=0.0;
            for(size_t i=0;i<vec_line2.size();i++) {
                std::cout<<"slope2s are ..... "<<vec_line2[i].line_slope<<endl;
                std::cout<<"intercept2s are ..... "<<vec_line2[i].line_intercept<<endl;

                average_slope2= average_slope2+ vec_line2[i].line_slope;
                average_intercept2= average_intercept2+ vec_line2[i].line_intercept;
            }
            average_slope2=(float)average_slope2/vec_line2.size();
            average_intercept2=(float)average_intercept2/vec_line2.size();

            float average_slope3=0.0;
            float average_intercept3=0.0;
            for(size_t i=0;i<vec_line3.size();i++) {
                std::cout<<"slope3s are ..... "<<vec_line3[i].line_slope<<endl;
                std::cout<<"intercept3s are ..... "<<vec_line3[i].line_intercept<<endl;

                average_slope3= (float)average_slope3+ vec_line3[i].line_slope;
                average_intercept3= (float)average_intercept3+ vec_line3[i].line_intercept;
            }
            average_slope3=(float)average_slope3/vec_line3.size();
            average_intercept3=(float)average_intercept3/vec_line3.size();

            cv::Vec4f l0,l1,l2,l3;

            l0[0]=0.0;l0[1]=average_intercept0;l0[2]=100.0;l0[3]=average_intercept0+100.0*average_slope0;
            l1[0]=0.0;l1[1]=average_intercept1;l1[2]=100.0;l1[3]=average_intercept1+100.0*average_slope1;

            l2[1]=0.0;l2[0]=average_intercept2;l2[2]=0.0;l2[3]=-average_intercept2*average_slope2;
            l3[1]=0.0;l3[0]=average_intercept3;l3[2]=2.0*average_intercept3;l3[3]=average_intercept3*average_slope3;

            std::cout<<"l0 l1 l2 l3    "<<l0<<" "<<l1<<" "<<l2<<" "<<l3<<endl;

            cv::Point2f pt = computeIntersect(l0, l2);
            std::cout<<"     inter     "<<pt<<endl;
            corner_opt.push_back(pt);
            pt = computeIntersect(l0, l3);
            corner_opt.push_back(pt);

            pt = computeIntersect(l1, l2);
            corner_opt.push_back(pt);
            pt = computeIntersect(l1, l3);
            corner_opt.push_back(pt);

            std::cout<<"corner_opt: "<<corner_opt<<endl;

            cv::Vec4i la,lb;
            la[0]=corner_opt[0].x;
            la[1]=corner_opt[0].y;
            la[2]=corner_opt[3].x;
            la[3]=corner_opt[3].y;

            lb[0]=corner_opt[1].x;
            lb[1]=corner_opt[1].y;
            lb[2]=corner_opt[2].x;
            lb[3]=corner_opt[2].y;

            pt = computeIntersect(la, lb);

            Center_u2=pt.x;
            Center_v2=pt.y;

            float zz=depth.at<uint16_t>(Center_v2,Center_u2)/1000.0;
            float xx=(Center_u2-CX)*FX*depth.at<uint16_t>(Center_v2,Center_u2)/1000.0;
            float yy=(Center_v2-CY)*FY*depth.at<uint16_t>(Center_v2,Center_u2)/1000.0;

            Eigen::Vector4f Point3D;
            Point3D <<xx,yy,zz,1;

            Point3D = Transform*Point3D;
            std::cout<<"POINT AFTER TRANSFORM "<<Point3D(0)<<" "<<Point3D(1)<<" "<<Point3D(2)<<endl;

            Eigen::Matrix4f Transform2;

            Transform2<<  1,  0,   0,    -Point3D(0),
                0,  1,   0,   -Point3D(1),
                0,  0,   1,   0,
                0,  0,   0,    1;

            Transform=Transform2*Transform;

            std::cout<<"DEpth at "<<xx<<" "<<yy<<" "<<zz<<endl;
            std::cout<<"TRANSFORM CHANGED, x y z "<<Transform<<endl;

            merosy_desk.position.x=Transform(0,3);
            merosy_desk.position.y=Transform(1,3);
        }

        Center_change++;
    } else if(Center_change==10) {
        resultImage=color.clone();
        cv::circle( resultImage, cv::Point(Center_u2, Center_v2), 5, cv::Scalar(0,255,0),3, 8, 0);

        cv::line( resultImage, corner_opt[0], corner_opt[1], cv::Scalar(255,0,0), 1, CV_AA);
        cv::line( resultImage, corner_opt[2], corner_opt[3], cv::Scalar(255,0,0), 1, CV_AA);
        cv::line( resultImage, corner_opt[0], corner_opt[2], cv::Scalar(255,0,0), 1, CV_AA);
        cv::line( resultImage, corner_opt[1], corner_opt[3], cv::Scalar(255,0,0), 1, CV_AA);

        for(size_t i=0;i<corner_opt.size();i++) {
            cv::circle( resultImage, corner_opt[i], 5, cv::Scalar(0,255,0),2, 8, 0);
        }
        std::cout<<"The no. of intersect is "<<corner_opt.size()<<endl;
    } else {
        resultImage=color;
    }
}


void Receiver::cloudProcesser(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    callback_counter ++;
    //-------z coordinate--------
    pcl::PassThrough<pcl::PointXYZRGB> pass_z;
    pass_z.setInputCloud (cloud);
    pass_z.setFilterFieldName ("x");
    pass_z.setFilterLimits (-20, 20);
    pass_z.filter (*cloud_clone);

    //===================================DOWN SAMPLING====================================
    // Down sample cloud using voxel grid

    //===================================the first frame====================================
    //down sampling, plane detection, desk detection, pass through
    if (callback_counter==1) {
        s_mode="MODE: Initial";
        s_mode2="Initial";

        //down sampling the 1st frame
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (cloud_clone);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter (*cloud_filtered);

        //normal-based plane detection
        int i=0,t=0;
        float cx=0,cy=0,cz=0;//float d=0;
        float nx=0,ny=0,nz=0;
        float dis=100;
        //  s_mode="MODE: Initial";
        // =============================normal estimation==================================
        pcl::search::Search<pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
        pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod (tree);
        normal_estimator.setInputCloud (cloud_filtered);

        normal_estimator.setRadiusSearch (0.05);
        normal_estimator.setViewPoint(0,-20,0);
        normal_estimator.compute (*normals);

        // =========================region growing planes clustering===============================
        pcl::RegionGrowing<pcl::PointXYZRGB, pcl::Normal> reg;
        reg.setMinClusterSize (1000);
        reg.setMaxClusterSize (1000000);
        reg.setSearchMethod (tree);
        reg.setNumberOfNeighbours (30);
        reg.setInputCloud (cloud_filtered);
        reg.setInputNormals (normals);
        reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold (1.0);   //_

        std::vector <pcl::PointIndices> cluster_indices;
        pcl::PointIndices::Ptr cluster_indices_desk (new pcl::PointIndices ());

        reg.extract (cluster_indices);
        std::cout<< "Number of planes is equal to " << cluster_indices.size () << std::endl;
        std::vector<pcl::PointIndices>::const_iterator it;

        //======================desk detecting by using spatial relation===========================
        // using the plane whose centroid is most nearest to the camera

        //======================calculate the normal of every plane===========================
        for(it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
            const std::vector<pcl::PointIndices> clusters_const=cluster_indices;
            const pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered_const=*cloud_filtered;
            float nx1;float ny1;float nz1;
            float curvature1;  Eigen::Vector4f pl1;

            computePointNormal (cloud_filtered_const, clusters_const[i].indices,pl1,curvature1);

            nx1=pl1(0);
            ny1=pl1(1);
            nz1=pl1(2);

            pcl::compute3DCentroid( *cloud_filtered,cluster_indices[i], centroid);
            float dis_current=sqrt(centroid[0]*centroid[0]+centroid[1]*centroid[1]+centroid[2]*centroid[2]);
            if(dis_current<dis)

            {
                cx=centroid[0];
                cy=centroid[1];
                cz=centroid[2];
                // normal
                nx=nx1;
                ny=ny1;
                nz=nz1;
                // index
                t=i;
                dis=dis_current;

            }
            i++;
        }

        cout<<"cx cy cz "<<cx<<" "<<cy<<" "<<cz<<endl;

        Center_u =  (cx/(FX*(cz))) + CX;
        Center_v =  (cy/(float)(FY*(cz))) + CY;

        *cluster_indices_desk=cluster_indices[t];

        // Make sure that the normal of the desk is pointing to the up side
        if(nz>0) {
            nx=-nx;
            ny=-ny;
            nz=-nz;
        }

        // Information of the desk to be publish
        merosy_desk.position.x=cx;
        merosy_desk.position.y=cy;
        merosy_desk.position.z=cz;

        merosy_desk.orientation.x=nx;
        merosy_desk.orientation.y=ny;
        merosy_desk.orientation.z=nz;

        merosy_desk.orientation.w=0;

        //========================centroid of desk as the new origin coordinate ===================
        centroid[0]=cx;
        centroid[1]=cy;
        centroid[2]=cz;

        desire_coeff[0] = 0;
        desire_coeff[1] = 0;
        desire_coeff[2] = 1;

        //====================================normal of the desk=====================================
        origin_coeff[0] = nx;
        origin_coeff[1] = ny;
        origin_coeff[2] = nz;

        //=================transform matrix calculation===============================
        // Step 1: Find axis (cross_norm)
        Eigen::Vector3f cross_product = origin_coeff.cross(desire_coeff);
        float cross_product_norm = cross_product.norm();
        Eigen::Vector3f cross_norm = (cross_product / cross_product_norm);
        // Step 2: Find angle (theta)
        float dot_product = origin_coeff.dot(desire_coeff);
        float norm_origin = origin_coeff.norm();
        float norm_desire = desire_coeff.norm();
        float dot_product_of_norms = norm_origin * norm_desire;
        float dot_product_divided_by_dot_product_of_norms = (dot_product / dot_product_of_norms);
        float theta_angle_rad = acos(dot_product_divided_by_dot_product_of_norms);
        // Step 3: Construct A, the skew-symmetric matrix corresponding to X
        Eigen::Matrix3f matrix_A = Eigen::Matrix3f::Identity();
        matrix_A(0,0) = 0.0;
        matrix_A(0,1) = -1.0 * (cross_norm(2));
        matrix_A(0,2) = cross_norm(1);
        matrix_A(1,0) = cross_norm(2);
        matrix_A(1,1) = 0.0;
        matrix_A(1,2) = -1.0 * (cross_norm(0));
        matrix_A(2,0) = -1.0 * (cross_norm(1));
        matrix_A(2,1) = cross_norm(0);
        matrix_A(2,2) = 0.0;
        // Step 4: Plug and chug.
        Eigen::Matrix3f IdentityMat = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f firstTerm = sin(theta_angle_rad) * matrix_A;
        Eigen::Matrix3f secondTerm = (1.0 - cos(theta_angle_rad)) * matrix_A * matrix_A;
        // This is the rotation matrix. Finished with the Rodrigues' Rotation Formula implementation.
        Eigen::Matrix3f matrix_R = IdentityMat + firstTerm + secondTerm;
        // We copy the rotation matrix into the matrix that will be used for the transformation.
        Transform(0,0) = matrix_R(0,0);
        Transform(0,1) = matrix_R(0,1);
        Transform(0,2) = matrix_R(0,2);
        Transform(1,0) = matrix_R(1,0);
        Transform(1,1) = matrix_R(1,1);
        Transform(1,2) = matrix_R(1,2);
        Transform(2,0) = matrix_R(2,0);
        Transform(2,1) = matrix_R(2,1);
        Transform(2,2) = matrix_R(2,2);

        // Now that we have the rotation matrix, we can use it to also find the translation to move the cloud to the origin.
        // First, rotate a point of interest to the new location.
        modelVectorAxisPoint[0] = centroid [0];
        modelVectorAxisPoint[1] = centroid [1];
        modelVectorAxisPoint[2] = centroid [2];
        Eigen::Vector3f modelVectorAxisPointTransformed =  matrix_R * modelVectorAxisPoint;
        // Add the translation to the matrix.
        Transform(0,3) = modelVectorAxisPointTransformed(0) * (-1.0);
        Transform(1,3) = modelVectorAxisPointTransformed(1) * (-1.0);
        Transform(2,3) = modelVectorAxisPointTransformed(2) * (-1.0);

        std::cout<<"MATRIX IS "<<Transform<<endl;

        // rotate it 90 degrees align with Daniel's
        Eigen::Matrix4f Transform2;
        Transform2<<  0, 1, 0,    0,
            -1,  0,   0,   0,
            0,  0,  1,    0,
            0,  0,   0,    1;
        Transform=Transform2*Transform;

        std::cout<<"MATRIX IS "<<Transform<<endl;
        //========================get the range of the desk==========================
        pcl::transformPointCloud (*cloud_filtered, *cloud_filtered, Transform);
        pcl::transformPointCloud (*cloud, *cloud, Transform);

        pcl::getMinMax3D( *cloud_filtered,cluster_indices_desk->indices, min_pt,max_pt);
        std::cout<<"The range of the desk is "<<min_pt<<endl<<max_pt<<endl;

        float Xcorner=max_pt[0], Ycorner=max_pt[1], Zcorner=0.0;
        TransformPoint(Xcorner, Ycorner, Zcorner);
        cv::Point2f corner1;

        corner1.y=(Ycorner/(float)(FY*(Zcorner))) + CY;
        corner1.x =  (Xcorner/(FX*(Zcorner))) + CX;
        corner.push_back(corner1);

        Xcorner=max_pt[0], Ycorner=min_pt[1], Zcorner=0.0;
        TransformPoint(Xcorner, Ycorner, Zcorner);
        corner1.y=(Ycorner/(float)(FY*(Zcorner))) + CY;
        corner1.x =  (Xcorner/(FX*(Zcorner))) + CX;
        corner.push_back(corner1);

        Xcorner=min_pt[0], Ycorner=max_pt[1], Zcorner=0.0;
        TransformPoint(Xcorner, Ycorner, Zcorner);
        corner1.y=(Ycorner/(float)(FY*(Zcorner))) + CY;
        corner1.x =  (Xcorner/(FX*(Zcorner))) + CX;
        corner.push_back(corner1);

        Xcorner=min_pt[0], Ycorner=min_pt[1], Zcorner=0.0;
        TransformPoint(Xcorner, Ycorner, Zcorner);
        corner1.y=(Ycorner/(float)(FY*(Zcorner))) + CY;
        corner1.x =  (Xcorner/(FX*(Zcorner))) + CX;
        corner.push_back(corner1);

        Center_change = 0; //just a index to show it's changed

        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud (cloud);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.3);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("x");
        pass.setFilterLimits (min_pt[0]-0.05,max_pt[0]+0.05);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);


        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (min_pt[1]-0.05,max_pt[1]+0.05);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.25);
        pass.filter (*cloud_desk2);

        *cloud_b=*cloud_desk2;
        *cloud_sum=*cloud_desk2;   // used to add up some frames together

        std::cout<<"desk in this frame has "<<cloud_desk2->points.size()<<"Points."<<endl;

        octree->setInputCloud(cloud_b); // assign point cloud to octree
        octree->addPointsFromInputCloud(); // add points from cloud to octree
        octree->switchBuffers ();

        std::cout<<"the 1st frame "<<endl;
    }

    // old method of building the model, going to be changed

    else if(callback_counter<=11 && callback_counter>=2)
    {
        s_mode="MODE: Initial";
        s_mode2="MODE: Initial";

        //==================================================================
        pcl::transformPointCloud (*cloud, *cloud_filtered, Transform);

        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud (cloud_filtered);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.2);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("x");
        pass.setFilterLimits (min_pt[0],max_pt[0]);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (min_pt[1],max_pt[1]);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.5);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk2);
        //==================================================================

        std::cout<<"The frame "<<callback_counter<<endl;
        //=====================spatial change detection==========================
        std::cout<<"desk in this frame has "<<cloud_desk2->points.size()<<"Points."<<endl;
        *cloud_sum=*cloud_sum+*cloud_desk2;

        // Add points from cloudB to octree
        octree->setInputCloud (cloud_desk2);
        octree->addPointsFromInputCloud ();

        // Get vector of point indices from octree voxels which did not exist in previous buffer
        octree->getPointIndicesFromNewVoxels (newPointIdxVector);
        pcl::PointIndices::Ptr difindices(new pcl::PointIndices());

        // Output points
        difindices->indices = newPointIdxVector;

        // Extract the inliers
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud (cloud_desk2);
        extract.setIndices (difindices);
        extract.setNegative (false);
        extract.filter (*cloud_p);  // extract the changed part

        std::cerr << "PointCloud changed has so far: " << cloud_p->width * cloud_p->height << " data points." << std::endl;

        octqueue.push_back(cloud_p->points.size());
        extract.setNegative (true);
        extract.filter (*cloud_b);  //unchanged part
        octqueue2.push_back(0);

        std::cerr << "PointCloud model has so far: " << cloud_b->width * cloud_b->height << " data points." << std::endl;

        //octree->deleteTree ();
        octree->deleteCurrentBuffer ();
        newPointIdxVector.clear();
        //octree->deletePreviousBuffer ();  //how to totally reset??????
        octree->setInputCloud (cloud_b);
        octree->addPointsFromInputCloud ();
        octree->switchBuffers ();

        if (callback_counter==11) {
            std::cout<<"the frame for model"<<endl;
            *cloud_model=*cloud_b;
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            seg.setOptimizeCoefficients (true);
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setMaxIterations (100);
            seg.setDistanceThreshold (0.02);


            // Segment the largest planar component from the remaining cloud
            seg.setInputCloud (cloud_desk2);
            //seg.setInputCloud (cloud_sum);
            seg.segment (*inliers, *coefficients);
            if (inliers->indices.size () == 0) {
                std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            }

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<pcl::PointXYZRGB> extract2;
            extract2.setInputCloud (cloud_desk2);
            extract2.setIndices (inliers);
            extract2.setNegative (false);

            // Get the points associated with the planar surface
            extract2.filter (*cloud_plane);
            std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

            // Remove the planar inliers, extract the rest
            extract2.setNegative (true);
            extract2.filter (*cloud_f);

            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZRGB>);
            tree2->setInputCloud (cloud_f);

            std::vector<pcl::PointIndices> cluster_indices2;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
            ec.setClusterTolerance (0.01); // 1cm
            ec.setMinClusterSize (100);
            ec.setMaxClusterSize (50000);
            ec.setSearchMethod (tree2);
            ec.setInputCloud (cloud_f);
            ec.extract (cluster_indices2);

            s_cluster_c.str("");
            s_cluster_c.clear();

            s_cluster_o.str("");
            s_cluster_o.clear();

            object_counter=0;
            int j = 0;
            s_cluster_o<<".";

            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices2.begin (); it != cluster_indices2.end (); ++it) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
                for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
                    cloud_cluster->points.push_back (cloud_f->points[*pit]);
                }

                pcl::compute3DCentroid( *cloud_cluster, centroid);
                if(centroid[2]>0) {
                    desk_object new_obj;
                    new_obj.obj_centroid=centroid;
                    new_obj.obj_index=j+1;
                    new_obj.obj_state=1;
                    new_obj.obj_onobj=0;
                    //------------------bbox------------------------
                    pcl::getMinMax3D( *cloud_cluster,new_obj.obj_minpt,new_obj.obj_maxpt);

                    std::cout<<"xmax is: "<<new_obj.obj_maxpt.x-new_obj.obj_minpt.x<<endl;
                    std::cout<<"is "<<new_obj.obj_minpt.x<<" and "<<new_obj.obj_maxpt.x<<endl;
                    //----------------------------------------------

                    //------------------oriented bbox---------------
                    pcl::ModelCoefficients::Ptr coefficients_plane_obj (new pcl::ModelCoefficients);
                    pcl::PointIndices::Ptr inliers_plane_obj (new pcl::PointIndices);
                    pcl::ExtractIndices<pcl::PointXYZRGB> extract_obj;
                    pcl::SACSegmentation<pcl::PointXYZRGB> seg_obj;
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_temp (new pcl::PointCloud<pcl::PointXYZRGB>);
                    float coez=1;
                    seg_obj.setOptimizeCoefficients (true);
                    seg_obj.setModelType (pcl::SACMODEL_PLANE);
                    seg_obj.setMethodType (pcl::SAC_RANSAC);
                    seg_obj.setMaxIterations (100);
                    seg_obj.setDistanceThreshold (0.005);

                    *cloud_cluster_temp=*cloud_cluster;

                    while(1) {
                        seg_obj.setInputCloud (cloud_cluster_temp);
                        seg_obj.segment (*inliers_plane_obj, *coefficients_plane_obj);
                        coez=coefficients_plane_obj->values[2];
                        if(abs(coez>0.5)) {
                            extract_obj.setInputCloud (cloud_cluster_temp);
                            extract_obj.setIndices (inliers_plane_obj);
                            extract_obj.setNegative (false);
                            extract_obj.filter (*cloud_plane_obj);
                            extract_obj.setNegative (true);
                            extract_obj.filter (*cloud_f_obj);
                            *cloud_cluster_temp = *cloud_f_obj;
                        } else {
                            break;
                        }
                    }

                    if(coefficients_plane_obj->values[0]>0) {
                        coefficients_plane_obj->values[0]=-coefficients_plane_obj->values[0];
                        coefficients_plane_obj->values[1]=-coefficients_plane_obj->values[1];
                    }

                    float plane_tan;
                    plane_tan=coefficients_plane_obj->values[0]/coefficients_plane_obj->values[1];
                    float plane_atan;
                    plane_atan=atan(plane_tan);

                    Eigen::Affine3f transform_obj = Eigen::Affine3f::Identity();
                    transform_obj.translation() << 0.0, 0.0, 0.0;

                    // The same rotation matrix as before; tetha radians arround Z axis
                    transform_obj.rotate (Eigen::AngleAxisf (plane_atan, Eigen::Vector3f::UnitZ()));

                    // Executing the transformation
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_obj (new pcl::PointCloud<pcl::PointXYZRGB> ());
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud2_obj (new pcl::PointCloud<pcl::PointXYZRGB> ());
                    // You can either apply transform_1 or transform_2; they are the same
                    pcl::transformPointCloud (*cloud_cluster, *transformed_cloud_obj, transform_obj);
                    pcl::getMinMax3D( *transformed_cloud_obj,bb_minpt,bb_maxpt);

                    bb_minpt.z=0;
                    float dx=bb_maxpt.x-bb_minpt.x;
                    float dy=bb_maxpt.y-bb_minpt.y;
                    float dz=bb_maxpt.z-bb_minpt.z;

                    //===================================================================================

                    float a;
                    a=transform_obj.matrix()(0,1);
                    transform_obj.matrix()(0,1)=transform_obj.matrix()(1,0);
                    transform_obj.matrix()(1,0)=a;

                    Eigen::Quaternionf qrotation;
                    qrotation=transform_obj.rotation();
                    new_obj.obj_qr=qrotation;
                    plane_tan=1/plane_tan;

                    plane_atan=180*(atan(plane_tan)/3.1415);
                    new_obj.obj_angle=plane_atan;
                    pcl::transformPointCloud (*transformed_cloud_obj,*transformed_cloud2_obj, transform_obj);
                    new_obj.obj_ecke.width  = 8;
                    new_obj.obj_ecke.height = 1;
                    new_obj.obj_ecke.points.resize (new_obj.obj_ecke.width * new_obj.obj_ecke.height);

                    // Generate the data
                    // Set a few outliers
                    new_obj.obj_ecke.points[0] = bb_minpt;
                    new_obj.obj_ecke.points[1] = bb_maxpt;

                    new_obj.obj_ecke.points[2] = bb_minpt;
                    new_obj.obj_ecke.points[2].x = bb_minpt.x+dx;


                    new_obj.obj_ecke.points[3] = bb_minpt;
                    new_obj.obj_ecke.points[3].y = bb_minpt.y+dy;


                    new_obj.obj_ecke.points[4] = bb_minpt;
                    new_obj.obj_ecke.points[4].x = bb_minpt.x+dx;
                    new_obj.obj_ecke.points[4].y = bb_minpt.y+dy;


                    new_obj.obj_ecke.points[5] = bb_minpt;
                    new_obj.obj_ecke.points[5].z = bb_minpt.z+dz;

                    new_obj.obj_ecke.points[6] = bb_minpt;
                    new_obj.obj_ecke.points[6].z = bb_minpt.z+dz;
                    new_obj.obj_ecke.points[6].x = bb_minpt.x+dx;


                    new_obj.obj_ecke.points[7] = bb_minpt;
                    new_obj.obj_ecke.points[7].z = bb_minpt.z+dz;
                    new_obj.obj_ecke.points[7].y = bb_minpt.y+dy;

                    pcl::transformPointCloud (new_obj.obj_ecke,new_obj.obj_ecke, transform_obj);

                    new_obj.obj_centroid[0]=(new_obj.obj_ecke.points[0].x+ new_obj.obj_ecke.points[1].x)/2;
                    new_obj.obj_centroid[1]=(new_obj.obj_ecke.points[0].y+ new_obj.obj_ecke.points[1].y)/2;
                    new_obj.obj_centroid[2]=dz/2;

                    new_obj.obj_p1.x = new_obj.obj_centroid[0];
                    new_obj.obj_p1.y = new_obj.obj_centroid[1];
                    new_obj.obj_p1.z = new_obj.obj_centroid[2];
                    new_obj.obj_dx=dx;
                    new_obj.obj_dy=dy;

                    if(dx>dy) {
                        new_obj.obj_p2.x=new_obj.obj_p1.x+0.3*coefficients_plane_obj->values[0];
                        new_obj.obj_p2.y=new_obj.obj_p1.y+0.3*coefficients_plane_obj->values[1];
                        new_obj.obj_p2.z=new_obj.obj_p1.z;
                    } else {
                        new_obj.obj_p2.z=new_obj.obj_p1.z;
                        if(new_obj.obj_angle>0) {
                            new_obj.obj_angle=new_obj.obj_angle-90;
                            new_obj.obj_p2.x=new_obj.obj_p1.x+0.3*coefficients_plane_obj->values[1];
                            new_obj.obj_p2.y=new_obj.obj_p1.y-0.3*coefficients_plane_obj->values[0];
                        } else {
                            new_obj.obj_angle=new_obj.obj_angle+90;
                            new_obj.obj_p2.x=new_obj.obj_p1.x-0.3*coefficients_plane_obj->values[1];
                            new_obj.obj_p2.y=new_obj.obj_p1.y+0.3*coefficients_plane_obj->values[0];
                        }
                    }
                    object_counter++;
                    vec_obj.push_back(new_obj);
                    std::cout<<"Origin object NO. "<<vec_obj[j].obj_index<< "is pushed back, with centroid of "<<centroid[0]<<" "<<centroid[1]<<" "<<centroid[2]<<std::endl;

                    s_cluster_c.str("");
                    s_cluster_c.clear();
                    s_cluster_c<<"Object "<<j+1<<" = "<<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)<<"( "<< new_obj.obj_centroid[0]<<setw(9)
                        <<setiosflags(ios::fixed)<<setprecision(3)<< new_obj.obj_centroid[1]<<setw(9)<<setiosflags(ios::fixed)
                        <<setprecision(3)<< new_obj.obj_centroid[2]<<" )"<<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)<<new_obj.obj_angle<<" degrees"<<std::endl;

                    v_cluster_c.push_back(s_cluster_c.str());

                    p1.x = new_obj.obj_centroid[0];
                    p1.y = new_obj.obj_centroid[1];
                    p1.z = new_obj.obj_centroid[2];

                    v_p1.push_back(p1);

                    p2.x=new_obj.obj_p2.x;
                    p2.y=new_obj.obj_p2.y;
                    p2.z=new_obj.obj_p2.z;
                    v_p2.push_back(p2);

                    v_qr.push_back(new_obj.obj_qr);
                    Eigen::Vector3f vtranslation;
                    vtranslation<<p1.x,p1.y,p1.z;
                    v_tr.push_back(vtranslation);

                    std::cout<<"p.x is: "<<new_obj.obj_ecke.points[2].x
                        <<" and "<<new_obj.obj_ecke.points[0].x<<endl;

                    v_dx.push_back(dx);
                    v_dy.push_back(dy);
                    v_dz.push_back(dz);

                    p1.x = 0;
                    p1.y = 0;
                    p1.z = 0;

                    cloud_cluster->width = cloud_cluster->points.size ();
                    cloud_cluster->height = 1;
                    cloud_cluster->is_dense = true;

                    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
                    j++;
                }
            }

            s_cluster.str("");
            s_cluster.clear();
            s_cluster<<"Number of objects is: "<<vec_obj.size();
            //==============================================================================
            b_mode=0;
            cloud_sum = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        }

    } else if(callback_counter>=12) {
        pcl::transformPointCloud (*cloud, *cloud_filtered, Transform);
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud (cloud_filtered);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.5);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("x");
        pass.setFilterLimits (min_pt[0],max_pt[0]);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("y");
        pass.setFilterLimits (min_pt[1],max_pt[1]);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk);

        pass.setInputCloud (cloud_desk);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (min_pt[2],max_pt[2]+0.5);
        //pass.setFilterLimitsNegative (true);
        pass.filter (*cloud_desk2);
        //==================================================================

        std::cout<<"The frame "<<callback_counter<<endl;
        octree->deleteCurrentBuffer ();
        newPointIdxVector.clear();

        octree->setInputCloud(cloud_model); // assign point cloud to octree. can be modified??
        octree->addPointsFromInputCloud(); // add points from cloud to octree
        octree->switchBuffers ();

        // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
        octree->setInputCloud (cloud_desk2);
        octree->addPointsFromInputCloud ();

        // Get vector of point indices from octree voxels which did not exist in previous buffer
        octree->getPointIndicesFromNewVoxels (newPointIdxVector);
        pcl::PointIndices::Ptr difindices (new pcl::PointIndices ());

        // Output points
        difindices->indices = newPointIdxVector;

        // Extract the inliers
        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud (cloud_desk2);
        extract.setIndices (difindices);
        extract.setNegative (false);
        extract.filter (*cloud_p);  // extract the changed part

        octqueue.erase(octqueue.begin());
        octqueue.push_back(cloud_p->points.size());

        //Mean and SD ==========================================================
        double sum = std::accumulate(octqueue.begin(), octqueue.end(), 0.0);
        double mean =  sum / octqueue.size(); //Average

        double accum  = 0.0;
        std::for_each (octqueue.begin(), octqueue.end(), [&](const double d) {
                accum  += (d-mean)*(d-mean);
                });

        double stdev = sqrt(accum/(octqueue.size()-1)); //SD
        //=====================================================================

        extract.setNegative (true);
        extract.filter (*cloud_b);  //unchanged part

        int missingpoints=abs(cloud_model->points.size()-cloud_b->points.size());
        octqueue2.erase(octqueue2.begin());
        octqueue2.push_back(missingpoints);

        //Mean and SD ==========================================================

        double sum2 = std::accumulate(octqueue2.begin(), octqueue2.end(), 0.0);
        double mean2 =  sum2 / octqueue2.size(); //average

        double accum2  = 0.0;
        std::for_each (octqueue2.begin(), octqueue2.end(), [&](const double d) {
                accum2  += (d-mean2)*(d-mean2);
                });

        double stdev2 = sqrt(accum2/(octqueue2.size()-1)); //SD
        int modelsize=cloud_model->points.size();
        float modelratio=1.0*missingpoints/modelsize;
        float modelratio2=1.0*cloud_p->points.size()/modelsize;
        std::cout<<"STD 1 AND 2 are   "<<stdev<<"   "<<stdev2<<endl;
        if((modelratio+modelratio2)>0.3 || bIsModuleRunning==false) {
            s_mode="MODE: Covered";   // 20% of the points changed, COVERED!
        } else {
            s_mode="MODE: Normal";
        }

        if(s_mode=="MODE: Normal") { // not covered
            if(b_mode==0) {  // not just after changed
                if((stdev<200&&(stdev2<200))||(callback_counter-frame_counter<11)) {    // stable or just changed
                    s_mode2="Stable";
                } else {
                    b_mode=1;
                    s_mode2="Changing...";
                }     // not stable and more than 11 frames after change, mark it
            } else if(b_mode==1) {    //  the mode just changed

                //======================wait until it's stable. then the following========================
                if(stdev<200&&(stdev2<200)) {    // it's stable
                    *cloud_model=*cloud_desk2;
                    frame_counter=callback_counter;
                    for(unsigned int ih=0;ih<vec_obj.size();ih++) {
                        vec_obj[ih].obj_state=0;
                    }
                    //=======================store the clustering info of the model========================
                    // wait until it's stable. then the following.
                    // Create the segmentation object for the planar model and set all the parameters
                    pcl::SACSegmentation<pcl::PointXYZRGB> seg;

                    seg.setOptimizeCoefficients (true);
                    seg.setModelType (pcl::SACMODEL_PLANE);
                    seg.setMethodType (pcl::SAC_RANSAC);
                    seg.setMaxIterations (100);
                    seg.setDistanceThreshold (0.02);

                    // Segment the largest planar component from the remaining cloud
                    seg.setInputCloud (cloud_desk2);
                    seg.segment (*inliers, *coefficients);
                    if (inliers->indices.size () == 0) {
                        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                    }
                    // Extract the planar inliers from the input cloud
                    pcl::ExtractIndices<pcl::PointXYZRGB> extract2;
                    extract2.setInputCloud (cloud_desk2);
                    extract2.setIndices (inliers);
                    extract2.setNegative (false);

                    // Get the points associated with the planar surface
                    extract2.filter (*cloud_plane);
                    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

                    // Remove the planar inliers, extract the rest
                    extract2.setNegative (true);
                    extract2.filter (*cloud_f);

                    // Creating the KdTree object for the search method of the extraction

                    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
                    tree->setInputCloud (cloud_f);

                    std::vector<pcl::PointIndices> cluster_indices;
                    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
                    ec.setClusterTolerance (0.01); // 0.5cm
                    ec.setMinClusterSize (100);
                    ec.setMaxClusterSize (50000);
                    ec.setSearchMethod (tree);
                    ec.setInputCloud (cloud_f);
                    ec.extract (cluster_indices);

                    s_cluster_c.str("");
                    s_cluster_c.clear();

                    s_cluster_o.str("");
                    s_cluster_o.clear();
                    s_cluster_o<<".";

                    int j = 0;
                    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
                        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
                            cloud_cluster->points.push_back (cloud_f->points[*pit]);
                        }

                        desk_object new_obj_temp;
                        pcl::getMinMax3D( *cloud_cluster,new_obj_temp.obj_minpt,new_obj_temp.obj_maxpt);

                        std::cout<<"new_obj_temp.obj_maxpt.x-new_obj_temp.obj_minpt.x is"<<(new_obj_temp.obj_maxpt.x-new_obj_temp.obj_minpt.x)<<endl;

                        if(((new_obj_temp.obj_maxpt.x-new_obj_temp.obj_minpt.x)>0.30)||((new_obj_temp.obj_maxpt.y-new_obj_temp.obj_minpt.y)>0.30)||(new_obj_temp.obj_maxpt.x>0.45) ||(new_obj_temp.obj_minpt.y<-0.45)||(new_obj_temp.obj_minpt.x<-0.45)||(new_obj_temp.obj_maxpt.y>0.45)) {
                            break;
                        }

                        pcl::compute3DCentroid( *cloud_cluster, centroid);
                        bool test_obj=false;

                        cout<<"///////////////pt///////////////"<<endl;
                        cout<<new_obj_temp.obj_minpt.x<<" "<<new_obj_temp.obj_maxpt.x<<endl;
                        cout<<new_obj_temp.obj_minpt.y<<" "<<new_obj_temp.obj_maxpt.y<<endl;
                        cout<<new_obj_temp.obj_maxpt.z<<endl;
                        cout<<"//////////////////////////////////////"<<endl;

                        pcl::ModelCoefficients::Ptr coefficients_plane_obj (new pcl::ModelCoefficients);
                        pcl::PointIndices::Ptr inliers_plane_obj (new pcl::PointIndices);
                        pcl::ExtractIndices<pcl::PointXYZRGB> extract_obj;
                        pcl::SACSegmentation<pcl::PointXYZRGB> seg_obj;
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_temp (new pcl::PointCloud<pcl::PointXYZRGB>);

                        float coez=1;

                        seg_obj.setOptimizeCoefficients (true);
                        seg_obj.setModelType (pcl::SACMODEL_PLANE);
                        seg_obj.setMethodType (pcl::SAC_RANSAC);
                        seg_obj.setMaxIterations (100);
                        seg_obj.setDistanceThreshold (0.005);

                        *cloud_cluster_temp=*cloud_cluster;

                        for(unsigned int aloop=0;aloop<5;aloop++) {
                            seg_obj.setInputCloud (cloud_cluster_temp);
                            seg_obj.segment (*inliers_plane_obj, *coefficients_plane_obj);
                            coez=coefficients_plane_obj->values[2];
                            if (inliers_plane_obj->indices.size () == 0) {
                                std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                                break;
                            } else {
                                if(abs(coez>0.5)) {

                                    extract_obj.setInputCloud (cloud_cluster_temp);
                                    extract_obj.setIndices (inliers_plane_obj);
                                    extract_obj.setNegative (false);
                                    extract_obj.filter (*cloud_plane_obj);
                                    extract_obj.setNegative (true);
                                    extract_obj.filter (*cloud_f_obj);
                                    *cloud_cluster_temp = *cloud_f_obj;
                                } else {
                                    break;
                                }
                            }
                        }
                        if(coefficients_plane_obj->values[0]>0) {
                            coefficients_plane_obj->values[0]=-coefficients_plane_obj->values[0];
                            coefficients_plane_obj->values[1]=-coefficients_plane_obj->values[1];
                        }
                        float plane_tan;
                        plane_tan=coefficients_plane_obj->values[0]/coefficients_plane_obj->values[1];
                        float plane_atan;
                        plane_atan=atan(plane_tan);
                        plane_tan=1/plane_tan;
                        plane_atan=180*(atan(plane_tan)/3.1415);

                        new_obj_temp.obj_angle=plane_atan;
                        cout<<new_obj_temp.obj_angle<<endl;
                        cout<<"//////////////////////////////////////"<<endl;

                        for(unsigned int ih=0;ih<vec_obj.size();ih++) {

                            cout<<"~~~~~~~~~~~~~pt~~~~~~~~~~~~~"<<endl;
                            cout<<vec_obj[ih].obj_minpt.x<<" "<<vec_obj[ih].obj_maxpt.x<<endl;
                            cout<<vec_obj[ih].obj_minpt.y<<" "<<vec_obj[ih].obj_maxpt.y<<endl;
                            cout<<vec_obj[ih].obj_maxpt.z<<endl;
                            cout<<vec_obj[ih].obj_angle<<endl;
                            cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;

                            if(fabs(new_obj_temp.obj_minpt.x-vec_obj[ih].obj_minpt.x)<0.02&&fabs(new_obj_temp.obj_minpt.y-vec_obj[ih].obj_minpt.y)<0.02
                                    &&fabs(new_obj_temp.obj_maxpt.x-vec_obj[ih].obj_maxpt.x)<0.05&&fabs(new_obj_temp.obj_maxpt.y-vec_obj[ih].obj_maxpt.y)<0.02
                                    &&fabs(new_obj_temp.obj_maxpt.z-vec_obj[ih].obj_maxpt.z)<0.02&&fabs(new_obj_temp.obj_angle-vec_obj[ih].obj_angle)<10)
                            {
                                vec_obj[ih].obj_state=1;
                                std::cout<<"The object still exists: obj "<<vec_obj[ih].obj_index<<endl;

                                if(vec_obj[ih].obj_onobj!=0) {
                                    int c=vec_obj[ih].obj_onobj;
                                    for(unsigned int it=0;it<vec_obj.size();it++) {
                                        if (vec_obj[it].obj_index==c) {
                                            vec_obj[it].obj_state=1;
                                        }
                                    }
                                }

                                test_obj=true;
                                break;
                            }
                        }
                        desk_object new_obj;

                        if(test_obj==false&&centroid[2]>0) {
                            std::cout<<"The test_obj is falllllllllllllse!! "<<endl;

                            new_obj.obj_centroid=centroid;
                            new_obj.obj_index=++object_counter;
                            new_obj.obj_state=1;
                            pcl::getMinMax3D( *cloud_cluster,new_obj.obj_minpt,new_obj.obj_maxpt);

                            for(unsigned int ih=0;ih<vec_obj.size();ih++)            //to judge whether an object is on another
                            {
                                std::cout<<"minptx, minpty, maxx, maxy, centroid "
                                    << new_obj.obj_minpt.x<<" "<<(vec_obj[ih].obj_minpt.x)<<endl
                                    <<new_obj.obj_minpt.y<<" "<<(vec_obj[ih].obj_minpt.y)<<endl
                                    <<new_obj.obj_maxpt.x<<" "<<(vec_obj[ih].obj_maxpt.x)<<endl
                                    <<new_obj.obj_maxpt.y<<" "<<(vec_obj[ih].obj_maxpt.y)<<endl
                                    <<centroid[2]<<" "<<vec_obj[ih].obj_centroid[2]<<endl;

                                if(new_obj.obj_minpt.x<=(vec_obj[ih].obj_minpt.x+0.02)&&new_obj.obj_minpt.y<=(vec_obj[ih].obj_minpt.y+0.02)
                                        &&new_obj.obj_maxpt.y>=(vec_obj[ih].obj_maxpt.y-0.02)
                                        &&(centroid[2]-vec_obj[ih].obj_centroid[2])>0.01)
                                {
                                    vec_obj[ih].obj_state=1;
                                    std::cout<<"The object is covered by another "<<vec_obj[ih].obj_index<<endl;
                                    new_obj.obj_onobj=vec_obj[ih].obj_index;

                                    pcl::PassThrough<pcl::PointXYZRGB> pass2;
                                    pass2.setInputCloud (cloud_cluster);
                                    pass2.setFilterFieldName ("z");
                                    pass2.setFilterLimits (vec_obj[ih].obj_maxpt.z,0.5);
                                    //pass.setFilterLimitsNegative (true);
                                    pass2.filter (*cloud_cluster);

                                    pcl::compute3DCentroid( *cloud_cluster, centroid);

                                    pcl::getMinMax3D( *cloud_cluster,new_obj.obj_minpt,new_obj.obj_maxpt);
                                }
                            }
                        }

                        if(test_obj==false&&cloud_cluster->size()>=20) {
                            //------------------oriented bbox---------------
                            pcl::ModelCoefficients::Ptr coefficients_plane_obj (new pcl::ModelCoefficients);
                            pcl::PointIndices::Ptr inliers_plane_obj (new pcl::PointIndices);
                            pcl::ExtractIndices<pcl::PointXYZRGB> extract_obj;
                            pcl::SACSegmentation<pcl::PointXYZRGB> seg_obj;
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane_obj (new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_temp (new pcl::PointCloud<pcl::PointXYZRGB>);

                            float coez=1;
                            seg_obj.setOptimizeCoefficients (true);
                            seg_obj.setModelType (pcl::SACMODEL_PLANE);
                            seg_obj.setMethodType (pcl::SAC_RANSAC);
                            seg_obj.setMaxIterations (100);
                            seg_obj.setDistanceThreshold (0.005);
                            *cloud_cluster_temp=*cloud_cluster;
                            for(unsigned int aloop=0;aloop<5;aloop++) {
                                seg_obj.setInputCloud (cloud_cluster_temp);
                                seg_obj.segment (*inliers_plane_obj, *coefficients_plane_obj);
                                coez=coefficients_plane_obj->values[2];
                                if (inliers_plane_obj->indices.size () == 0) {
                                    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                                    break;
                                } else {
                                    if(abs(coez>0.5)) {
                                        extract_obj.setInputCloud (cloud_cluster_temp);
                                        extract_obj.setIndices (inliers_plane_obj);
                                        extract_obj.setNegative (false);
                                        extract_obj.filter (*cloud_plane_obj);
                                        extract_obj.setNegative (true);
                                        extract_obj.filter (*cloud_f_obj);
                                        *cloud_cluster_temp = *cloud_f_obj;
                                    } else {
                                        break;
                                    }
                                }
                            }

                            if(coefficients_plane_obj->values[0]>0) {
                                coefficients_plane_obj->values[0]=-coefficients_plane_obj->values[0];
                                coefficients_plane_obj->values[1]=-coefficients_plane_obj->values[1];
                            }

                            float plane_tan;
                            plane_tan=coefficients_plane_obj->values[0]/coefficients_plane_obj->values[1];
                            float plane_atan;
                            plane_atan=atan(plane_tan);

                            Eigen::Affine3f transform_obj = Eigen::Affine3f::Identity();
                            transform_obj.translation() << 0.0, 0.0, 0.0;

                            // The same rotation matrix as before; tetha radians arround Z axis
                            transform_obj.rotate (Eigen::AngleAxisf (plane_atan, Eigen::Vector3f::UnitZ()));

                            // Executing the transformation
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud_obj (new pcl::PointCloud<pcl::PointXYZRGB> ());
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud2_obj (new pcl::PointCloud<pcl::PointXYZRGB> ());
                            // You can either apply transform_1 or transform_2; they are the same
                            pcl::transformPointCloud (*cloud_cluster, *transformed_cloud_obj, transform_obj);
                            pcl::getMinMax3D( *transformed_cloud_obj,bb_minpt,bb_maxpt);

                            if(bb_minpt.z<0.04) {
                                bb_minpt.z=0;
                            }

                            float dx=bb_maxpt.x-bb_minpt.x;
                            float dy=bb_maxpt.y-bb_minpt.y;
                            float dz=bb_maxpt.z-bb_minpt.z;

                            //===================================================================================

                            float a;
                            a=transform_obj.matrix()(0,1);
                            transform_obj.matrix()(0,1)=transform_obj.matrix()(1,0);
                            transform_obj.matrix()(1,0)=a;

                            Eigen::Quaternionf qrotation;
                            qrotation=transform_obj.rotation();
                            new_obj.obj_qr=qrotation;

                            plane_tan=1/plane_tan;
                            plane_atan=180*(atan(plane_tan)/3.1415);

                            new_obj.obj_angle=plane_atan;

                            pcl::transformPointCloud (*transformed_cloud_obj,*transformed_cloud2_obj, transform_obj);

                            new_obj.obj_ecke.width  = 8;
                            new_obj.obj_ecke.height = 1;
                            new_obj.obj_ecke.points.resize (new_obj.obj_ecke.width * new_obj.obj_ecke.height);

                            // Generate the data
                            // Set a few outliers
                            new_obj.obj_ecke.points[0] = bb_minpt;
                            new_obj.obj_ecke.points[1] = bb_maxpt;

                            new_obj.obj_ecke.points[2] = bb_minpt;
                            new_obj.obj_ecke.points[2].x = bb_minpt.x+dx;

                            new_obj.obj_ecke.points[3] = bb_minpt;
                            new_obj.obj_ecke.points[3].y = bb_minpt.y+dy;

                            new_obj.obj_ecke.points[4] = bb_minpt;
                            new_obj.obj_ecke.points[4].x = bb_minpt.x+dx;
                            new_obj.obj_ecke.points[4].y = bb_minpt.y+dy;

                            new_obj.obj_ecke.points[5] = bb_minpt;
                            new_obj.obj_ecke.points[5].z = bb_minpt.z+dz;

                            new_obj.obj_ecke.points[6] = bb_minpt;
                            new_obj.obj_ecke.points[6].z = bb_minpt.z+dz;
                            new_obj.obj_ecke.points[6].x = bb_minpt.x+dx;

                            new_obj.obj_ecke.points[7] = bb_minpt;
                            new_obj.obj_ecke.points[7].z = bb_minpt.z+dz;
                            new_obj.obj_ecke.points[7].y = bb_minpt.y+dy;

                            pcl::transformPointCloud (new_obj.obj_ecke,new_obj.obj_ecke, transform_obj);

                            new_obj.obj_centroid[0]=(new_obj.obj_ecke.points[0].x+ new_obj.obj_ecke.points[1].x)/2;
                            new_obj.obj_centroid[1]=(new_obj.obj_ecke.points[0].y+ new_obj.obj_ecke.points[1].y)/2;
                            new_obj.obj_centroid[2]=(new_obj.obj_ecke.points[0].z+ new_obj.obj_ecke.points[1].z)/2;

                            new_obj.obj_p1.x = new_obj.obj_centroid[0];
                            new_obj.obj_p1.y = new_obj.obj_centroid[1];
                            new_obj.obj_p1.z = new_obj.obj_centroid[2];
                            new_obj.obj_dx=dx;
                            new_obj.obj_dy=dy;

                            if(dx>dy) {
                                new_obj.obj_p2.x=new_obj.obj_p1.x+0.3*coefficients_plane_obj->values[0];
                                new_obj.obj_p2.y=new_obj.obj_p1.y+0.3*coefficients_plane_obj->values[1];
                                new_obj.obj_p2.z=new_obj.obj_p1.z;
                            } else {
                                new_obj.obj_p2.z=new_obj.obj_p1.z;
                                if(new_obj.obj_angle>0) {
                                    new_obj.obj_angle=new_obj.obj_angle-90;
                                    new_obj.obj_p2.x=new_obj.obj_p1.x+0.3*coefficients_plane_obj->values[1];
                                    new_obj.obj_p2.y=new_obj.obj_p1.y-0.3*coefficients_plane_obj->values[0];
                                } else {
                                    new_obj.obj_angle=new_obj.obj_angle+90;
                                    new_obj.obj_p2.x=new_obj.obj_p1.x-0.3*coefficients_plane_obj->values[1];
                                    new_obj.obj_p2.y=new_obj.obj_p1.y+0.3*coefficients_plane_obj->values[0];
                                }
                            }

                            vec_obj.push_back(new_obj);
                            std::cout<<"new obj is pushed back"<<vec_obj.size()<<endl;
                            s_cluster_o<<"New object "<<new_obj.obj_index<<" is placed!"<<endl;
                        }

                        cloud_cluster->width = cloud_cluster->points.size ();
                        cloud_cluster->height = 1;
                        cloud_cluster->is_dense = true;

                        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
                        j++;
                    }
                    for(unsigned int ih=0;ih<vec_obj.size();ih++) {
                        if(vec_obj[ih].obj_state==0) {
                            cout<<"This object is taken away: "<<vec_obj[ih].obj_index<<endl;
                            s_cluster_o<<"Old object "<<vec_obj[ih].obj_index<<" is taken away!"<<endl;
                            vec_obj.erase(vec_obj.begin()+ih);
                            ih--;
                        }
                    }

                    cout<<"No. of objects is "<<vec_obj.size()<<endl;
                    s_cluster.str("");
                    s_cluster.clear();
                    s_cluster<<"Number of objects is: "<<vec_obj.size();

                    v_cluster_c.clear();
                    v_p1.clear();
                    v_p2.clear();

                    v_tr.clear();
                    v_qr.clear();
                    v_dx.clear();
                    v_dy.clear();
                    v_dz.clear();

                    for(unsigned int ih=0;ih<vec_obj.size();ih++) {
                        s_cluster_c.str("");
                        s_cluster_c.clear();
                        s_cluster_c<<"Object "<<vec_obj[ih].obj_index<<" = "<<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)
                            <<"( "<<vec_obj[ih].obj_centroid[0]<<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)<<vec_obj[ih].obj_centroid[1]
                            <<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)<<vec_obj[ih].obj_centroid[2]<<" )"<<setw(9)<<setiosflags(ios::fixed)<<setprecision(3)<<vec_obj[ih].obj_angle<<" degrees"<<std::endl;
                        v_cluster_c.push_back(s_cluster_c.str());

                        v_p1.push_back(vec_obj[ih].obj_p1);
                        v_p2.push_back(vec_obj[ih].obj_p2);

                        v_qr.push_back(vec_obj[ih].obj_qr);
                        Eigen::Vector3f vtranslation;
                        vtranslation<<vec_obj[ih].obj_p1.x,vec_obj[ih].obj_p1.y,vec_obj[ih].obj_p1.z;
                        v_tr.push_back(vtranslation);

                        v_dx.push_back(vec_obj[ih].obj_dx);

                        std::cout<<"dx is: "<<vec_obj[ih].obj_ecke.points[2].x
                            <<" and "<<vec_obj[ih].obj_ecke.points[0].x<<endl;
                        v_dy.push_back(vec_obj[ih].obj_dy);
                        v_dz.push_back(vec_obj[ih].obj_ecke.points[5].z-vec_obj[ih].obj_ecke.points[0].z);

                        p1.x = 0;
                        p1.y = 0;
                        p1.z = 0;
                    }

                    tenframecheck=0;
                    cloud_sum = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
                    b_mode=0;
                }
            }

            else if(s_mode=="MODE: Covered")
            {
                b_mode=1;
                s_mode2=".";
            }
        }
    }
}


void Receiver::cloudViewer()
{
    std::cout<<"CLoudViewer started... callback counter is "<<callback_counter<<endl;
    cv::Mat color, depth;
    pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));

    const std::string cloudName = "rendered";
    const std::string cloudName2 = "rendered2";

    lock.lock();
    color = this->color;
    depth = this->depth;
    updateCloud = false;
    lock.unlock();

    createCloud(depth, color, cloud);
    int v1(0);

    visualizer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
    visualizer->addCoordinateSystem(0.1,v1);
    visualizer->addPointCloud(cloud, cloudName,v1);
    visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
    visualizer->initCameraParameters();
    visualizer->setPosition(mode == BOTH ? color.cols : 0, 0);
    visualizer->setShowFPS(true);
    visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
    visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);

    int v2(0);
    visualizer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);
    visualizer->addPointCloud(cloud, cloudName2,v2);
    visualizer->setBackgroundColor(0.1, 0.1, 0.1,v2);
    visualizer->addText ("Cluster_ini", 10, 10, 20 ,1,1,1,"v1 text", v1);
    visualizer->addText ("Cluster_ini3", 10, 10, 20 ,1,1,1,"v1 text3", v1);
    visualizer->addText ("MODE_ini", 10, 300, 10 ,1,1,1,"v2 text", v2);
    visualizer->addText ("ini", 10, 20, 10 ,1,0,0,"v2 text2", v2);
    visualizer->addCube (bb_minpt.x,bb_maxpt.x,bb_minpt.y,bb_maxpt.y,bb_minpt.z,bb_maxpt.z,1,0,0, "box3", v1);

    struct colorlist {
        double R;
        double G;
        double B;
    };
    std::vector<colorlist> colorvec;

    colorlist acolor;
    acolor.R=1;
    acolor.B=0;
    acolor.G=0;
    colorvec.push_back(acolor);
    acolor.R=0;
    acolor.B=1;
    acolor.G=0;
    colorvec.push_back(acolor);
    acolor.R=0;
    acolor.B=0;
    acolor.G=1;
    colorvec.push_back(acolor);
    acolor.R=1;
    acolor.B=1;
    acolor.G=0;
    colorvec.push_back(acolor);
    acolor.R=0;
    acolor.B=1;
    acolor.G=1;
    colorvec.push_back(acolor);
    acolor.R=1;
    acolor.B=0;
    acolor.G=1;
    colorvec.push_back(acolor);
    acolor.R=1;
    acolor.B=1;
    acolor.G=1;
    colorvec.push_back(acolor);

    std::vector<std::string> s_color;  // the name for text
    std::vector<std::string> s_color2; // the name for sphere
    std::vector<std::string> s_color3; // the name for cubes
    std::vector<std::string> s_color4; // the name for arrows
    std::stringstream s_temp;
    for(int i=0;i<7;i++)
    {
        s_temp.str("");
        s_temp.clear();
        s_temp<<"color"<<i;
        s_color.push_back(s_temp.str());
        s_temp<<"t";
        s_color2.push_back(s_temp.str());
        s_temp<<"c";
        s_color3.push_back(s_temp.str());
        s_temp<<"a";
        s_color4.push_back(s_temp.str());

        visualizer->addText ("Color", 10, 1000, 20 ,1,1,1,s_color[i], v1);
        visualizer->addSphere (p1,0.01,1,0,0, s_color2[i], v1);
        visualizer->addCube (0,0,0,0,0,0,1,0,0, s_color3[i], v1);
        visualizer->addArrow(p1,p1,0,0,0,0,s_color4[i], v1);
    }

    for(; running && ros::ok();) {
        if(updateCloud) {
            std::cout<<"CLoud updated... callback counter is "<<callback_counter<<endl;
            lock.lock();
            color = this->color;
            depth = this->depth;
            updateCloud = false;
            lock.unlock();
            createCloud(depth, color, cloud);
            cloudProcesser(cloud);
            pub_desk.publish(merosy_desk);
            merosy_objvec.objs.clear();

            visualizer->updatePointCloud(cloud_desk, cloudName);
            visualizer->updatePointCloud(cloud_filtered, cloudName2);  //cloud_p

            visualizer->updateText(s_mode,10,10,20,1,1,1,"v2 text");
            visualizer->updateText(s_mode2,10,30,15,1,0,0,"v2 text2");
            visualizer->updateText(s_cluster.str(),10,10,20,1,1,1,"v1 text");

            for(unsigned int i=0;i<7;i++) {
                visualizer->removeShape(s_color3[i],v1);
                visualizer->removeShape(s_color4[i],v1);
            }
            for(unsigned int i=0;i<7;i++) {
                if(i<v_p1.size()) {
                    int colorindex=(vec_obj[i].obj_index-1)%7;
                    visualizer->updateText(v_cluster_c[i],0,140-i*15,15,colorvec[colorindex].R,colorvec[colorindex].G,colorvec[colorindex].B,s_color[i]);

                    visualizer->updateSphere (v_p1[i],0.03,colorvec[colorindex].R,colorvec[colorindex].G,colorvec[colorindex].B, s_color2[i]);
                    visualizer->addCube(v_tr[i],v_qr[i],v_dx[i],v_dy[i],v_dz[i],s_color3[i],v1);
                    visualizer->addArrow(v_p2[i],v_p1[i],colorvec[colorindex].R,colorvec[colorindex].G,colorvec[colorindex].B,0,s_color4[i],v1);

                    merosy_object.id=vec_obj[i].obj_index;
                    merosy_object.center_x=v_p1[i].x;
                    merosy_object.center_y=v_p1[i].y;
                    merosy_object.center_z=v_p1[i].z;
                    merosy_object.max_x=vec_obj[i].obj_maxpt.x;
                    merosy_object.max_y=vec_obj[i].obj_maxpt.y;
                    merosy_object.max_z=vec_obj[i].obj_maxpt.z;
                    merosy_object.min_x=vec_obj[i].obj_minpt.x;
                    merosy_object.min_y=vec_obj[i].obj_minpt.y;
                    merosy_object.min_z=vec_obj[i].obj_minpt.z;// if this is small than 0.01, can be set as 0

                    if(v_dx[i]>v_dy[i]) {
                        merosy_object.length=v_dx[i];
                        merosy_object.width=v_dy[i];
                    } else {
                        merosy_object.length=v_dy[i];
                        merosy_object.width=v_dx[i];
                    }

                    merosy_object.depth=v_dz[i];
                    merosy_object.angle=vec_obj[i].obj_angle;
                    merosy_objvec.objs.push_back(merosy_object);
                }
                if(i>=v_p1.size()) {
                    visualizer->updateText(".",0,140-i*15,15,1,1,1,s_color[i]);
                    visualizer->updateSphere (p1,0.01,1,1,1, s_color2[i]);
                    visualizer->addCube (0,0,0,0,0,0,1,0,0, s_color3[i], v1);
                    visualizer->addArrow(p2,p2,1,1,1,s_color4[i],v1);
                }
            }

            pub_objvec.publish(merosy_objvec);
            visualizer->updateText(s_cluster_o.str(),10,350,20,1,1,0,"v1 text3");
        }
        if(save) {
            save = false;
            cv::Mat depthDisp;
            dispDepth(depth, depthDisp, 12000.0f);
            saveCloudAndImages(cloud, color, depth, depthDisp);
        }
        visualizer->spinOnce(10);
    }
    visualizer->close();
}


void Receiver::keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *)
{
    if(event.keyUp()) {
        switch(event.getKeyCode()) {
            case 27:
            case 'q':
                running = false;
                break;
            case ' ':
            case 's':
                save = true;
                break;
        }
    }
}


void Receiver::readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
{
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
}


void Receiver::readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
{
    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC) {
        *itC = cameraInfo->K[i];
    }
}


void Receiver::dispDepth(const cv::Mat &in, cv::Mat &out, const float maxValue)
{
    cv::Mat tmp = cv::Mat(in.rows, in.cols, CV_8U);
    const uint32_t maxInt = 255;

#pragma omp parallel for
    for(int r = 0; r < in.rows; ++r) {
        const uint16_t *itI = in.ptr<uint16_t>(r);
        uint8_t *itO = tmp.ptr<uint8_t>(r);

        for(int c = 0; c < in.cols; ++c, ++itI, ++itO) {
            *itO = (uint8_t)std::min((*itI * maxInt / maxValue), 255.0f);
        }
    }

    cv::applyColorMap(tmp, out, cv::COLORMAP_JET);
}


void Receiver::combine(const cv::Mat &inC, const cv::Mat &inD, cv::Mat &out)
{
    out = cv::Mat(inC.rows, inC.cols, CV_8UC3);
#pragma omp parallel for
    for(int r = 0; r < inC.rows; ++r) {
        const cv::Vec3b
            *itC = inC.ptr<cv::Vec3b>(r),
            *itD = inD.ptr<cv::Vec3b>(r);
        cv::Vec3b *itO = out.ptr<cv::Vec3b>(r);

        for(int c = 0; c < inC.cols; ++c, ++itC, ++itD, ++itO) {
            itO->val[0] = (itC->val[0] + itD->val[0]) >> 1;
            itO->val[1] = (itC->val[1] + itD->val[1]) >> 1;
            itO->val[2] = (itC->val[2] + itD->val[2]) >> 1;
        }
    }
}


void Receiver::createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) const
{
    const float badPoint = std::numeric_limits<float>::quiet_NaN();
#pragma omp parallel for
    for(int r = 0; r < depth.rows; ++r) {
        pcl::PointXYZRGB *itP = &cloud->points[r * depth.cols];
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>();

        for(size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX) {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if(isnan(depthValue) || depthValue <= 0.001) {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
                itP->rgba = 0;
                continue;
            }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
        }
    }
}


void Receiver::saveCloudAndImages(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
{
    oss.str("");
    oss << "./" << std::setfill('0')  << frame;

    const std::string baseName = oss.str();
    const std::string cloudName = baseName + "_cloud.pcd";
    const std::string colorName = baseName + "_color.jpg";
    const std::string depthName = baseName + "_depth.png";
    const std::string depthColoredName = baseName + "_depth_colored.png";

    std::cout << "saving cloud: " << cloudName << std::endl;
    writer.writeBinary(cloudName, *cloud);
    std::cout << "saving color: " << colorName << std::endl;
    cv::imwrite(colorName, color, params);
    std::cout << "saving depth: " << depthName << std::endl;
    cv::imwrite(depthName, depth, params);
    std::cout << "saving depth: " << depthColoredName << std::endl;
    cv::imwrite(depthColoredName, depthColored, params);
    std::cout << "saving complete!" << std::endl;
    ++frame;
}


void Receiver::createLookup(size_t width, size_t height)
{
    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);

    FX = fx;
    FY = fy;
    CX = cx;
    CY = cy;
    std::cout<<"FXFYCXCY= "<<FX<<" "<<FY<<" "<<CX<<" "<<CY<<endl;
    float *it;
    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it) {
        *it = (r - cy) * fy;
    }
    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it) {
        *it = (c - cx) * fx;
    }
}

/* vim: set ft=cpp ts=4 sw=4 et ai : */
