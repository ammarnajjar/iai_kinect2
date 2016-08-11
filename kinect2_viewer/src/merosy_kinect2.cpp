/**
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include "Receiver.h"


void help(const std::string &path)
{
    std::cout << path << " [options]" << std::endl
        << "  name: 'any string' equals to the kinect2_bridge topic base name" << std::endl
        << "  mode: 'qhd', 'hd', 'sd' or 'ir'" << std::endl
        << "  visualization: 'image', 'cloud' or 'both'" << std::endl
        << "  options:" << std::endl
        << "  'compressed' use compressed instead of raw topics" << std::endl
        << "  'approx' use approximate time synchronization" << std::endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "merosy_kinect2", ros::init_options::AnonymousName);
    if(!ros::ok()) {
        return 1;
    }

    std::string ns = K2_DEFAULT_NS;
    std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
    std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    bool useExact = true;
    bool useCompressed = false;
    Receiver::Mode mode = Receiver::CLOUD;

    for(size_t i = 1; i < (size_t)argc; ++i) {
        std::string param(argv[i]);
        if(param == "-h" || param == "--help" || param == "-?" || param == "--?") {
            help(argv[0]);
            ros::shutdown();
            return 0;
        } else if(param == "qhd") {
            topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
            topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        } else if(param == "hd") {
            topicColor = K2_TOPIC_HD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
            topicDepth = K2_TOPIC_HD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        } else if(param == "ir") {
            topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_IR K2_TOPIC_IMAGE_RECT;
            topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        } else if(param == "sd") {
            topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
            topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
        } else if(param == "approx") {
            useExact = false;
        } else if(param == "compressed") {
            useCompressed = true;
        } else if(param == "image") {
            mode = Receiver::IMAGE;
        } else if(param == "cloud") {
            mode = Receiver::CLOUD;
        } else if(param == "both") {
            mode = Receiver::BOTH;
        } else {
            ns = param;
        }
    }

    topicColor = "/" + ns + topicColor;
    topicDepth = "/" + ns + topicDepth;
    std::cout << "topic color: " << topicColor << std::endl;
    std::cout << "topic depth: " << topicDepth << std::endl;

    Receiver receiver(topicColor, topicDepth, useExact, useCompressed);
    receiver.callback_counter = 0;
    receiver.Transform = Eigen::Matrix4f::Identity();
    std::cout << "starting receiver..." << std::endl;
    receiver.run(mode);
    ros::shutdown();
    return 0;
}

/* vim: set ft=cpp ts=4 sw=4 et ai : */
