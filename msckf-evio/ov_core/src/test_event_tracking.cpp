/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <deque>
#include <sstream>
#include <unistd.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cam/CamRadtan.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

using namespace ov_core;

// Our feature extractor
TrackBase *extractor;

// FPS counter, and other statistics
// https://gamedev.stackexchange.com/a/83174
int frames = 0;
int num_lostfeats = 0;
int num_margfeats = 0;
int featslengths = 0;
int clone_states = 10;
std::deque<double> clonetimes;
ros::Time time_start;

// How many cameras we will do visual tracking on (mono=1, stereo=2)
int max_cameras = 1;

// Our master function for tracking
void handle_stereo(double time0, double time1, cv::Mat img0, cv::Mat img1);

void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

// Main function
int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path.txt";
  if (argc > 1) {
    config_path = argv[1];
  }

  // Initialize this as a ROS node
  ros::init(argc, argv, "test_tracking");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);

  // Load parameters
  auto parser = std::make_shared<ov_core::YamlParser>(config_path, false);
  parser->set_node_handler(nh);

  // Verbosity
  std::string verbosity = "DEBUG";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Our camera topics (left and right stereo)
  std::string topic_camera0, topic_camera1;
  nh->param<std::string>("topic_camera0", topic_camera0, "/cam0/image_raw");
  nh->param<std::string>("topic_camera1", topic_camera1, "/cam1/image_raw");
  parser->parse_external("relative_config_imucam", "cam" + std::to_string(0), "rostopic", topic_camera0);
  parser->parse_external("relative_config_imucam", "cam" + std::to_string(1), "rostopic", topic_camera1);

  // Location of the ROS bag we want to read in
  std::string path_to_bag;
  nh->param<std::string>("path_bag", path_to_bag, "/home/patrick/datasets/euroc_mav/V1_01_easy.bag");
  // nh->param<std::string>("path_bag", path_to_bag, "/home/patrick/datasets/rpng_aruco/aruco_room_01.bag");
  PRINT_INFO("ros bag path is: %s\n", path_to_bag.c_str());

  // Get our start location and how much of the bag we want to play
  // Make the bag duration < 0 to just process to the end of the bag
  double bag_start, bag_durr;
  nh->param<double>("bag_start", bag_start, 0);
  nh->param<double>("bag_durr", bag_durr, -1);

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(4);

  // Parameters for our extractor
  int num_pts = 600;
  int num_aruco = 1024;
  int fast_threshold = 30;
  int grid_x = 5;
  int grid_y = 5;
  int min_px_dist = 15;
  double knn_ratio = 0.70;
  bool do_downsizing = false;
  bool use_stereo = false;
  parser->parse_config("max_cameras", max_cameras, false);
  parser->parse_config("num_pts", num_pts, false);
  parser->parse_config("num_aruco", num_aruco, false);
  parser->parse_config("clone_states", clone_states, false);
  parser->parse_config("fast_threshold", fast_threshold, false);
  parser->parse_config("grid_x", grid_x, false);
  parser->parse_config("grid_y", grid_y, false);
  parser->parse_config("min_px_dist", min_px_dist, false);
  parser->parse_config("knn_ratio", knn_ratio, false);
  parser->parse_config("do_downsizing", do_downsizing, false);
  parser->parse_config("use_stereo", use_stereo, false);

  // Histogram method
  ov_core::TrackBase::HistogramMethod method;
  std::string histogram_method_str = "NONE";
  parser->parse_config("histogram_method", histogram_method_str, false);
  if (histogram_method_str == "NONE") {
    method = ov_core::TrackBase::NONE;
  } else if (histogram_method_str == "HISTOGRAM") {
    method = ov_core::TrackBase::HISTOGRAM;
  } else if (histogram_method_str == "CLAHE") {
    method = ov_core::TrackBase::CLAHE;
  } else {
    printf(RED "invalid feature histogram specified:\n" RESET);
    printf(RED "\t- NONE\n" RESET);
    printf(RED "\t- HISTOGRAM\n" RESET);
    printf(RED "\t- CLAHE\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Debug print!
  PRINT_DEBUG("max cameras: %d\n", max_cameras);
  PRINT_DEBUG("max features: %d\n", num_pts);
  PRINT_DEBUG("max aruco: %d\n", num_aruco);
  PRINT_DEBUG("clone states: %d\n", clone_states);
  PRINT_DEBUG("grid size: %d x %d\n", grid_x, grid_y);
  PRINT_DEBUG("fast threshold: %d\n", fast_threshold);
  PRINT_DEBUG("min pixel distance: %d\n", min_px_dist);
  PRINT_DEBUG("downsize aruco image: %d\n", do_downsizing);
  PRINT_DEBUG("stereo tracking: %d\n", use_stereo);

  // Fake camera info (we don't need this, as we are not using the normalized coordinates for anything)
  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras;
  for (int i = 0; i < 1; i++) {
    Eigen::Matrix<double, 8, 1> cam0_calib;
    // cam0_calib << 199.092366542, 198.82882047, 132.192071378, 110.712660011, -0.368436311798, 0.150947243557, -0.000296130534385, -0.000759431726241;
    // std::shared_ptr<CamBase> camera_calib = std::make_shared<CamRadtan>(240, 180);
    
    // cam0_calib << 259.355, 259.58, 177.005, 137.922, -0.373464, 0.139924, -0.000309077, 0.000635965; // vicon
    // cam0_calib << 249.69341447817564, 248.41625664694038, 176.74240257052816, 129.47631010746218, -0.3674645003666135, 0.13685246896794986, -0.0004006812242899775, 3.0969134907500055e-05; // hku
    // cam0_calib << 172.98992850734132, 172.98303181090185, 163.33639726024606, 134.99537889030861, -0.027576733308582076, -0.006593578674675004, 0.0008566938165177085, -0.00030899587045247486; // fpv
    // std::shared_ptr<CamBase> camera_calib = std::make_shared<CamRadtan>(346, 260);

    cam0_calib << 591.812, 591.546, 248.422, 254.646, -0.162484, 0.229050, 0, 0;
    std::shared_ptr<CamBase> camera_calib = std::make_shared<CamRadtan>(512, 512);
    camera_calib->set_value(cam0_calib);
    cameras.insert({i, camera_calib});
  }

  // Lets make a feature extractor
  std::vector<size_t> e_ids = {};
  extractor = new TrackKLT(cameras, num_pts, num_aruco, use_stereo, method, fast_threshold, grid_x, grid_y, min_px_dist, e_ids);
  
  // ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/TS_left", 100, imageCallback);
  // ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/ov_msckf/adaptive_time_surface", 100, imageCallback);
  // ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/davis346/image_raw", 100, imageCallback);
  // ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/davis_left/image_raw", 100, imageCallback);
  // ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/dvs/image_raw", 100, imageCallback);
  ros::Subscriber sub = nh->subscribe<sensor_msgs::Image>("/camera/image_raw", 100, imageCallback);

  ros::spin();

  // Done!
  return EXIT_SUCCESS;
}

void imageCallback(const sensor_msgs::Image::ConstPtr& msg){
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  static cv::Mat last_image;
  static double last_time;

  cv::Mat current_image;
  cv::cvtColor(cv_ptr->image, current_image, cv::COLOR_BGR2GRAY);
  cv::Mat mask = cv::Mat::zeros(cv::Size(current_image.cols, current_image.rows), CV_8UC1);

  cv::resize(current_image, current_image, cv::Size(346, 260));

  double current_time = cv_ptr->header.stamp.toSec();

  // Process this new image
  ov_core::CameraData message;
  message.timestamp = cv_ptr->header.stamp.toSec();
  message.sensor_ids.push_back(0);
  message.images.push_back(current_image);
  message.masks.push_back(mask);

  extractor->feed_new_camera(message);

  // If we do the first tracking, then return
  if(last_image.empty()){
    last_image = current_image.clone();
    last_time = current_time;
    return;
  }

  // Get Feature database
  std::shared_ptr<FeatureDatabase> database = extractor->get_feature_database();
  std::vector<std::shared_ptr<Feature>> feats_tracks = database->features_containing(last_time);

  cv::Mat output_image;
  output_image = cv::Mat::zeros(cv::Size(current_image.cols * 2, current_image.rows), CV_8UC1);

  cv::Rect roi1(cv::Point(0,0), cv::Size(current_image.cols, current_image.rows));
  last_image.copyTo(output_image(roi1));

  cv::Rect roi2(cv::Point(current_image.cols, 0), cv::Size(current_image.cols, current_image.rows));
  current_image.copyTo(output_image(roi2));

  cv::cvtColor(output_image, output_image, cv::COLOR_GRAY2BGR);

  std::string path = "/media/zju/SSD/evins_results/tracking/default/";

  for(size_t i = 0; i < feats_tracks.size(); i++){
    for (auto const &pair : feats_tracks[i]->timestamps) {
      if(std::find(pair.second.begin(), pair.second.end(), current_time) != pair.second.end()){
        auto it = std::find(pair.second.begin(), pair.second.end(), last_time);
        int index = std::distance(pair.second.begin(), it);

        for(auto const &pair2 : feats_tracks[i]->uvs){
          cv::Point2f pt_c = cv::Point2f((pair2.second[index])[0], (pair2.second[index])[1]);
          cv::Point2f pt_n = cv::Point2f((pair2.second[index+1])[0] + current_image.cols,(pair2.second[index+1])[1]);;
          cv::circle(output_image, pt_c, 2, cv::Scalar(255, 0, 0), cv::FILLED);
          cv::circle(output_image, pt_n, 2, cv::Scalar(255, 0, 0), cv::FILLED);
          cv::line(output_image, pt_c, pt_n, cv::Scalar(0,255,0));
        }
        
      }else{
        for(auto const &pair2 : feats_tracks[i]->uvs){
          auto it = std::find(pair.second.begin(), pair.second.end(), last_time);
          int index = std::distance(pair.second.begin(), it);
          cv::Point2f pt_c = cv::Point2f((pair2.second[index])[0], (pair2.second[index])[1]);
          cv::circle(output_image, pt_c, 2, cv::Scalar(0, 0, 255), cv::FILLED);
        }
      }
    }
  }

  std::string file_path = path + std::to_string(current_time) + ".png";
  cv::imwrite(file_path, output_image);
  std::cout << "write image to " << file_path << std::endl;

  // Get lost tracks
  std::vector<std::shared_ptr<Feature>> feats_lost = database->features_not_containing_newer(last_time);

  num_lostfeats += feats_lost.size();

  // Mark theses feature pointers as deleted
  for (size_t i = 0; i < feats_lost.size(); i++) {
    // Total number of measurements
    int total_meas = 0;
    for (auto const &pair : feats_lost[i]->timestamps) {
      total_meas += (int)pair.second.size();
    }
    // Update stats
    featslengths += total_meas;
    feats_lost[i]->to_delete = true;
  }

  // Push back the current time, as a clone time
  clonetimes.push_back(last_time);

  // Marginalized features if we have reached 5 frame tracks
  if ((int)clonetimes.size() >= clone_states) {
    // Remove features that have reached their max track length
    double margtime = clonetimes.at(0);
    clonetimes.pop_front();
    std::vector<std::shared_ptr<Feature>> feats_marg = database->features_containing(margtime);
    num_margfeats += feats_marg.size();
    // Delete theses feature pointers
    for (size_t i = 0; i < feats_marg.size(); i++) {
      feats_marg[i]->to_delete = true;
    }
  }

  // Tell the feature database to delete old features
  database->cleanup();

  last_image = current_image.clone();
  last_time = current_time;
  
}

/**
 * This function will process the new stereo pair with the extractor!
 */
void handle_stereo(double time0, double time1, cv::Mat img0, cv::Mat img1) {

  // Animate our dynamic mask moving
  // Very simple ball bounding around the screen example
  cv::Mat mask = cv::Mat::zeros(cv::Size(img0.cols, img0.rows), CV_8UC1);
  static cv::Point2f ball_center;
  static cv::Point2f ball_velocity;
  if (ball_velocity.x == 0 || ball_velocity.y == 0) {
    ball_center.x = (float)img0.cols / 2.0f;
    ball_center.y = (float)img0.rows / 2.0f;
    ball_velocity.x = 2.5;
    ball_velocity.y = 2.5;
  }
  ball_center += ball_velocity;
  if (ball_center.x < 0 || (int)ball_center.x > img0.cols)
    ball_velocity.x *= -1;
  if (ball_center.y < 0 || (int)ball_center.y > img0.rows)
    ball_velocity.y *= -1;
  cv::circle(mask, ball_center, 100, cv::Scalar(255), cv::FILLED);

  // Process this new image
  ov_core::CameraData message;
  message.timestamp = time0;
  message.sensor_ids.push_back(0);
  message.images.push_back(img0);
  message.masks.push_back(mask);
  if (max_cameras == 2) {
    message.sensor_ids.push_back(1);
    message.images.push_back(img1);
    message.masks.push_back(mask);
  }
  extractor->feed_new_camera(message);

  // Display the resulting tracks
  cv::Mat img_active, img_history;
  extractor->display_active(img_active, 255, 0, 0, 0, 0, 255);
  extractor->display_history(img_history, 255, 255, 0, 255, 255, 255);

  // Show our image!
  cv::imshow("Active Tracks", img_active);
  cv::imshow("Track History", img_history);
  cv::waitKey(1);

  // Get lost tracks
  std::shared_ptr<FeatureDatabase> database = extractor->get_feature_database();
  std::vector<std::shared_ptr<Feature>> feats_lost = database->features_not_containing_newer(time0);
  num_lostfeats += feats_lost.size();

  // Mark theses feature pointers as deleted
  for (size_t i = 0; i < feats_lost.size(); i++) {
    // Total number of measurements
    int total_meas = 0;
    for (auto const &pair : feats_lost[i]->timestamps) {
      total_meas += (int)pair.second.size();
    }
    // Update stats
    featslengths += total_meas;
    feats_lost[i]->to_delete = true;
  }

  // Push back the current time, as a clone time
  clonetimes.push_back(time0);

  // Marginalized features if we have reached 5 frame tracks
  if ((int)clonetimes.size() >= clone_states) {
    // Remove features that have reached their max track length
    double margtime = clonetimes.at(0);
    clonetimes.pop_front();
    std::vector<std::shared_ptr<Feature>> feats_marg = database->features_containing(margtime);
    num_margfeats += feats_marg.size();
    // Delete theses feature pointers
    for (size_t i = 0; i < feats_marg.size(); i++) {
      feats_marg[i]->to_delete = true;
    }
  }

  // Tell the feature database to delete old features
  database->cleanup();

  // Debug print out what our current processing speed it
  // We want the FPS to be as high as possible
  ros::Time time_curr = ros::Time::now();
  // if (time_curr.toSec()-time_start.toSec() > 2) {
  if (frames > 60) {
    // Calculate the FPS
    double fps = (double)frames / (time_curr.toSec() - time_start.toSec());
    double lpf = (double)num_lostfeats / frames;
    double fpf = (double)featslengths / num_lostfeats;
    double mpf = (double)num_margfeats / frames;
    // DEBUG PRINT OUT
    PRINT_DEBUG("fps = %.2f | lost_feats/frame = %.2f | track_length/lost_feat = %.2f | marg_tracks/frame = %.2f\n", fps, lpf, fpf, mpf);
    // Reset variables
    frames = 0;
    time_start = time_curr;
    num_lostfeats = 0;
    num_margfeats = 0;
    featslengths = 0;
  }
  frames++;
}
