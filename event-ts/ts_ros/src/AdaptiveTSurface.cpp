#include <AdaptiveTSurface.h>
#include <TicToc.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/Float32.h>
#include <glog/logging.h>
#include <thread>

namespace adaptive_time_surface {
double contrast_Variance(const cv::Mat& image){
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    static cv::Vec4d mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    const double contrast = stddev[0]*stddev[0];

    return contrast;
}

AdaptiveTSurface::AdaptiveTSurface(ros::NodeHandle & nh, ros::NodeHandle nh_private)
  : nh_(nh)
{
  // setup subscribers and publishers
  event_sub_ = nh_.subscribe("events", 100, &AdaptiveTSurface::eventsCallback, this);
  sync_topic_ = nh_.subscribe("sync", 1, &AdaptiveTSurface::syncCallback, this);
  image_transport::ImageTransport it_(nh_);
  adaptive_time_surface_pub_ = it_.advertise("adaptive_time_surface", 1);

  // read parameters
  nh_private.param<bool>("use_sim_time", bUse_Sim_Time_, true);
  nh_private.param<bool>("ignore_polarity", ignore_polarity_, true);

  int TS_mode;
  nh_private.param<int>("time_surface_mode", TS_mode, 0);
  time_surface_mode_ = (TimeSurfaceMode)TS_mode;
  nh_private.param<int>("median_blur_kernel_size", median_blur_kernel_size_, 1);
  nh_private.param<int>("max_event_queue_len", max_event_queue_length_, 20);
  nh_private.param<bool>("publish_ts", bPublish_ts_, 20);

  nh_private.param<double>("weight_threshold", weight_threshold_, 0.1);
  nh_private.param<double>("coeff", coeff, 0.1);
  nh_private.param<double>("contrast_threshold", contrast_threshold, 0.5);
  nh_private.param<int>("filter_size", filter_size_, 1);
  nh_private.param<int>("min_valid_pixels", min_valid_pixels_, 1000);

  bCamInfoAvailable_ = false;
  bSensorInitialized_ = false;

  if(pEventQueueMat_)
    pEventQueueMat_->clear();
  sensor_size_ = cv::Size(0,0);
  ad_ = std::make_shared<AdaptiveDecay>(weight_threshold_, coeff);
}

AdaptiveTSurface::AdaptiveTSurface(ros::NodeHandle & nh): nh_(nh)
{
  image_transport::ImageTransport it_(nh_);
  adaptive_time_surface_pub_ = it_.advertise("adaptive_time_surface", 1);

  std::string path = ros::package::getPath("ts_ros") + "/cfg/parameters.yaml";
  std::cout << path << std::endl;
  cv::FileStorage fs(path, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    std::cout << "Error: Failed to open the config file!" << std::endl;
    exit;
  }

  fs["use_sim_time"] >> bUse_Sim_Time_;
  fs["ignore_polarity"] >> ignore_polarity_;
  fs["decay_ms"] >> decay_ms_;

  int TS_mode;

  fs["time_surface_mode"] >> TS_mode;
  time_surface_mode_ = (TimeSurfaceMode)TS_mode;

  fs["median_blur_kernel_size"] >> median_blur_kernel_size_;
  fs["max_event_queue_len"] >> max_event_queue_length_;
  fs["publish_ts"] >> bPublish_ts_;
  fs["map_per_msg"] >> bMap_per_Msg_;

  // Parameter Reading
  fs["weight_threshold"] >> weight_threshold_;
  fs["min_valid_pixels"] >> min_valid_pixels_;
  fs["coeff"] >> coeff;
  fs["contrast_threshold"] >> contrast_threshold;
  fs["filter_size"] >> filter_size_;

  bCamInfoAvailable_ = false;
  bSensorInitialized_ = false;

  if(pEventQueueMat_)
    pEventQueueMat_->clear();
  sensor_size_ = cv::Size(0,0);
  ad_ = std::make_shared<AdaptiveDecay>(weight_threshold_, coeff);
}

AdaptiveTSurface::~AdaptiveTSurface()
{
  if(adaptive_time_surface_pub_!=nullptr)
    adaptive_time_surface_pub_.shutdown();
}

void AdaptiveTSurface::init(int width, int height)
{
  sensor_size_ = cv::Size(width, height);
  bSensorInitialized_ = true;
  pEventQueueMat_.reset(new EventQueueMat(width, height, max_event_queue_length_));
  ROS_INFO("Sensor size: (%d x %d)", sensor_size_.width, sensor_size_.height);
}

bool AdaptiveTSurface::createATSAtTime_hyperthread(const ros::Time& external_sync_time)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  boost::posix_time::ptime rt1, rt2, rt3, rt4;

  rt1 = boost::posix_time::microsec_clock::local_time();
  if(!bSensorInitialized_)
    return false;

  double current_activity = 0;
  auto it = std::lower_bound(events_.begin(), events_.end(), external_sync_time,
                              [](const EventData& e, const ros::Time& t) {
                                  return e.ts < t;
                              });

  if (it != events_.begin()) {
      --it; // 回退到小于 external_sync_time 的最近事件
      current_activity = it->activity;
  }

  if(current_activity == 0){
    return false;
  }

  double delta_time = (1.0/weight_threshold_ - 1.0) / (current_activity * coeff);
  double time_init = external_sync_time.toSec() - delta_time;

  rt2 = boost::posix_time::microsec_clock::local_time();

  // create exponential-decayed Time Surface map.
  cv::Mat time_surface_map;
  time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

  // distribute jobs
  std::vector<Job> jobs(NUM_THREAD_TS);
  size_t num_col_per_thread = sensor_size_.width / NUM_THREAD_TS;
  size_t res_col = sensor_size_.width % NUM_THREAD_TS;
  int event_count = 0;
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pEventQueueMat_ = pEventQueueMat_.get();
    jobs[i].pTimeSurface_ = &time_surface_map;
    jobs[i].start_col_ = num_col_per_thread * i;
    if(i == NUM_THREAD_TS - 1)
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1 + res_col;
    else
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1;
    jobs[i].start_row_ = 0;
    jobs[i].end_row_ = sensor_size_.height - 1;
    jobs[i].external_sync_time_ = external_sync_time;
    jobs[i].activity_ = current_activity;
    jobs[i].time_init_ = time_init;
    jobs[i].event_count_ = &event_count;
  }

  // hyper thread processing
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREAD_TS);
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
    threads.emplace_back(std::bind(&AdaptiveTSurface::thread_ats, this, jobs[i]));
  for(auto& thread:threads)
    if(thread.joinable())
      thread.join();

  rt3 = boost::posix_time::microsec_clock::local_time();

  // polarity
  if(!ignore_polarity_)
    time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
  else
    time_surface_map = 255.0 * time_surface_map;
  time_surface_map.convertTo(time_surface_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);

  temp_ = time_surface_map;
  temp_time_ = external_sync_time;

  if(event_count < min_valid_pixels_){
    return false;
  }

  double contrast = contrast_Variance(temp_);
  if(temp_contrast == 0){
    temp_contrast = contrast;
  }else{
    if(contrast < contrast_threshold * temp_contrast){
      std::cout << "ATS Contrast smaller than thres." << std::endl;
      return false;
    }
    temp_contrast = temp_contrast * 0.9 + contrast * 0.1;
  }
  
  rt4 = boost::posix_time::microsec_clock::local_time();
  if(bPublish_ts_){
    publish_event_frame();

    double time_step1 = (rt2 - rt1).total_microseconds() * 1e-6;
    double time_step2 = (rt3 - rt2).total_microseconds() * 1e-6;
    double time_step3 = (rt4 - rt3).total_microseconds() * 1e-6;
    double total_time = (rt4 - rt1).total_microseconds() * 1e-6;
    std::cout << std::fixed << "TS Gen TIme: <1>:" << time_step1 << " <2>:" << time_step2 
      << " <3>:" << time_step3 << " <Total>:" << total_time << " s." << std::endl;
  }

  return true;
}

void AdaptiveTSurface::thread_ats(Job &job)
{
  EventQueueMat & eqMat = *job.pEventQueueMat_;
  cv::Mat& adaptive_time_surface_map = *job.pTimeSurface_;
  size_t start_col = job.start_col_;
  size_t end_col = job.end_col_;
  size_t start_row = job.start_row_;
  size_t end_row = job.end_row_;
  size_t i_thread = job.i_thread_;

  double activity = job.activity_;
  double time_init = job.time_init_;
  int& event_count = *job.event_count_;

  for(size_t y = start_row; y <= end_row; y++)
    for(size_t x = start_col; x <= end_col; x++)
    {
      EventData most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, job.external_sync_time_, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0 && most_recent_stamp_at_coordXY.toSec() >= time_init)
        {
          event_count++;
          const double dt = (job.external_sync_time_ - most_recent_stamp_at_coordXY).toSec();
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          double expVal = 1.0 / (1 + coeff * most_recent_event_at_coordXY_before_T.activity * dt);
          if(!ignore_polarity_)
            expVal *= polarity;

          adaptive_time_surface_map.at<double>(y,x) = expVal;

        }
      } // a most recent event is available
    }
}



bool AdaptiveTSurface::createAdaptiveTSurface(ros::Time& ats_ts){
  std::lock_guard<std::mutex> lock(data_mutex_);

  double current_activity = 0;
  for(auto it = events_.rbegin(); it != events_.rend(); ++it)
  {
    const EventData& e = *it;
    if(e.ts < ats_ts)
    {
      current_activity = (*it).activity;
    }
  }

  if(current_activity == 0){
    return false;
  }

  double time_init = ats_ts.toSec() - (1.0 / weight_threshold_ - 1.0) / (current_activity * coeff);

  int count_events = 0;

  cv::Mat ats_map = cv::Mat::zeros(sensor_size_, CV_64F);
  // Loop through all coordinates
  for(int y=0; y<sensor_size_.height; ++y)
  {
    for(int x=0; x<sensor_size_.width; ++x)
    {
      EventData most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, ats_ts, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0 && most_recent_stamp_at_coordXY.toSec() >= time_init)
        {
          count_events++;
          const double dt = (ats_ts - most_recent_stamp_at_coordXY).toSec();
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          double expVal = 1.0 / (1 + coeff * most_recent_event_at_coordXY_before_T.activity * dt);

          if(!ignore_polarity_)
            expVal *= polarity;
            
          ats_map.at<double>(y,x) = expVal;
        }
      } // a most recent event is available
    }// loop x
  }// loop y

  if(count_events < min_valid_pixels_){
    return false;
  }

  if(!ignore_polarity_)
  {
    ats_map = 255.0 * (ats_map + 1.0) / 2.0;
  }
  else{
    ats_map = 255.0 * ats_map;
  }
      
  ats_map.convertTo(ats_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(ats_map, ats_map, 2 * median_blur_kernel_size_ + 1);

  temp_ = ats_map;
  temp_time_ = ats_ts;

  return true;
}

void AdaptiveTSurface::syncCallback(const std_msgs::TimeConstPtr& msg)
{
  if(bUse_Sim_Time_)
    sync_time_ = ros::Time::now();
  else
    sync_time_ = msg->data;

#ifdef ESVO_TS_LOG
    TicToc tt;
    tt.tic();
#endif
    if(NUM_THREAD_TS == 1)
      createAdaptiveTSurface(sync_time_);
    if(NUM_THREAD_TS > 1)
      createATSAtTime_hyperthread(sync_time_);
#ifdef ESVO_TS_LOG
    LOG(INFO) << "Time Surface map's creation takes: " << tt.toc() << " ms.";
#endif
}

void AdaptiveTSurface::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  if(!bSensorInitialized_)
    init(msg->width, msg->height);

  feed_new_events(msg);

  while(!temp_map_time_queue_.empty()){
    createATSAtTime_hyperthread(temp_map_time_queue_.front());
    temp_map_time_queue_.pop_front();
  }

  clearEventQueue();
}

void AdaptiveTSurface::feed_new_events(const dvs_msgs::EventArray::ConstPtr& msg){
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_)
    init(msg->width, msg->height);

  int interval = filter_size_;
  int interval_count = 0;
  for(const dvs_msgs::Event& e : msg->events){
    EventData event_data(e);

    events_.push_back(event_data);
    int i = events_.size() - 2;
    while(i >= 0 && events_[i].ts > event_data.ts)
    {
      events_[i+1] = events_[i];
      i--; 
    }
    events_[i+1] = event_data;

    EventData& last_event = events_.back();

    static double curr_activity = 0;
    interval_count++;
    
    if((interval_count % interval) == 0){
      interval_count = 0;
      bool if_propagated = ad_->propagate(last_event);
      if(if_propagated == true){
        pEventQueueMat_->insertEvent(last_event);
        curr_activity = last_event.activity;

        // Method 1: adaptive output frequency
        // double temp_weight = 1.0 / (1 + coeff * last_event.activity * (last_event.ts - temp_map_oldest_event_time_).toSec());
        // if(temp_weight <= weight_threshold){
        //   temp_map_time_queue_.push_back(last_event.ts);
        //   temp_map_oldest_event_time_ = last_event.ts;
        //   temp_weight = 1.0;
        // }
      }
    }else if(curr_activity != 0){
      last_event.activity = curr_activity;
      pEventQueueMat_->insertEvent(last_event);
    }
  }

  // Method 2: last timestamp from last event
  if(bMap_per_Msg_){
    temp_map_time_queue_.push_back(events_.back().ts);
  }
}

bool AdaptiveTSurface::get_latest_map(cv::Mat &map, ros::Time &time){
  ros::Time ats_ts = get_latest_event_timestamp();

  if(createATSAtTime_hyperthread(ats_ts)){
    map = temp_.clone();
    time = ats_ts;
    return true;
  }else{
    map.release();
    time = ros::Time(0);
  }
  return false;
}

bool AdaptiveTSurface::get_specific_map(cv::Mat &map, ros::Time &time){
  if(createATSAtTime_hyperthread(time)){
    map = temp_.clone();
    return true;
  }
    
  map.release();
  time = ros::Time(0);
  return false;
}

bool AdaptiveTSurface::get_next_map(cv::Mat &map, ros::Time &time){
  if(!temp_map_time_queue_.empty()){
    time = temp_map_time_queue_.front();
    temp_map_time_queue_.pop_front();
    if(createATSAtTime_hyperthread(time)){
      map = temp_.clone();
      return true;
    }
  }

  map.release();
  time = ros::Time(0);
  return false;
}

ros::Time AdaptiveTSurface::get_latest_event_timestamp(){
  std::lock_guard<std::mutex> lock(data_mutex_);
  ros::Time result;
  if(events_.empty()){
    result = ros::Time(0);
  }else{
    result = events_.back().ts;
  }
  return result;
}

ros::Time AdaptiveTSurface::get_oldest_event_timestamp(){
  std::lock_guard<std::mutex> lock(data_mutex_);
  ros::Time result;
  if(events_.empty()){
    result = ros::Time(0);
  }else{
    result = events_.front().ts;
  }
  return result;

}

void AdaptiveTSurface::save_event_ts(){
  std::string name = std::to_string(temp_time_.toSec()) + ".png";
  cv::imwrite(debug_path + name, temp_);
}

void AdaptiveTSurface::save_event_frames(const dvs_msgs::EventArray::ConstPtr& msg, std::string path){
    cv::Mat event_map = cv::Mat::zeros(sensor_size_, CV_8UC3);

    for(const dvs_msgs::Event& e : msg->events)
    {
      cv::Point2i pt(e.x, e.y);
      if(e.polarity)
        event_map.at<cv::Vec3b>(pt) = cv::Vec3b(0, 0, 255);
      else
        event_map.at<cv::Vec3b>(pt) = cv::Vec3b(255, 0, 0);
    }

    double t = msg->header.stamp.toSec();
    std::string name = std::to_string(t) + ".png";
    cv::imwrite(path + name, event_map);
}

void AdaptiveTSurface::publish_event_frame(){
  static cv_bridge::CvImage cv_image;
  cv_image.encoding = "mono8";
  cv_image.image = temp_.clone();

  if(adaptive_time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_image.header.stamp = temp_time_;
    adaptive_time_surface_pub_.publish(cv_image.toImageMsg());
  }
}

void AdaptiveTSurface::clearEventQueue()
{
  static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 100000;
  if (events_.size() > MAX_EVENT_QUEUE_LENGTH)
  {
    size_t remove_events = events_.size() - MAX_EVENT_QUEUE_LENGTH;
    events_.erase(events_.begin(), events_.begin() + remove_events);
  }
}

} // namespace esvo_time_surface