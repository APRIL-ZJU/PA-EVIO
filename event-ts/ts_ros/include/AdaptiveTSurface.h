#ifndef adaptive_time_surface_H_
#define adaptive_time_surface_H_

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <mutex>
#include <Eigen/Eigen>

namespace adaptive_time_surface
{

class EventData: public dvs_msgs::Event
{
  public:
    double activity;
    EventData(){
      ts = ros::Time(0);
      x = 0;
      y = 0;
      polarity = 0;
      activity = 0;
    }
    EventData(const dvs_msgs::Event& e){
      ts = e.ts;
      x = e.x;
      y = e.y;
      polarity = e.polarity;
      activity = 0;
    }
};


#define NUM_THREAD_TS 12
using EventQueue = std::deque<EventData>;

class AdaptiveDecay
{
  public:
  AdaptiveDecay() = delete;

  AdaptiveDecay(double w_th, double coeff){
    coeff_ = coeff;
    w_th_ = w_th;
    activity_ = 0;
    weight_ = 0;
    count_ = 0;
    t_init_ = ros::Time(0);
  }

  bool propagate(EventData& last_event){
    if(last_event.ts < t_last_){
      // std::cout << "Events are not in order" << std::endl;
      // reset(time_event);
      
      return false;
    }

    beta_ = 1.0 / (1 + coeff_ * activity_ * (last_event.ts - t_last_).toSec());
    activity_ = beta_ * activity_ + 1.0;

    last_event.activity = activity_;
    t_last_ = last_event.ts;

    return true;
  }

  void reset(ros::Time& t0){
    t_init_ = t0;
    count_ = 0;
  }

  double get_weight(){  
    return weight_;
  }

  double get_activity(){
    return activity_;
  }

  double get_coeff(){
    return coeff_;
  }

  void set_threshold(double w_th){
    w_th_ = w_th;
  }

  private:
  double activity_;
  double beta_;
  double weight_;

  ros::Time t_init_;
  ros::Time t_last_;
  int count_;

  double w_th_;
  double coeff_ = 0.1;
};


class EventQueueMat 
{
public:
  EventQueueMat(int width, int height, int queueLen)
  {
    width_ = width;
    height_ = height;
    queueLen_ = queueLen;
    eqMat_ = std::vector<EventQueue>(width_ * height_, EventQueue());
  }

  void insertEvent(const EventData& e)
  {
    if(!insideImage(e.x, e.y))
      return;
    else
    {
      EventQueue& eq = getEventQueue(e.x, e.y);
      eq.push_back(e);
      while(eq.size() > queueLen_)
        eq.pop_front();
    }
  }

  bool getMostRecentEventBeforeT(
    const size_t x,
    const size_t y,
    const ros::Time& t,
    EventData* ev)
  {
    if(!insideImage(x, y))
      return false;

    EventQueue& eq = getEventQueue(x, y);
    if(eq.empty())
      return false;

    for(auto it = eq.rbegin(); it != eq.rend(); ++it)
    {
      const EventData& e = *it;
      if(e.ts < t)
      {
        *ev = *it;
        return true;
      }
    }
    return false;
  }

  void clear()
  {
    eqMat_.clear();
  }

  bool insideImage(const size_t x, const size_t y)
  {
    return !(x < 0 || x >= width_ || y < 0 || y >= height_);
  }

  inline EventQueue& getEventQueue(const size_t x, const size_t y)
  {
    return eqMat_[x + width_ * y];
  }

  size_t width_;
  size_t height_;
  size_t queueLen_;
  std::vector<EventQueue> eqMat_;
};

class AdaptiveTSurface
{
  struct Job
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EventQueueMat* pEventQueueMat_;
    cv::Mat* pTimeSurface_;
    size_t start_col_, end_col_;
    size_t start_row_, end_row_;
    size_t i_thread_;
    ros::Time external_sync_time_;
    double decay_sec_;
    double activity_;
    double time_init_;
    int* event_count_;
  };

public:
  AdaptiveTSurface(ros::NodeHandle & nh);
  AdaptiveTSurface(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~AdaptiveTSurface();

  // event processing
  void feed_new_events(const dvs_msgs::EventArray::ConstPtr& msg);
  bool get_latest_map(cv::Mat &map, ros::Time &time);
  bool get_specific_map(cv::Mat &map, ros::Time &time);
  bool get_next_map(cv::Mat &map, ros::Time &time);
  ros::Time get_latest_event_timestamp();
  ros::Time get_oldest_event_timestamp();

  void save_event_ts();
  void publish_event_frame();
  void save_event_frames(const dvs_msgs::EventArray::ConstPtr& msg, std::string path);

private:
  ros::NodeHandle nh_;
  // core
  void init(int width, int height);
  bool createATSAtTime_hyperthread(const ros::Time& external_sync_time);
  void thread_ats(Job &job); // single thread version (This is enough for DAVIS240C and DAVIS346)
  bool createAdaptiveTSurface(ros::Time& ats_ts); // hyper thread version (This is for higher resolution)

  // callbacks
  void syncCallback(const std_msgs::TimeConstPtr& msg);
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);

  // utils
  void clearEventQueue();

  // sub & pub
  ros::Subscriber event_sub_;
  ros::Subscriber sync_topic_;
  image_transport::Publisher adaptive_time_surface_pub_;

  // online parameters
  bool bCamInfoAvailable_;
  bool bUse_Sim_Time_;  
  cv::Size sensor_size_;
  ros::Time sync_time_;
  bool bSensorInitialized_;
  bool bPublish_ts_;
  bool bMap_per_Msg_;

  // Thresholds
  double weight_threshold_ = 0; // 110(0.05)
  int min_valid_pixels_; // 1000
  double coeff; // 0.1
  double contrast_threshold; // 0.5
  double temp_contrast;
  int filter_size_ = 1;

  cv::Mat temp_ ;
  ros::Time temp_time_ = ros::Time(0);
  std::deque<ros::Time> temp_map_time_queue_;
  ros::Time temp_map_oldest_event_time_;  // temp_map_time aquisition

  // offline parameters
  double decay_ms_;
  bool ignore_polarity_;
  int median_blur_kernel_size_;
  int max_event_queue_length_;
  int events_maintained_size_;

  // containers
  EventQueue events_;
  std::shared_ptr<EventQueueMat> pEventQueueMat_;
  std::shared_ptr<AdaptiveDecay> ad_;

  // thread mutex
  std::mutex data_mutex_;

  std::string debug_path = "/media/zju/SSD/evins_results/davis240c/images/";

  // Time Surface Mode
  enum TimeSurfaceMode
  {
    BACKWARD,// used in the T-RO20 submission
    FORWARD
  } time_surface_mode_;
};
} // namespace adaptive_time_surface
#endif // adaptive_time_surface_H_