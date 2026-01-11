#include <TimeSurface.h>
#include <AdaptiveTSurface.h>

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "esvo_time_surface");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  esvo_time_surface::TimeSurface ts(nh, nh_private);
  // adaptive_time_surface::AdaptiveTSurface ts(nh, nh_private);

  ros::spin();

  return 0;
}
