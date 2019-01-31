#include "simcontrol/image_publisher_nodelet.h"

namespace simcontrol {

void ImagePublisherNodelet::onInit() {
    ROS_INFO_STREAM("Hello");
}

PLUGINLIB_EXPORT_CLASS(simcontrol::ImagePublisherNodelet,
    nodelet::Nodelet);

}

