#ifndef IMAGE_PUBLISHER_NODELET_H
#define IMAGE_PUbLISHER_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace simcontrol {
class ImagePublisherNodelet : public nodelet::Nodelet {
public:
  ImagePublisherNodelet() { return; }
  ~ImagePublisherNodelet() { return; }
private:
  virtual void onInit();
};
}
#endif

