#ifndef _LOGGING_H_
#define _LOGGING_H_

#define LOG(status) LOG_##status()

class LOG_FATAL {
public:
  LOG_FATAL() {}
  ~LOG_FATAL() {}
  template <typename T>
  LOG_FATAL& operator<<(const T& obj) {
    return *this;
  }
};

#endif
