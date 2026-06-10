#pragma once
#include <condition_variable>
#include <mutex>

class ThreadBarrier {
 public:
  int m_threadCount = 0;
  int m_currentThreadCount = 0;

  std::mutex m_mutex;
  std::condition_variable m_cv;

 public:
  inline ThreadBarrier(int threadCount) { m_threadCount = threadCount; };

 public:
  inline void Sync() {
    bool wait = false;

    m_mutex.lock();

    m_currentThreadCount = (m_currentThreadCount + 1) % m_threadCount;

    wait = (m_currentThreadCount != 0);

    m_mutex.unlock();

    if (wait) {
      std::unique_lock<std::mutex> lk(m_mutex);
      m_cv.wait(lk);
    } else {
      m_cv.notify_all();
    }
  };
};