// -*- c++ -*-
#include <QApplication>
#include <QThread>
#include <iostream>
#include <QWidget>
#include <unistd.h>

class gui_application : public QWidget {
public:
  void ping() {
    std::cout << "Hello world\n";
  }
};

int main(int argc, char **argv) {
  QApplication handle{argc, argv};

  gui_application gui{};

  QThread *thread = QThread::create([&gui] {
    while (true) {
      gui.ping();
      sleep(1);
    }
  });

  thread->setObjectName("ImageProcessingWorker");
  thread->start();

  gui.resize(250, 150);
  gui.setWindowTitle("Clicker");
  gui.show();
  return handle.exec();
}
