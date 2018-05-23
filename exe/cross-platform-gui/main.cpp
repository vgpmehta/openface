// -*- c++ -*-
#include <QApplication>

#include "OpenfaceWidget.h"

int main(int argc, char **argv) {
  QApplication app{argc, argv};
  OpenfaceWidget widget{};
  widget.show();
  return app.exec();
}
