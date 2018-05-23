// -*- c++ -*-
#ifndef OPENFACEWIDGET_H
#define OPENFACEWIDGET_H

#include <QPixmap>
#include <QWidget>
#include <QGraphicsView>
#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QSplitter>

#include "render-thread.h"

class OpenfaceWidget : public QMainWindow {
  Q_OBJECT
public:
  OpenfaceWidget(QWidget *parent = 0);

protected:

private slots:
  void updatePixmap(const QImage &image, double scaleFactor);
private:
  RenderThread m_thread;
  QGraphicsPixmapItem m_pixmap;
  QWidget m_information_container;
  QSplitter m_main_app_splitter;
  QGraphicsView m_graphics_view;
};

#endif
