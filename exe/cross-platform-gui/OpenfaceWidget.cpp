#include <iostream>
#include <QApplication>

#include "OpenfaceWidget.h"

OpenfaceWidget::OpenfaceWidget(QWidget *parent) : QMainWindow(parent) {
  QObject::connect(&m_thread, &RenderThread::renderedImage,
		   this, &OpenfaceWidget::updatePixmap);
  QObject::connect(&m_thread, &QThread::finished, this, &QObject::deleteLater);
  m_graphics_view.setScene(new QGraphicsScene(this));
  m_graphics_view.scene()->addItem(&m_pixmap);
  setCentralWidget(&m_graphics_view);
  setWindowTitle(tr("OpenfaceWidget"));
  resize(1280, 720);
  m_thread.start();
}

void OpenfaceWidget::updatePixmap(const QImage &qimg, double scaleFactor) {
  m_pixmap.setPixmap(QPixmap::fromImage(qimg.rgbSwapped()));
  m_graphics_view.fitInView(&m_pixmap, Qt::KeepAspectRatio);
  update();
}
