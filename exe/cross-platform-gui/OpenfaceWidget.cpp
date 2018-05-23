#include <iostream>

#include <QApplication>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>

#include "OpenfaceWidget.h"

OpenfaceWidget::OpenfaceWidget(QWidget *parent) : QMainWindow(parent), m_main_app_splitter(this) {

  QTimer *timer_worker = new QTimer{this};
  QObject::connect(timer_worker, &QTimer::timeout, &m_thread, &RenderThread::do_csv_work);

  timer_worker->start(2000);

  QObject::connect(&m_thread, &RenderThread::renderedImage,
		   this, &OpenfaceWidget::updatePixmap);
  QObject::connect(&m_thread, &QThread::finished, this, &QObject::deleteLater);

  QVBoxLayout layout{};
  QPushButton *quitBtn = new QPushButton{"Close", this};
  QObject::connect(quitBtn, &QPushButton::clicked, qApp, &QApplication::quit);

  layout.addWidget(quitBtn);

  m_information_container.setLayout(&layout);

  m_main_app_splitter.addWidget(&m_graphics_view);
  m_main_app_splitter.addWidget(&m_information_container);

  m_graphics_view.setScene(new QGraphicsScene(this));
  m_graphics_view.scene()->addItem(&m_pixmap);
  m_graphics_view.setMinimumWidth(950);
  setCentralWidget(&m_main_app_splitter);
  setWindowTitle(tr("OpenfaceWidget"));
  resize(1280, 720);
  m_thread.start();
}

void OpenfaceWidget::updatePixmap(const QImage &qimg, double scaleFactor) {
  m_pixmap.setPixmap(QPixmap::fromImage(qimg.rgbSwapped()));
  m_graphics_view.fitInView(&m_pixmap, Qt::KeepAspectRatio);
  update();
}
