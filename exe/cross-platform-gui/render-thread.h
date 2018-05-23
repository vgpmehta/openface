// -*- c++ -*-
#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QMutex>
#include <QObject>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

class RenderThread : public QThread {
  Q_OBJECT
public:
  RenderThread(QObject *parent = 0);
  ~RenderThread();

signals:
  void renderedImage(const QImage &image, double scaleFactor);

protected:
  void run() override;
public slots:
  void do_csv_work(void);
private:
  QMutex mutex;
  QWaitCondition condition;
  QSize resultSize;

  bool restart;
  bool abort;
};
#endif
