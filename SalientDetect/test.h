#pragma once
#include "Salient.h"
#include "Implement.h"
#include <opencv2\tracking.hpp>
#include "Tracker.h"
#include "kcf\kcftracker.hpp"
#include "reader.h"

void testDetector(cv::Mat& image);

void testVideo();

void testCvKCF();

void testKCF();

void testTracker();

