#pragma once
#include "Salient.h"
#include "Implement.h"
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include "salientTracker.h"
#include "kcf/kcftracker.hpp"
#include "reader.h"

void testDetector();

void testVideo();

//void testCvKCF();

void testKCF(const char* path);

void testTracker(const char* path);

