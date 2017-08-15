/*
 * tracker.h
 *
 *  Created on: 2017Äê8ÔÂ10ÈÕ
 *      Author: JYF
 */

#ifndef SRC_TRACKER_H_
#define SRC_TRACKER_H_

#include "util.h"
#include "features.h"
#include "impl.h"

void showFrame(string file_list, string gt_file);

void runTrackerSimple(string file_list, string gt_file, string ope_file);

void runTracker(string file_list, string gt_file, bool bOneFrameLag=true);

#endif /* SRC_TRACKER_H_ */
