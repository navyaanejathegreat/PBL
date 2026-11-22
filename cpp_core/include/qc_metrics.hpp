// Detection metrics and recycling quality control for the C++ demo app.
#pragma once
#include <map>
#include <string>
#include <vector>

namespace qc {

struct Box {
    int image = 0, cls = 0;
    double conf = 1.0;
    double cx = 0, cy = 0, w = 0, h = 0;  // normalised centre/size
};

double iou(const Box& a, const Box& b);

// Average precision with 101-point interpolation (COCO / Ultralytics style).
// tp: 1/0 per prediction sorted by confidence (descending); n_gt: number of ground-truth boxes.
double average_precision(const std::vector<int>& tp_sorted, int n_gt);

struct ClassResult {
    int n_gt = 0, n_pred = 0;
    std::vector<double> ap;        // one per IoU threshold 0.50:0.05:0.95
    double precision = 0, recall = 0, f1 = 0;   // at the operating confidence, IoU 0.5
};

struct Report {
    std::vector<ClassResult> classes;
    double map50 = 0, map50_95 = 0, macro_f1 = 0, mean_iou = 0;
    int classes_scored = 0;
};

// Greedy matching per image and class, highest confidence first (COCO protocol).
Report evaluate(const std::vector<Box>& gt, const std::vector<Box>& preds, int n_classes, double op_conf);

struct BinResult {
    std::string stream;
    int items = 0, correct = 0, hazards = 0, data_bearing = 0;
    double purity() const { return items ? double(correct) / items : 1.0; }
};

struct SortReport {
    std::vector<BinResult> bins;
    double overall_purity = 0, recovery = 0;
    int routed = 0, gt_items = 0, gt_recovered = 0, contaminated_bins = 0;
};

// Every detection with conf >= op_conf goes to the bin of its predicted stream. It is "correct" when
// it overlaps (IoU >= 0.5) a real item whose class belongs to that same stream.
SortReport sort_quality(const std::vector<Box>& gt, const std::vector<Box>& preds,
                        const std::vector<std::string>& stream_of_class,
                        const std::vector<std::string>& class_names, double op_conf, double purity_target);

}  // namespace qc
