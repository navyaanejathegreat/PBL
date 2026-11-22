#include "qc_metrics.hpp"

#include <algorithm>
#include <numeric>
#include <set>

namespace qc {

double iou(const Box& a, const Box& b) {
    double ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2, ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    double bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2, bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
    double iw = std::max(0.0, std::min(ax2, bx2) - std::max(ax1, bx1));
    double ih = std::max(0.0, std::min(ay2, by2) - std::max(ay1, by1));
    double inter = iw * ih, uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0.0;
}

double average_precision(const std::vector<int>& tp, int n_gt) {
    // Same recipe as Ultralytics' compute_ap: pad the PR curve with (0,1) and (1,0), take the precision
    // envelope, interpolate it linearly at 101 recall points and integrate with the trapezoid rule.
    if (n_gt == 0) return 0.0;
    size_t n = tp.size();
    std::vector<double> mrec(1, 0.0), mpre(1, 1.0);
    double ctp = 0;
    for (size_t i = 0; i < n; ++i) {
        ctp += tp[i];
        mrec.push_back(ctp / n_gt);
        mpre.push_back(ctp / double(i + 1));
    }
    mrec.push_back(n ? mrec.back() : 1.0);  // precision drops to 0 after the last detection
    mpre.push_back(0.0);
    mrec.push_back(1.0);
    mpre.push_back(0.0);
    for (size_t i = mpre.size(); i-- > 1;) mpre[i - 1] = std::max(mpre[i - 1], mpre[i]);
    auto interp = [&](double x) {  // np.interp(x, mrec, mpre)
        size_t j = std::upper_bound(mrec.begin(), mrec.end(), x) - mrec.begin();
        if (j == 0) return mpre[0];
        if (j >= mrec.size()) return mpre.back();
        double x0 = mrec[j - 1], x1 = mrec[j];
        return x1 > x0 ? mpre[j - 1] + (mpre[j] - mpre[j - 1]) * (x - x0) / (x1 - x0) : mpre[j];
    };
    double area = 0, prev = interp(0.0);
    for (int k = 1; k <= 100; ++k) {
        double y = interp(k / 100.0);
        area += 0.5 * (prev + y) / 100.0;
        prev = y;
    }
    return area;
}

namespace {
typedef std::map<std::pair<int, int>, std::vector<int> > Index;  // (image, cls) -> box indices

Index index_by(const std::vector<Box>& v) {
    Index idx;
    for (size_t i = 0; i < v.size(); ++i) idx[std::make_pair(v[i].image, v[i].cls)].push_back(int(i));
    return idx;
}

// tp flag for every prediction at one IoU threshold; also returns matched IoUs.
std::vector<int> match(const std::vector<Box>& gt, const std::vector<Box>& preds, const Index& gi,
                       const Index& pi, double thr, double min_conf, std::vector<double>* ious) {
    std::vector<int> tp(preds.size(), 0);
    for (Index::const_iterator it = pi.begin(); it != pi.end(); ++it) {
        std::vector<int> p = it->second;
        std::sort(p.begin(), p.end(), [&](int a, int b) { return preds[a].conf > preds[b].conf; });
        Index::const_iterator g = gi.find(it->first);
        if (g == gi.end()) continue;
        std::vector<char> used(g->second.size(), 0);
        for (size_t a = 0; a < p.size(); ++a) {
            if (preds[p[a]].conf < min_conf) continue;
            double best = thr;
            int bj = -1;
            for (size_t j = 0; j < g->second.size(); ++j) {
                if (used[j]) continue;
                double v = iou(preds[p[a]], gt[g->second[j]]);
                if (v >= best) { best = v; bj = int(j); }
            }
            if (bj >= 0) {
                used[bj] = 1;
                tp[p[a]] = 1;
                if (ious) ious->push_back(best);
            }
        }
    }
    return tp;
}
}  // namespace

Report evaluate(const std::vector<Box>& gt, const std::vector<Box>& preds, int n_classes, double op_conf) {
    Report r;
    r.classes.assign(n_classes, ClassResult());
    for (size_t i = 0; i < gt.size(); ++i) r.classes[gt[i].cls].n_gt++;
    Index gi = index_by(gt), pi = index_by(preds);

    std::vector<int> order(preds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return preds[a].conf > preds[b].conf; });

    for (int t = 0; t < 10; ++t) {
        double thr = 0.5 + 0.05 * t;
        std::vector<int> tp = match(gt, preds, gi, pi, thr, 0.0, nullptr);
        std::vector<std::vector<int> > per(n_classes);
        for (size_t k = 0; k < order.size(); ++k) per[preds[order[k]].cls].push_back(tp[order[k]]);
        for (int c = 0; c < n_classes; ++c) r.classes[c].ap.push_back(average_precision(per[c], r.classes[c].n_gt));
    }

    std::vector<double> ious;
    std::vector<int> tp_op = match(gt, preds, gi, pi, 0.5, op_conf, &ious);
    std::vector<int> tp_c(n_classes, 0);
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i].conf < op_conf) continue;
        r.classes[preds[i].cls].n_pred++;
        tp_c[preds[i].cls] += tp_op[i];
    }
    double s50 = 0, s5095 = 0, sf1 = 0;
    for (int c = 0; c < n_classes; ++c) {
        ClassResult& cr = r.classes[c];
        cr.precision = cr.n_pred ? double(tp_c[c]) / cr.n_pred : 0.0;
        cr.recall = cr.n_gt ? double(tp_c[c]) / cr.n_gt : 0.0;
        cr.f1 = cr.precision + cr.recall > 0 ? 2 * cr.precision * cr.recall / (cr.precision + cr.recall) : 0.0;
        if (cr.n_gt == 0) continue;  // classes absent from the test set are not scored
        r.classes_scored++;
        s50 += cr.ap[0];
        s5095 += std::accumulate(cr.ap.begin(), cr.ap.end(), 0.0) / cr.ap.size();
        sf1 += cr.f1;
    }
    if (r.classes_scored) {
        r.map50 = s50 / r.classes_scored;
        r.map50_95 = s5095 / r.classes_scored;
        r.macro_f1 = sf1 / r.classes_scored;
    }
    r.mean_iou = ious.empty() ? 0.0 : std::accumulate(ious.begin(), ious.end(), 0.0) / ious.size();
    return r;
}

SortReport sort_quality(const std::vector<Box>& gt, const std::vector<Box>& preds,
                        const std::vector<std::string>& stream_of, const std::vector<std::string>& names,
                        double op_conf, double purity_target) {
    static const char* kStreams[] = {"temp", "screen", "lamp", "large", "small", "it", "battery"};
    static const std::set<std::string> kData = {"Smartphone", "Bar-Phone", "Laptop", "Desktop-PC", "HDD",
                                                "SSD", "USB-Flash-Drive", "Server"};
    SortReport s;
    std::map<std::string, int> bin_of;
    for (int i = 0; i < 7; ++i) {
        BinResult b;
        b.stream = kStreams[i];
        bin_of[b.stream] = i;
        s.bins.push_back(b);
    }
    // group by image
    std::map<int, std::vector<int> > gimg, pimg;
    for (size_t i = 0; i < gt.size(); ++i) gimg[gt[i].image].push_back(int(i));
    for (size_t i = 0; i < preds.size(); ++i)
        if (preds[i].conf >= op_conf) pimg[preds[i].image].push_back(int(i));
    s.gt_items = int(gt.size());

    for (std::map<int, std::vector<int> >::iterator it = gimg.begin(); it != gimg.end(); ++it) {
        std::vector<int>& g = it->second;
        std::vector<int> p = pimg[it->first];
        std::sort(p.begin(), p.end(), [&](int a, int b) { return preds[a].conf > preds[b].conf; });
        std::vector<char> used(g.size(), 0);
        for (size_t a = 0; a < p.size(); ++a) {
            const Box& d = preds[p[a]];
            const std::string& st = stream_of[d.cls];
            BinResult& bin = s.bins[bin_of[st]];
            bin.items++;
            s.routed++;
            if (kData.count(names[d.cls])) bin.data_bearing++;
            // best unused real item that overlaps
            double best = 0.5;
            int bj = -1;
            for (size_t j = 0; j < g.size(); ++j) {
                if (used[j]) continue;
                double v = iou(d, gt[g[j]]);
                if (v >= best) { best = v; bj = int(j); }
            }
            if (bj >= 0) {
                used[bj] = 1;
                if (stream_of[gt[g[bj]].cls] == st) {
                    bin.correct++;
                    s.gt_recovered++;
                } else if (names[gt[g[bj]].cls] == "Battery") {
                    bin.hazards++;  // a real battery sent to the wrong bin
                }
            }
        }
    }
    int correct = 0;
    for (size_t i = 0; i < s.bins.size(); ++i) {
        correct += s.bins[i].correct;
        if (s.bins[i].items && s.bins[i].purity() < purity_target) s.contaminated_bins++;
    }
    s.overall_purity = s.routed ? double(correct) / s.routed : 0.0;
    s.recovery = s.gt_items ? double(s.gt_recovered) / s.gt_items : 0.0;
    return s;
}

}  // namespace qc
