// qc_app: E-waste recycling quality controller (C++ demo app).
//
// Reads the detector output written by python/export_predictions.py and reports
//   * detection quality: mAP@0.5, mAP@0.5:0.95, macro F1, mean IoU, per-class table
//   * sorting quality: purity of every WEEE bin, recovery rate, contamination flags,
//     batteries sent to the wrong bin (fire hazard), data-bearing devices
//   * speed: model FPS (from the deployment wrapper) and the C++ controller's own throughput
//
// Usage:  qc_app --dir qc_input\yolo26n [--conf 0.25] [--purity 0.95] [--out results\qc]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "args.hpp"
#include "portable.hpp"
#include "qc_metrics.hpp"

using qc::Box;

static std::vector<std::vector<std::string> > read_csv(const std::string& path) {
    std::ifstream f(path.c_str());
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<std::string> > rows;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) cells.push_back(c);
        rows.push_back(cells);
    }
    return rows;
}

int main(int argc, char** argv) {
    try {
        nn::Args args(argc, argv);
        std::string dir = args.str("dir", "qc_input");
        double op_conf = args.num("conf", 0.25), target = args.num("purity", 0.95);
        std::string out = args.str("out", "");

        std::vector<std::string> names, stream_of;
        std::vector<std::vector<std::string> > rows = read_csv(dir + "/classes.csv");
        for (size_t i = 0; i < rows.size(); ++i) { names.push_back(rows[i][1]); stream_of.push_back(rows[i][2]); }

        std::vector<Box> gt, preds;
        rows = read_csv(dir + "/gt.csv");
        for (size_t i = 0; i < rows.size(); ++i) {
            Box b; b.image = nn::to_int(rows[i][0]); b.cls = nn::to_int(rows[i][1]);
            b.cx = nn::to_double(rows[i][2]); b.cy = nn::to_double(rows[i][3]);
            b.w = nn::to_double(rows[i][4]); b.h = nn::to_double(rows[i][5]);
            gt.push_back(b);
        }
        rows = read_csv(dir + "/preds.csv");
        for (size_t i = 0; i < rows.size(); ++i) {
            Box b; b.image = nn::to_int(rows[i][0]); b.cls = nn::to_int(rows[i][1]);
            b.conf = nn::to_double(rows[i][2]);
            b.cx = nn::to_double(rows[i][3]); b.cy = nn::to_double(rows[i][4]);
            b.w = nn::to_double(rows[i][5]); b.h = nn::to_double(rows[i][6]);
            preds.push_back(b);
        }
        int n_images = 0;
        double total_ms = 0, inf_ms = 0;
        rows = read_csv(dir + "/timing.csv");
        if (!rows.empty()) {
            n_images = nn::to_int(rows[0][0]);
            total_ms = nn::to_double(rows[0][1]);
            inf_ms = nn::to_double(rows[0][3]);
        }

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        qc::Report r = qc::evaluate(gt, preds, int(names.size()), op_conf);
        qc::SortReport s = qc::sort_quality(gt, preds, stream_of, names, op_conf, target);
        double qc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== E-waste recycling quality controller ===\n"
                  << "input: " << dir << "   images: " << n_images << "   real items: " << gt.size()
                  << "   detections: " << preds.size() << "\n\n";
        std::cout << "-- Detection quality (test set) --\n"
                  << "  mAP@0.5       " << r.map50 << "\n"
                  << "  mAP@0.5:0.95  " << r.map50_95 << "\n"
                  << "  macro F1      " << r.macro_f1 << "   (confidence >= " << op_conf << ", IoU 0.5)\n"
                  << "  mean IoU      " << r.mean_iou << "   (of correctly detected items)\n"
                  << "  classes scored " << r.classes_scored << " of " << names.size() << "\n\n";

        std::cout << "-- Sorting quality (each detection routed to its WEEE bin) --\n";
        std::cout << "  bin        items  correct  purity   flag\n";
        for (size_t i = 0; i < s.bins.size(); ++i) {
            const qc::BinResult& b = s.bins[i];
            if (!b.items) continue;
            std::string flag = b.purity() < target ? "CONTAMINATED" : "ok";
            if (b.hazards) flag += "  + " + nn::to_str(b.hazards) + " battery(ies): FIRE RISK";
            std::cout << "  " << std::left << std::setw(9) << b.stream << std::right << std::setw(7) << b.items
                      << std::setw(9) << b.correct << std::setw(9) << b.purity() << "   " << flag << "\n";
        }
        std::cout << "  overall purity " << s.overall_purity << "   recovery " << s.recovery
                  << " (" << s.gt_recovered << "/" << s.gt_items << " real items reached the right bin)\n"
                  << "  bins below the " << target << " purity target: " << s.contaminated_bins << "\n\n";

        std::cout << std::setprecision(1) << "-- Speed --\n";
        if (n_images && total_ms > 0)
            std::cout << "  model, end to end   " << n_images / (total_ms / 1000.0) << " FPS\n"
                      << "  model, network only " << n_images / (inf_ms / 1000.0) << " FPS\n";
        std::cout << "  C++ controller      " << qc_ms << " ms for all " << n_images << " images ("
                  << (qc_ms > 0 ? n_images / (qc_ms / 1000.0) : 0.0) << " images/s)\n\n";

        if (!out.empty()) {
            std::ofstream f((out + "_summary.csv").c_str());
            f << std::setprecision(4) << std::fixed;
            f << "metric,value\nmap50," << r.map50 << "\nmap50_95," << r.map50_95 << "\nmacro_f1," << r.macro_f1
              << "\nmean_iou," << r.mean_iou << "\noverall_purity," << s.overall_purity << "\nrecovery," << s.recovery
              << "\ncontaminated_bins," << s.contaminated_bins << "\nfps_end_to_end,"
              << (total_ms > 0 ? n_images / (total_ms / 1000.0) : 0) << "\nfps_network,"
              << (inf_ms > 0 ? n_images / (inf_ms / 1000.0) : 0) << "\ncontroller_ms," << qc_ms << "\n";
            std::ofstream pc((out + "_per_class.csv").c_str());
            pc << std::setprecision(4) << std::fixed << "class,stream,gt,pred,precision,recall,f1,ap50,ap50_95\n";
            for (size_t c = 0; c < names.size(); ++c) {
                const qc::ClassResult& cr = r.classes[c];
                double m = 0;
                for (size_t k = 0; k < cr.ap.size(); ++k) m += cr.ap[k];
                pc << names[c] << "," << stream_of[c] << "," << cr.n_gt << "," << cr.n_pred << "," << cr.precision
                   << "," << cr.recall << "," << cr.f1 << "," << cr.ap[0] << "," << m / cr.ap.size() << "\n";
            }
            std::ofstream bf((out + "_bins.csv").c_str());
            bf << std::setprecision(4) << std::fixed << "bin,items,correct,purity,battery_hazards,data_bearing\n";
            for (size_t i = 0; i < s.bins.size(); ++i)
                bf << s.bins[i].stream << "," << s.bins[i].items << "," << s.bins[i].correct << ","
                   << s.bins[i].purity() << "," << s.bins[i].hazards << "," << s.bins[i].data_bearing << "\n";
            std::cout << "Saved " << out << "_summary.csv, _per_class.csv, _bins.csv\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
