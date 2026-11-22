# cpp_core: from-scratch neural-network module (C++)

Covers Project 07 items 3–6 (Units I–III). Eigen is used only as the array type;
neurons, backprop, losses and optimizers are written by hand.

| Program | What it demonstrates | Outputs (in `results/`) |
|---|---|---|
| `neurons` | McCulloch-Pitts AND/OR/NAND/NOT, XOR needs two layers, sigmoid neuron trained by GD, step vs sigmoid vs tanh | `activations.csv`, `sigmoid_neuron_loss.csv` |
| `sandbox` | tensor creation, broadcasting, inner products, reshaping (`Eigen::Map`), hand-derived gradient checked numerically, SVD of e-waste crops | `svd_spectrum.csv`, `svd_reconstruction.csv`, `eigen_images.csv` |
| `mlp_train gradcheck` | backprop vs central differences for sigmoid / tanh / ReLU | console |
| `mlp_train train` | one MLP run: logs, confusion matrix, weights, config with hyperparameters vs learned parameter count | `<run>_log.csv`, `<run>_confusion.csv`, `<run>_config.csv`, `<run>_weights.bin` |
| `mlp_train compare-opt` | batch GD vs SGD vs Momentum vs Nesterov, same seed | `compare_optimizers.csv` + logs |
| `mlp_train ablation` | baseline, then one change at a time: init, normalization, L2, flip augmentation, sigmoid/tanh | `ablation.csv` + logs |
| `learning_rules` | Unit III unsupervised rules: plain Hebb vs Oja (converges to PC1, checked against SVD), competitive winner-take-all learning (cluster purity vs labels), restricted Boltzmann machine with CD-1 (reconstructions, filters, Gibbs samples, free energy) | `hebbian_*.csv`, `oja_vs_pc1.csv`, `competitive_*.csv`, `rbm_*.csv` |
| `unit_tests` | gates, softmax stability, gradient checks, optimizers, data helpers | console, exit code |

## 1. Build

**Option A: plain g++ (works now, even with old MinGW g++ 6.3)**
```
cd cpp_core
build_gpp.bat
```
It downloads Eigen 3.4.0 into `third_party\eigen` with git on first run, builds into `bin\`, then runs the unit tests.

**Option B: CMake (after installing Visual Studio 2022 Build Tools, "Desktop development with C++")**
Open "Developer PowerShell for VS 2022", then:
```
cd cpp_core
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

## 2. Make the crops (Python, label preparation only)
```
pip install pillow pyyaml
python python\export_crops.py --src D:\ewaste_v2 --out crops --classes 10 --per-class 600
```
Uses the dataset's own train/valid/test split, so the crops stay leakage-safe.
Use `D:\ewaste_final` instead if `ewaste_v2` is not built yet.

## 3. Run the experiments
```
mkdir results
bin\neurons results
bin\sandbox --data crops\train.csv --out results
bin\mlp_train gradcheck
bin\mlp_train compare-opt --data crops --epochs 30 --out results
bin\mlp_train ablation    --data crops --epochs 30 --out results
bin\learning_rules --data crops --out results
```
(With the CMake build the programs are in `build\Release\` instead of `bin\`.)

Useful options: `--hidden 256,128 --act relu --init he --opt momentum --lr 0.05 --momentum 0.9 --l2 1e-4 --batch 64 --epochs 30 --seed 0 --no-norm --augment`.
`compare-opt` uses lr 0.5 for full-batch GD, 0.05 for SGD and 0.01 for Momentum/Nesterov unless `--lr` is given.

## 4. Figures for the report
```
pip install matplotlib pandas numpy
python python\plot_results.py --results results --classes crops\classes.txt
```
