# GeoForget

**GeoForget** is a mathematically rigorous framework for machine unlearning, leveraging **Riemannian geometry** to overcome the limitations of standard Euclidean optimization. This repository provides code and methodology to safely and effectively erase information from deep learning models while maximizing utility.

---

## 🚀 Core Idea: "Rails vs. Anchors"

- **Standard unlearning** methods (like L2UL or Finetuning) treat all parameters equally (Euclidean distance), often damaging model utility by "dragging" parameters off the manifold of valid solutions.
- **GeoForget** redefines the problem using the **Fisher-Rao Metric**:

  - **Rails:** Instead of freezing weights (EWC "anchors"), the Fisher Information Matrix builds "rails" guiding optimization, so the model slides along the curvature of the "Remain Data" manifold.
  - **Engine:** A targeted adversarial attack pushes the model away from the forget class, while a KL-divergence term safely bounds the unlearning energy.

---

## 🌟 Features

- **Riemannian Optimization:** Uses the Fisher Information Matrix (FIM) to measure geodesic (true curved) distance in parameter space.
- **Guided Surgery:** Automatically identifies and locks "innocent" parameters (high Fisher info) while releasing "guilty" ones (low Fisher info) for modification.
- **Safety Bounded:** Replaces unbounded Cross-Entropy with KL-Divergence to prevent gradient explosion during unlearning.
- **Adversarial Robustness:** Incorporates L2-bounded adversarial training to ensure the forgotten class is truly erased.

---

## 📐 Mathematical Formulation

In standard deep learning, parameters are treated as points in a flat Euclidean space (all changes are equal).

GeoForget treats parameters as points on a curved statistical manifold, equipped with the Fisher Information Matrix (FIM) as the Riemannian metric.

- **Approximated Geodesic Distance:**
  ```
  (θ - θ*)ᵗ · Fisher · (θ - θ*)
  ```

- **Optimization Objective:**
  Minimize:
  ```
  Total Loss = Loss_Forget + λ_adv * Loss_Adv + λ_geo * Geodesic_Dist
  ```
  - **Forgetting Loss:** Push forget class distribution toward uniform noise (KL Divergence).
  - **Adversarial Loss:** Robustly pushes decision boundary away.
  - **Geodesic Pull:** Restores and locks "Remain" knowledge via Fisher Information.

---

## 🛠️ Usage

**Installation:**  
Clone the repository and install dependencies (standard PyTorch stack).

**Run GeoForget Unlearning Example:**
```bash
python main.py \
  --method geoforget \
  --unlearn_rate 1e-4 \
  --adv_lambda 0.1 \
  --geo_lambda 10.0 \
  --description "fisher_rao_preservation"
```

### Parameters

- `--geo_lambda`: Controls Riemannian constraint strength. Higher = stricter preservation of remain accuracy.
- `--adv_lambda`: Controls "push" force away from forget class.
- `--adv_eps`: Radius of adversarial sphere (default: 0.4).

---

## 📊 GeoForget Results

| Dataset         | Remain Acc | Forget Acc | MIA Privacy Score |
|-----------------|:----------:|:----------:|:-----------------:|
| CIFAR-10        |   92.8%    |    0.0%    |      0.87         |
| CIFAR-100       |   73.1%    |    0.5%    |      0.71         |
| VGG Face        |   89.2%    |    0.0%    |      0.78         |

*Note: "MIA Privacy Score" is normalized (1.0 = Indistinguishable from Retrained Model). Higher is better.*

GeoForget demonstrates superior privacy-utility trade-offs on complex manifolds like Faces and CIFAR-100.


## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

*By Mohamed Ansaff Abdul Salam*
