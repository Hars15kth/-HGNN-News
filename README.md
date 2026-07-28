# Spherical Weather Prediction with SO(3)-Equivariant Neural Networks

A geometry-aware forecasting pipeline comparing a conventional convolutional neural network with an SO(3)-equivariant spherical model for ERA5 500 hPa wind prediction.

The project investigates whether respecting the rotational geometry of the Earth improves global vector-field prediction relative to applying planar convolutions to a latitude-longitude grid.

The central modelling hypothesis is:

$$\text{Spherical geometry}+\text{rotation equivariance}\Longrightarrow\text{more coherent global prediction}.$$

The repository is designed for reproducibility, stable cache generation, modular experimentation, and Windows-native execution without Docker.

---

## 1. Problem Formulation

Let the atmospheric state at time $t$ be represented by

$$\mathbf{X}*t:\mathbb{S}^2\longrightarrow\mathbb{R}^{d*{\mathrm{in}}},$$

where:

* $\mathbb{S}^2$ is the spherical surface of the Earth;
* $d_{\mathrm{in}}$ is the number of input variables;
* $\mathbf{X}_t(p)$ is the atmospheric state at position $p$.

The prediction target is the 500 hPa wind field

$$\mathbf{Y}_{t+\Delta t}:\mathbb{S}^2\longrightarrow\mathbb{R}^{2}.$$

At position $p$,

$$\mathbf{Y}*{t+\Delta t}(p)=(u*{t+\Delta t}(p),v_{t+\Delta t}(p)),$$

where:

* $u$ is the zonal wind component;
* $v$ is the meridional wind component;
* $\Delta t$ is the forecast horizon.

The learning objective is to construct a model

$$f_\theta:\mathbf{X}*t\longmapsto\widehat{\mathbf{Y}}*{t+\Delta t},$$

such that

$$\widehat{\mathbf{Y}}*{t+\Delta t}\approx\mathbf{Y}*{t+\Delta t}.$$

---

## 2. Spherical Domain

The Earth is represented by the unit sphere

$$\mathbb{S}^2={x\in\mathbb{R}^3:|x|_2=1}.$$

A latitude-longitude grid stores this domain as a rectangular array, but the underlying geometry remains spherical.

A planar convolution is naturally translation equivariant:

$$f(T_a\mathbf{X})=T_a f(\mathbf{X}),$$

where $T_a$ denotes translation by displacement $a$.

Global atmospheric fields instead transform naturally under rotations

$$R\in\mathrm{SO}(3).$$

A spherical equivariant model is designed to satisfy

$$f_\theta(R\cdot\mathbf{X})=R\cdot f_\theta(\mathbf{X}).$$

Rotating the input and then predicting should therefore be equivalent to predicting first and rotating the output afterward.

---

## 3. The Rotation Group SO(3)

The three-dimensional rotation group is

$$\mathrm{SO}(3)={R\in\mathbb{R}^{3\times3}:R^{\mathsf{T}}R=\mathbf{I},\det(R)=1}.$$

For a spherical signal $\mathbf{X}$, the rotated field is

$$(R\cdot\mathbf{X})(p)=\mathbf{X}(R^{-1}p).$$

An equivariant transformation $\Phi$ satisfies

$$\Phi(R\cdot\mathbf{X})=R\cdot\Phi(\mathbf{X}).$$

This encodes the structural rule

$$\text{same physical pattern under rotation}\Longrightarrow\text{consistently transformed representation}.$$

The model therefore does not need to learn equivalent atmospheric structures independently at every orientation.

---

## 4. Spherical Mesh Construction

The spherical model operates on a discrete mesh

$$\mathcal{M}=(\mathcal{V},\mathcal{E}),$$

where:

* $\mathcal{V}$ is the set of spherical vertices;
* $\mathcal{E}$ is the set of neighbourhood relations.

Each vertex is embedded in Cartesian coordinates:

$$\mathbf{r}_i=(x_i,y_i,z_i)\in\mathbb{S}^2.$$

For latitude $\phi_i$ and longitude $\lambda_i$,

$$x_i=\cos(\phi_i)\cos(\lambda_i),$$

$$y_i=\cos(\phi_i)\sin(\lambda_i),$$

and

$$z_i=\sin(\phi_i).$$

For each vertex $i$, the $k$ nearest neighbours are

$$\mathcal{N}_k(i)={j_1,j_2,\ldots,j_k}.$$

The spherical distance between vertices $i$ and $j$ is

$$d_{\mathbb{S}^2}(i,j)=\arccos(\mathbf{r}_i^{\mathsf{T}}\mathbf{r}_j).$$

Neighbourhood construction is performed in chunks to avoid allocating the full pairwise distance matrix.

---

## 5. Baseline CNN

The baseline treats the atmospheric field as a planar tensor

$$\mathbf{X}\in\mathbb{R}^{C\times H\times W}.$$

A convolutional layer computes

$$\mathbf{Z}^{(\ell+1)}=\sigma(\mathbf{K}^{(\ell)}*\mathbf{Z}^{(\ell)}+\mathbf{b}^{(\ell)}),$$

where:

* $*$ denotes planar convolution;
* $\mathbf{K}^{(\ell)}$ is a learned kernel;
* $\mathbf{b}^{(\ell)}$ is a bias term;
* $\sigma$ is a nonlinear activation.

The baseline prediction is

$$\widehat{\mathbf{Y}}*{\mathrm{CNN}}=f*{\mathrm{CNN}}(\mathbf{X}).$$

This model is computationally efficient but does not explicitly preserve spherical rotational structure.

Potential geometric limitations include:

* latitude-dependent grid distortion;
* artificial longitudinal boundaries;
* polar singularities;
* orientation-sensitive filters;
* nonuniform physical area per grid cell.

---

## 6. Spherical Equivariant Model

The spherical model represents atmospheric variables on the mesh:

$$\mathbf{X}*{\mathcal{M}}\in\mathbb{R}^{|\mathcal{V}|\times d*{\mathrm{in}}}.$$

A geometry-aware message-passing layer is written as

$$\mathbf{h}_i^{(\ell+1)}=\sigma\left(\mathbf{W}*0^{(\ell)}\mathbf{h}*i^{(\ell)}+\sum*{j\in\mathcal{N}(i)}\Psi*\ell(\mathbf{r}_i,\mathbf{r}_j)\mathbf{h}_j^{(\ell)}\right).$$

Here:

* $\mathbf{h}_i^{(\ell)}$ is the representation at vertex $i$;
* $\mathcal{N}(i)$ is the spherical neighbourhood of $i$;
* $\Psi_\ell$ is a geometry-dependent interaction map.

The spherical prediction is

$$\widehat{\mathbf{Y}}*{\mathrm{sph}}=f*{\mathrm{sph}}(\mathbf{X}_{\mathcal{M}}).$$

The architecture is designed so that

$$f_{\mathrm{sph}}(R\cdot\mathbf{X}*{\mathcal{M}})=R\cdot f*{\mathrm{sph}}(\mathbf{X}_{\mathcal{M}}).$$

---

## 7. Equivariant Feature Types

SO(3)-equivariant representations may be decomposed into angular orders:

$$\mathbf{h}=\bigoplus_{\ell=0}^{L}\mathbf{h}^{(\ell)}.$$

The $\ell=0$ component transforms as a scalar.

The $\ell=1$ component transforms as a vector.

Higher-order components encode more complex directional structure.

Tensor-product interactions satisfy

$$\mathbf{h}^{(\ell_1)}\otimes\mathbf{h}^{(\ell_2)}=\bigoplus_{\ell=|\ell_1-\ell_2|}^{\ell_1+\ell_2}\mathbf{h}^{(\ell)}.$$

This allows scalar, vector, and higher-order features to interact while preserving their transformation rules.

---

## 8. Wind-Vector Representation

At location $p$, the predicted wind vector is

$$\widehat{\mathbf{w}}(p)=(\widehat{u}(p),\widehat{v}(p)).$$

Its magnitude is

$$|\widehat{\mathbf{w}}(p)|_2=\sqrt{\widehat{u}(p)^2+\widehat{v}(p)^2}.$$

Its direction is

$$\widehat{\alpha}(p)=\arctan2(\widehat{v}(p),\widehat{u}(p)).$$

The true wind vector is

$$\mathbf{w}(p)=(u(p),v(p)).$$

The forecasting task therefore evaluates:

* component-wise accuracy;
* vector magnitude;
* angular coherence;
* spatial consistency;
* divergence behaviour;
* hemispheric robustness.

---

## 9. Training Objectives

The component-wise mean-squared error is

$$\mathcal{L}*{\mathrm{MSE}}=\frac{1}{2N}\sum*{i=1}^{N}\left[(\widehat{u}_i-u_i)^2+(\widehat{v}_i-v_i)^2\right].$$

The component-wise mean absolute error is

$$\mathcal{L}*{\mathrm{MAE}}=\frac{1}{2N}\sum*{i=1}^{N}\left[|\widehat{u}_i-u_i|+|\widehat{v}_i-v_i|\right].$$

The magnitude error is

$$\mathcal{L}*{\mathrm{mag}}=\frac{1}{N}\sum*{i=1}^{N}\left||\widehat{\mathbf{w}}_i|_2-|\mathbf{w}_i|_2\right|.$$

A directional loss is

$$\mathcal{L}*{\mathrm{ang}}=\frac{1}{N}\sum*{i=1}^{N}\left[1-\frac{\widehat{\mathbf{w}}_i^{\mathsf{T}}\mathbf{w}_i}{|\widehat{\mathbf{w}}_i|_2|\mathbf{w}_i|_2+\varepsilon}\right].$$

A composite objective can be defined as

$$\mathcal{L}=\lambda_1\mathcal{L}*{\mathrm{MSE}}+\lambda_2\mathcal{L}*{\mathrm{mag}}+\lambda_3\mathcal{L}_{\mathrm{ang}}.$$

---

## 10. Cache Generation

The pipeline generates reusable NumPy caches for the baseline and spherical representations.

A planar cache has the form

$$\mathbf{C}_{\mathrm{grid}}\in\mathbb{R}^{N\times T\times C\times H\times W}.$$

A spherical cache has the form

$$\mathbf{C}_{\mathrm{sph}}\in\mathbb{R}^{N\times T\times|\mathcal{V}|\times C}.$$

Cache generation is divided into chunks:

$$\mathbf{C}=\bigcup_{b=1}^{B}\mathbf{C}^{(b)}.$$

Each output is written using the sequence

$$\text{temporary write}\longrightarrow\text{integrity check}\longrightarrow\text{atomic rename}.$$

This prevents interrupted writes from being mistaken for complete cache files.

The cache system supports:

* restartable preprocessing;
* deterministic sample ordering;
* bounded memory usage;
* corruption-resistant output generation;
* consistent feature reuse across experiments.

---

## 11. Reproducibility

For random seed $s$,

$$s_{\mathrm{Python}}=s_{\mathrm{NumPy}}=s_{\mathrm{PyTorch}}=s.$$

A controlled model comparison requires

$$\mathcal{D}*{\mathrm{train}}^{\mathrm{CNN}}=\mathcal{D}*{\mathrm{train}}^{\mathrm{sph}},$$

$$\mathcal{D}*{\mathrm{val}}^{\mathrm{CNN}}=\mathcal{D}*{\mathrm{val}}^{\mathrm{sph}},$$

and

$$\mathcal{D}*{\mathrm{test}}^{\mathrm{CNN}}=\mathcal{D}*{\mathrm{test}}^{\mathrm{sph}}.$$

The evaluation functions must also remain fixed:

$$\mathcal{M}*{\mathrm{CNN}}=\mathcal{M}*{\mathrm{sph}}.$$

The repository additionally uses:

* deterministic sample indexing;
* version-pinned dependencies;
* chunked mesh construction;
* atomic NumPy saves;
* explicit output manifests;
* Windows-native execution;
* restart-safe preprocessing.

---

## 12. Evaluation Metrics

### Mean-Squared Error

$$\mathrm{MSE}=\frac{1}{2N}\sum_{i=1}^{N}\left[(\widehat{u}_i-u_i)^2+(\widehat{v}_i-v_i)^2\right].$$

### Mean Absolute Error

$$\mathrm{MAE}=\frac{1}{2N}\sum_{i=1}^{N}\left[|\widehat{u}_i-u_i|+|\widehat{v}_i-v_i|\right].$$

### Root-Mean-Squared Error

$$\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.$$

### Angular Error

$$\mathrm{AngleError}=\frac{1}{N}\sum_{i=1}^{N}\arccos\left(\frac{\widehat{\mathbf{w}}_i^{\mathsf{T}}\mathbf{w}_i}{|\widehat{\mathbf{w}}_i|_2|\mathbf{w}_i|_2+\varepsilon}\right).$$

### Magnitude Mean Absolute Error

$$\mathrm{MagMAE}=\frac{1}{N}\sum_{i=1}^{N}\left||\widehat{\mathbf{w}}_i|_2-|\mathbf{w}_i|_2\right|.$$

### Absolute Divergence Error

$$\mathrm{DivAbs}=\frac{1}{N}\sum_{i=1}^{N}\left|\nabla\cdot\widehat{\mathbf{w}}_i-\nabla\cdot\mathbf{w}_i\right|.$$

### Northern-Hemisphere MSE

$$\mathrm{MSE}*{\mathrm{North}}=\frac{1}{|\mathcal{N}|}\sum*{i\in\mathcal{N}}|\widehat{\mathbf{w}}_i-\mathbf{w}_i|_2^2.$$

### Southern-Hemisphere MSE

$$\mathrm{MSE}*{\mathrm{South}}=\frac{1}{|\mathcal{S}|}\sum*{i\in\mathcal{S}}|\widehat{\mathbf{w}}_i-\mathbf{w}_i|_2^2.$$

---

## 13. Performance Summary

| Metric                    | Baseline CNN | Spherical Model | Relative Change |
| :------------------------ | -----------: | --------------: | --------------: |
| MSE                       |       0.5319 |      **0.5257** |  **1.2% lower** |
| MAE                       |       0.5533 |      **0.5471** |  **1.1% lower** |
| RMSE                      |       0.7293 |      **0.7250** |  **0.6% lower** |
| Angle Error               |   **0.4181** |          0.4336 |     3.7% higher |
| Magnitude MAE             |       0.5512 |      **0.4546** | **17.5% lower** |
| Divergence Absolute Error |   **0.1841** |          0.1941 |     5.4% higher |
| MSE North                 |       0.6267 |      **0.6235** |  **0.5% lower** |
| MSE South                 |       0.4370 |      **0.4279** |  **2.1% lower** |
| Composite Score           |       0.6798 |      **0.6676** |  **1.8% lower** |

The spherical model performs better on seven of the nine raw metric rows shown above: MSE, MAE, RMSE, magnitude MAE, northern MSE, southern MSE, and composite score.

It performs worse on:

* angle error;
* divergence absolute error.

The largest improvement is in magnitude MAE:

$$\Delta_{\mathrm{MagMAE}}=\frac{0.5512-0.4546}{0.5512}\times100%\approx17.5%.$$

The MSE improvement is

$$\Delta_{\mathrm{MSE}}=\frac{0.5319-0.5257}{0.5319}\times100%\approx1.2%.$$

The southern-hemisphere improvement is

$$\Delta_{\mathrm{South}}=\frac{0.4370-0.4279}{0.4370}\times100%\approx2.1%.$$

The composite-score improvement is

$$\Delta_{\mathrm{Composite}}=\frac{0.6798-0.6676}{0.6798}\times100%\approx1.8%.$$

---

## 14. Result Interpretation

The spherical model achieves lower values for:

* total MSE;
* total MAE;
* RMSE;
* magnitude MAE;
* northern-hemisphere MSE;
* southern-hemisphere MSE;
* composite score.

The strongest improvement occurs in predicted wind magnitude.

This indicates that the equivariant model reconstructs vector-field strength more accurately, even though its mean directional error is slightly higher.

The observed trade-off is

$$\text{improved magnitude and aggregate field accuracy}\quad\text{versus}\quad\text{slightly weaker angular and divergence accuracy}.$$

The southern-hemisphere improvement is larger than the northern-hemisphere improvement:

$$2.1%>0.5%.$$

This suggests that the value of spherical geometry is not spatially uniform and should be examined through regional error maps rather than aggregate metrics alone.

---

## 15. Spatial Error Analysis

Let the pointwise vector error of model $m$ be

$$E_m(p)=|\widehat{\mathbf{w}}_m(p)-\mathbf{w}(p)|_2.$$

Define the spherical-model improvement field as

$$\Delta E(p)=E_{\mathrm{CNN}}(p)-E_{\mathrm{sph}}(p).$$

Then:

* $\Delta E(p)>0$ means the spherical model has lower error;
* $\Delta E(p)<0$ means the CNN baseline has lower error.

The improvement maps reveal where geometric equivariance helps and where the planar model remains stronger.

Reported improvements are concentrated in:

* mid-latitude regions;
* parts of the southern hemisphere;
* regions with strong magnitude variation;
* spatially coherent vector structures.

---

## 16. Visual Outputs

The primary wind-field visualisation compares:

* ERA5 ground truth;
* baseline CNN prediction;
* spherical-model prediction;
* baseline absolute error;
* spherical absolute error;
* model-improvement heatmaps.

The comparison includes zonal wind $u$ and meridional wind $v$.

Red regions in the improvement map indicate

$$E_{\mathrm{sph}}(p)<E_{\mathrm{CNN}}(p).$$

Blue regions indicate

$$E_{\mathrm{CNN}}(p)<E_{\mathrm{sph}}(p).$$

These maps provide a spatial explanation of the aggregate metric differences.

---

## 17. Architectural Significance

The project demonstrates the principle

$$\text{representation symmetry should match data symmetry}.$$

When a domain has known transformations, the model can be designed so that its internal features transform consistently with those transformations.

This principle applies to systems involving:

* globally distributed inputs;
* orientation-dependent fields;
* spatially indexed representations;
* geometry-aware retrieval;
* multiresolution data;
* distributed cache generation;
* representations shared across coordinate systems.

A conventional model learns transformation behaviour indirectly.

An equivariant model constrains that behaviour directly:

$$\text{input transformation}\Longrightarrow\text{predictable feature transformation}.$$

This can improve consistency and reduce the need to relearn equivalent structures from multiple orientations.

---

## 18. Repository Structure

```text
.
├── scripts/
│   ├── preprocess.py
│   ├── build_baseline_cache.py
│   ├── build_spherical_cache.py
│   ├── train_baseline.py
│   └── train_spherical.py
├── models/
│   ├── baseline_cnn.py
│   ├── spherical_model.py
│   └── tensor_product_blocks.py
├── notebooks/
│   └── weather_analysis.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── summaries/
├── cache/
│   ├── baseline/
│   └── spherical/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 19. How to Run

Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python scripts/preprocess.py
```

Build the baseline cache:

```bash
python scripts/build_baseline_cache.py
```

Build the spherical cache:

```bash
python scripts/build_spherical_cache.py
```

Train the baseline model:

```bash
python scripts/train_baseline.py
```

Train the spherical model:

```bash
python scripts/train_spherical.py
```

Launch the analysis notebook:

```bash
jupyter notebook notebooks/weather_analysis.ipynb
```

---

## 20. Environment

* Python
* PyTorch
* NumPy
* SciPy
* scikit-learn
* xarray
* NetCDF4
* Matplotlib
* Jupyter
* Windows-native execution
* no Docker requirement
* chunked KNN construction
* atomic NumPy cache saves

---

## 21. Generated Artifacts

### Processed Data

* normalised ERA5 tensors;
* zonal and meridional wind targets;
* train, validation, and test indices;
* latitude-longitude coordinate arrays;
* spherical Cartesian coordinates.

### Mesh Data

* spherical mesh vertices;
* KNN neighbourhood indices;
* geodesic distances;
* mesh metadata;
* coordinate mappings.

### Cached Data

* baseline grid caches;
* spherical mesh caches;
* integrity summaries;
* sample manifests;
* temporary atomic-write files.

### Evaluation Outputs

* metric tables;
* wind-field comparisons;
* magnitude-error maps;
* angular-error maps;
* hemispheric evaluations;
* improvement heatmaps;
* composite-score summaries.

---

## 22. Limitations

The experiment does not establish universal superiority of spherical models over planar CNNs.

The spherical model performs worse on:

* angular error;
* divergence absolute error.

Performance depends on

$$M=F(\mathcal{D},\mathcal{M},\theta,\Omega,\Lambda,\Delta t),$$

where:

* $\mathcal{D}$ is the ERA5 dataset;
* $\mathcal{M}$ is the spherical mesh;
* $\theta$ is the model parameterisation;
* $\Omega$ is the optimisation procedure;
* $\Lambda$ is the collection of loss weights;
* $\Delta t$ is the forecast horizon.

The benchmark-specific conclusion is:

$$\text{spherical model}>\text{baseline CNN}$$

for seven of the nine reported metric rows and for the composite score.

Directional and divergence performance may require additional physical constraints or revised loss weighting.

---

## Core Result

$$\mathrm{MSE}*{\mathrm{sph}}=0.5257<0.5319=\mathrm{MSE}*{\mathrm{CNN}}.$$

$$\mathrm{MagMAE}*{\mathrm{sph}}=0.4546<0.5512=\mathrm{MagMAE}*{\mathrm{CNN}}.$$

$$\mathrm{MSE}*{\mathrm{South,sph}}=0.4279<0.4370=\mathrm{MSE}*{\mathrm{South,CNN}}.$$

$$\mathrm{Composite}*{\mathrm{sph}}=0.6676<0.6798=\mathrm{Composite}*{\mathrm{CNN}}.$$

The benchmark supports the principle:

$$\text{geometry-aware equivariant modelling}>\text{planar modelling alone}$$

for the dominant global-field metrics in this ERA5 wind-prediction experiment.

---

## Licence

This repository is licensed for personal portfolio and evaluation use only.

Redistribution, commercial use, resale, or incorporation into commercial systems is prohibited without explicit written permission from the author.




