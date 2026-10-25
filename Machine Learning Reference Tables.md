# Machine Learning Reference Tables

Oct 25, 2026 · @Sama

The key comparison tables from our machine learning lessons, collected in one place. New tables are added after each session.

## Session 1: Machine learning, unsupervised learning and K-means

### The three paradigms of machine learning

| Paradigm | Data it learns from | Goal | Examples |
| --- | --- | --- | --- |
| Supervised | Inputs X with labels y | Predict y for new X | Spam detection, house prices |
| Unsupervised | Inputs X only, no labels | Find structure in X | Customer segments, anomaly detection |
| Reinforcement | Actions and rewards from an environment | Learn behaviour that maximizes reward | Game playing, robot control |

### Unsupervised learning: four kinds of questions

| Family | Question it answers | Real example |
| --- | --- | --- |
| Clustering | Which points belong together? | Customer segmentation, grouping news articles |
| Dimensionality reduction | Can I describe the data with fewer variables? | Showing 50 features in a 2D plot |
| Anomaly detection | Which points don't fit? | Credit card fraud, faulty sensors |
| Association rules | Which items occur together? | "Customers who buy bread also buy butter" |

### Where K-means struggles

| Problem | Why | Better choice |
| --- | --- | --- |
| Non-round shapes (moons, rings) | Boundaries between clusters are straight lines | DBSCAN, spectral clustering |
| Very different cluster sizes | Each boundary sits halfway between two centroids | Gaussian mixture models |
| Outliers | The mean is pulled toward extreme values | K-medoids |
| Unscaled features | A feature in thousands (income) dominates one in units (age) | Scale the features first |
| Categorical data | Categories have no mean | K-modes |
| Unknown K | K must be chosen in advance | Hierarchical clustering, DBSCAN |
| Many dimensions | Distances become similar | Reduce dimensions first (PCA), then cluster |

## Session 2: Algorithms, seeds and random\_state

### Five properties of an algorithm

| Property | Meaning | Tea example |
| --- | --- | --- |
| Input | It takes zero or more inputs | Water, tea bag, cup |
| Output | It produces at least one result | A cup of tea |
| Definiteness | Every step is exact and unambiguous | "Boil 250 ml of water", not "some water" |
| Finiteness | It stops after a finite number of steps | "Steep 3 minutes", not "steep until ready" |
| Effectiveness | Every step can actually be carried out | "Add sugar" works; "make it taste perfect" doesn't |

### Algorithm vs program vs model

| Term | What it is | K-means example |
| --- | --- | --- |
| Algorithm | The idea: the steps, independent of any language | "Assign to nearest centroid, move centroid to mean, repeat" |
| Program | An algorithm written in a programming language | scikit-learn's `KMeans` class |
| Model | What you get after running a learning algorithm on data | The centroid coordinates learned from your dataset |

### random\_state values in scikit-learn

| Value | Behaviour | When to use it |
| --- | --- | --- |
| `None` (default) | Fresh randomness each run; results can change | When you deliberately want variety |
| An integer, e.g. `0` or `42` | Same random choices every run; fully reproducible | Homework, reports, debugging, comparing results |
| A generator object | Shares one generator across several steps | Advanced pipelines |

## Session 3: Scaling, cluster evaluation and the scikit-learn API

### Scalers and transformers compared

| Tool | Type | Output | Robust to outliers | Changes shape | Typical use before K-means |
| --- | --- | --- | --- | --- | --- |
| `MinMaxScaler` | Scaler | \[0, 1\] on the fitted data | No | No | Bounded features: scores, percentages, pixels |
| `StandardScaler` | Scaler | Mean 0, std 1, unbounded | No | No | The default |
| `RobustScaler` | Scaler | Median 0, IQR 1, unbounded | Yes | No | Outliers you want to keep (incomes) |
| `MaxAbsScaler` | Scaler | \[−1, 1\] | No | No | Sparse data (keeps zeros) |
| `QuantileTransformer` | Transformer | Uniform \[0, 1\] or normal | Yes (very) | Yes | Strong skew, extreme outliers, only order matters |
| `PowerTransformer` | Transformer | Close to normal, standardized | Partly | Yes | Right-skewed features (incomes, prices) |

### What fit learns for each scaler

| Tool | What `fit` learns (attributes) | What `transform` does |
| --- | --- | --- |
| `MinMaxScaler` | `data_min_`, `data_max_` | Moves the min to 0 and the max to 1 |
| `StandardScaler` | `mean_`, `scale_`, `var_` | Subtracts the mean, divides by the standard deviation |
| `RobustScaler` | `center_` (median), `scale_` (IQR) | Subtracts the median, divides by the IQR |
| `QuantileTransformer` | `quantiles_` | Replaces each value by its rank, mapped to uniform or normal |
| `PowerTransformer` | `lambdas_` | Applies a power function, then standardizes |

### Scores for judging clusters

| Score | Function | Measures | Better when | Needs true labels? |
| --- | --- | --- | --- | --- |
| Inertia | `km.inertia_` | Squared distance of points to their own centroid | Lower, but always falls as K grows | No |
| Silhouette | `silhouette_score(X, labels)` | Fit to own cluster vs nearest other cluster | Higher (−1 to 1) | No |
| Davies–Bouldin | `davies_bouldin_score(X, labels)` | Similarity of each cluster to its most similar one | **Lower** | No |
| Calinski–Harabasz | `calinski_harabasz_score(X, labels)` | Spread between clusters vs within clusters | Higher | No |
| Adjusted Rand index | `adjusted_rand_score(y, labels)` | Agreement with the true groups | Higher (1 = perfect) | **Yes** |

### What a silhouette value means

| Silhouette value | Meaning |
| --- | --- |
| Close to 1 | Well inside its own cluster, far from the others |
| Around 0 | On the border between two clusters |
| Negative | Closer to another cluster than to its own: probably in the wrong cluster |

### fit, transform, fit\_transform, predict, fit\_predict

| Method | What it does | Returns | On a scaler | On KMeans |
| --- | --- | --- | --- | --- |
| `fit(X)` | Learns parameters from X | The object itself | Learns `mean_`, `scale_` | Learns `cluster_centers_`, `labels_`, `inertia_` |
| `transform(X)` | Applies learned parameters to X | A new array, same number of rows | The scaled data | Distances to each centroid |
| `fit_transform(X)` | `fit(X)`, then `transform(X)`, on the same X | The transformed array | Learn and scale in one call | Learn, then distances |
| `predict(X)` | One output per row, using learned parameters | Labels, shape (n,) | — (doesn't exist) | Nearest centroid for each row |
| `fit_predict(X)` | `fit(X)`, then the labels of those same rows | Labels, shape (n,) | — (doesn't exist) | Same as `fit(X).labels_` |
