# Machine Learning Reference Tables

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
| A `np.random.RandomState` instance | Shares one generator across several steps (a NumPy `Generator` is not accepted) | Advanced pipelines |

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

## Session 4: Dimensionality reduction with PCA and t-SNE

### PCA: key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `n_components` | `None` | How many components to keep. An integer keeps exactly that many; a float between 0 and 1 (e.g. `0.95`) keeps as many as needed to explain that share of the variance; `None` keeps all of them, `min(n_samples, n_features)`; `"mle"` estimates the number automatically |
| `whiten` | `False` | If `True`, rescales each component to unit variance. Can help some downstream models, but loses the relative variances |
| `svd_solver` | `"auto"` | The algorithm used to compute the components. `"auto"` chooses based on the data size |
| `random_state` | `None` | Only used by the randomized solvers (`"randomized"`, `"arpack"`). Otherwise PCA is deterministic |

### PCA: what `fit` learns

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `components_` | (n_components, n_features) | The **loadings**: each row is one principal component, written as weights on the original features |
| `explained_variance_ratio_` | (n_components,) | Share of the total variance captured by each component, largest first |
| `explained_variance_` | (n_components,) | Variance captured by each component, in the units of the (scaled) data |
| `singular_values_` | (n_components,) | Singular values of the centred data, linked to the explained variance |
| `mean_` | (n_features,) | The feature means subtracted before projecting. PCA centres the data but does **not** scale it |
| `n_components_` | integer | The number of components actually kept, useful when `n_components` is a float or `"mle"` |

### Choosing the number of components

| Goal | Typical choice | How to decide |
| --- | --- | --- |
| Visualise the data | 2 (or 3) | Fixed by the plot; check how much variance the plot shows with `explained_variance_ratio_.sum()` |
| Compress features before a model or clustering | Enough for 90–95% of the variance | `n_components=0.95`, or the cumulative curve `explained_variance_ratio_.cumsum()` |
| Find the "natural" dimensionality | Where the scree plot flattens | Plot `explained_variance_ratio_` and look for the elbow |
| Remove noise | Drop the last components | Keep the components that carry clear structure; reconstruct with `inverse_transform` |

### Reading PCA loadings

| What you see in `components_` | What it means |
| --- | --- |
| A large positive or negative weight on a feature | That feature strongly drives the component |
| A weight close to 0 | That feature hardly matters for this component |
| All features with the same sign | The component measures an overall level (e.g. "strong in every subject") |
| Two groups of features with opposite signs | The component is a trade-off between the groups (e.g. sciences vs humanities) |
| The whole row's signs flipped | The same component pointing the other way. The overall sign is arbitrary; only signs relative to each other matter |

### t-SNE: key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `n_components` | `2` | Dimensions of the map. Use 2 or 3; the default `"barnes_hut"` method only supports fewer than 4 |
| `perplexity` | `30.0` | Roughly how many neighbours each point pays attention to. Typical range 5–50; must be **smaller than the number of samples** |
| `init` | `"pca"` | Starting layout. `"pca"` is more stable and keeps the global layout better than `"random"` |
| `random_state` | `None` | The optimisation is random; set it to reproduce the same map |
| `max_iter` | `1000` | Number of optimisation iterations (called `n_iter` in versions before 1.5) |
| `learning_rate` | `"auto"` | Step size of the optimisation; `"auto"` adapts it to the data size |
| `early_exaggeration` | `12.0` | How strongly natural groups are pulled apart early in the optimisation |
| `metric` | `"euclidean"` | Distance used between points in the original space |

After fitting, `embedding_` holds the map coordinates and `kl_divergence_` the final mismatch between the original neighbourhoods and the map (lower is better, comparable only between runs on the same data).

### The effect of perplexity

| Perplexity | Each point pays attention to | What the map tends to show |
| --- | --- | --- |
| Low (2–5) | Its few nearest neighbours | Very local detail; real groups can break into separate fragments |
| Medium (5–50) | A neighbourhood | Local groups with some larger structure. The usual range |
| High (close to the number of samples) | A large part of the data | Broad structure; small groups can merge |

Structure that appears at several perplexities is real. Structure that appears at only one value is often an artefact.

### What you can and can't read from a t-SNE map

| On the map | Can you trust it? | Why |
| --- | --- | --- |
| Which points are neighbours | Yes | Preserving neighbourhoods is exactly what t-SNE optimises |
| That separate groups exist | Mostly | Check with several perplexities and seeds |
| The size of a cluster | No | t-SNE expands dense clusters and shrinks sparse ones |
| The distance between two clusters | No | Distances between groups are not preserved |
| The density of points | No | t-SNE evens out density |
| The axes | No | They have no units and no meaning, unlike PCA components |

### PCA vs t-SNE

| | PCA | t-SNE |
| --- | --- | --- |
| Type | Linear | Non-linear |
| Preserves | Global variance | Local neighbourhoods |
| Main hyperparameter | `n_components` | `perplexity` |
| Same result every run | Yes | Only with a fixed `random_state` |
| Can transform new data | Yes: `transform`, `inverse_transform` | No: only `fit_transform` |
| Axes have a meaning | Yes: read the loadings | No |
| Speed on large data | Fast | Slow; run PCA to about 50 dimensions first |
| Typical use | Preprocessing, compression, a first look at the data | Visualisation only |
| Use its output as model features | Yes, inside a `Pipeline` | No |

## Session 5: Density-based and hierarchical clustering

### DBSCAN: key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `eps` | `0.5` | The neighbourhood radius: how close two points must be to count as neighbours. Measured in the units of the (scaled) data and the chosen metric |
| `min_samples` | `5` | Points needed within `eps` for a point to be a core point, **counting the point itself** |
| `metric` | `"euclidean"` | The distance used, e.g. `"manhattan"` or `"cosine"`, or `"precomputed"` for your own distance matrix |
| `algorithm` | `"auto"` | How neighbours are searched (brute force, KD-tree, ball tree). Changes speed, never the result |

### DBSCAN: the three kinds of point

| Point type | Rule | Role | Label |
| --- | --- | --- | --- |
| Core point | At least `min_samples` points within `eps`, itself included | Starts and grows a cluster | Cluster number |
| Border point | Not a core point, but within `eps` of a core point | Joins a cluster but doesn't extend it | Cluster number |
| Noise point | Neither core nor border | Belongs to no cluster | `-1` |

### What happens when you change the DBSCAN parameters

| Change | Number of clusters | Noise points | Risk |
| --- | --- | --- | --- |
| Larger `eps` | Fewer | Fewer | Separate clusters merge; at the extreme, everything is one cluster |
| Smaller `eps` | More, smaller | More | Real clusters break into fragments, most points become noise |
| Larger `min_samples` | Fewer | More | Only the densest regions remain clusters |
| Smaller `min_samples` | More | Fewer | Chains of sparse points form clusters |

### Choosing DBSCAN parameters

| Parameter | Rule of thumb | Tool |
| --- | --- | --- |
| `min_samples` | At least the number of features + 1; often 2 × the number of features; larger for noisy data | Domain knowledge: the smallest group you'd still call a cluster |
| `eps` | The distance at the knee of the k-distance graph | `NearestNeighbors(n_neighbors=min_samples)`, then sort each point's distance to its last neighbour and plot it. Points below the chosen `eps` become core points |
| Scaling | Always scale first | `StandardScaler` in a `Pipeline`, because `eps` is one distance for all features |

### HDBSCAN: key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `min_cluster_size` | `5` | The smallest group that counts as a cluster; smaller groups become noise. The main parameter to tune |
| `min_samples` | `None` (same as `min_cluster_size`) | How conservative the clustering is: larger values label more points as noise |
| `cluster_selection_epsilon` | `0.0` | Merges clusters closer than this distance. Useful to avoid many tiny clusters |
| `cluster_selection_method` | `"eom"` | `"eom"` keeps the most stable, usually larger clusters; `"leaf"` gives more, smaller, more uniform clusters |
| `metric` | `"euclidean"` | The distance used |
| `store_centers` | `None` | Set to `"centroid"`, `"medoid"` or `"both"` to store `centroids_` and/or `medoids_` |

After fitting, `labels_` holds the clusters (`-1` = noise) and `probabilities_` how strongly each point belongs to its cluster, from 0 to 1 (noise = 0).

### DBSCAN vs HDBSCAN

| | DBSCAN | HDBSCAN |
| --- | --- | --- |
| Main parameters | `eps`, `min_samples` | `min_cluster_size` (and `min_samples`) |
| Needs a distance threshold | Yes: `eps` | No |
| Clusters of different densities | Struggles: one `eps` can't fit all | Handles them |
| Noise label | `-1` | `-1` |
| Membership strength per point | No | Yes: `probabilities_` |
| `predict` on new points | No | No |
| Available since | Long-standing | scikit-learn 1.3 |
| Prefer it when | You know a meaningful distance threshold and densities are similar | Densities vary, or you don't know a good `eps` |

### AgglomerativeClustering: key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `n_clusters` | `2` | Stop merging at this many clusters. Set to `None` when using `distance_threshold` |
| `distance_threshold` | `None` | Cut the tree at this height instead. Exactly one of `n_clusters` and `distance_threshold` must be set |
| `linkage` | `"ward"` | How the distance between two clusters is measured (see the next table) |
| `metric` | `"euclidean"` | The distance between points. Replaced `affinity`, which was removed in scikit-learn 1.4 |
| `compute_distances` | `False` | Store the merge heights in `distances_` even when `distance_threshold` isn't used, e.g. to draw a dendrogram |

After fitting: `labels_`, `n_clusters_` (useful with `distance_threshold`), `children_` (which clusters merged at each step) and, when computed, `distances_`. To draw the dendrogram, use SciPy: `scipy.cluster.hierarchy.linkage` and `dendrogram`.

### Linkage methods

| Linkage | Distance between two clusters | Cluster shapes it finds | Weakness | Metrics allowed |
| --- | --- | --- | --- | --- |
| `ward` | The merge that increases the within-cluster variance the least | Compact, round, similar sizes; closest to K-means | Can't follow long or curved shapes | Euclidean only |
| `complete` | The farthest pair of points | Compact clusters of similar diameter | Sensitive to outliers | Any |
| `average` | The average distance over all pairs | A compromise between single and complete | Still affected by outliers, though less than `complete` | Any |
| `single` | The closest pair of points | Long, curved or chain-like shapes | One stray point can bridge two clusters (chaining) | Any |

### Reading a dendrogram

| What you see | What it means |
| --- | --- |
| The leaves at the bottom | The individual data points |
| A horizontal bar | A merge of two clusters |
| The height of a bar | How far apart the two clusters were when they merged |
| A point that joins very high up | An outlier |
| A long vertical stretch with no merges | A large gap: the clusters below it are well separated |
| A horizontal cut | The number of vertical lines it crosses is the number of clusters. Cut through the longest vertical stretches |

### Distance metrics

| Metric | Measures | Name in scikit-learn | Typical use |
| --- | --- | --- | --- |
| Euclidean | The straight-line distance | `"euclidean"` or `"l2"` | The default almost everywhere |
| Manhattan | The sum of the absolute differences along each feature | `"manhattan"`, `"l1"` or `"cityblock"` | Grid-like data; less affected by one large difference in a single feature |
| Cosine | The angle between two vectors, ignoring their length | `"cosine"` | Text and other data where proportions matter more than size |
| Precomputed | Your own distance matrix | `"precomputed"` | Travel times, custom similarities |

Which estimators let you change the metric: `DBSCAN`, `HDBSCAN`, `AgglomerativeClustering` (not with `ward`) and `TSNE`. `KMeans` always uses Euclidean distance.

### Clustering algorithms compared

| | `KMeans` | `DBSCAN` | `HDBSCAN` | `AgglomerativeClustering` |
| --- | --- | --- | --- | --- |
| Core idea | Assign points to the nearest centre | Grow clusters through dense regions | DBSCAN at every density level; keep the stable clusters | Repeatedly merge the two closest clusters |
| You choose | `n_clusters` | `eps`, `min_samples` | `min_cluster_size` | `n_clusters` or `distance_threshold`, plus `linkage` |
| Number of clusters | An input | Found from the data | Found from the data | An input, or read from the cut |
| Cluster shapes | Round, similar sizes | Any shape | Any shape, different densities | Depends on the linkage |
| Outliers | Forced into a cluster | Labelled `-1` | Labelled `-1` | Join late; become tiny clusters |
| `predict` on new points | Yes | No | No | No |
| `random_state` | Yes | No | No | No |
| Centroids | `cluster_centers_` | No | Optional (`store_centers`) | No |
| Large datasets | Fast (or `MiniBatchKMeans`) | Fairly fast | Fairly fast | Slow: memory grows with the square of the number of rows |
| Needs scaling | Yes | Yes | Yes | Yes |

### Which algorithm for which situation

| Situation | Good choice | Why |
| --- | --- | --- |
| You know the number of groups and expect compact clusters | `KMeans` | Simple, fast, assigns new points with `predict` |
| Clusters have irregular shapes and there is noise | `DBSCAN` | Follows dense regions and labels noise as `-1` |
| Clusters have very different densities | `HDBSCAN` | No single `eps` is needed |
| You want to find anomalies | `DBSCAN` or `HDBSCAN` | Noise points are the anomaly candidates |
| You want a hierarchy, or to choose the number of clusters afterwards | `AgglomerativeClustering` | The dendrogram shows every level |
| Millions of rows | `KMeans` or `MiniBatchKMeans` | The others become slow or memory-hungry |
| Many features | PCA first, then cluster | Distances become less informative in many dimensions |

### Methods available on each estimator

| Estimator | `fit` | `transform` | `fit_transform` | `inverse_transform` | `predict` | `fit_predict` |
| --- | --- | --- | --- | --- | --- | --- |
| `StandardScaler` | Yes | Yes | Yes | Yes | — | — |
| `PCA` | Yes | Yes | Yes | Yes | — | — |
| `TSNE` | Yes | — | Yes | — | — | — |
| `KMeans` | Yes | Yes (distances to centroids) | Yes | — | Yes | Yes |
| `DBSCAN` | Yes | — | — | — | — | Yes |
| `HDBSCAN` | Yes | — | — | — | — | Yes |
| `AgglomerativeClustering` | Yes | — | — | — | — | Yes |

Checked against scikit-learn 1.9.
