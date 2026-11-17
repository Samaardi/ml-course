# Machine Learning Course
 
Study materials for a machine learning course covering unsupervised learning (UML) and supervised learning (SML), and preparing for the **scikit-learn Associate Practitioner** certification by Probabl.
 
## 👉 [Open the flashcards](https://samaardi.github.io/ml-course/)
 
## 📊 [Machine Learning Reference Tables](Machine%20Learning%20Reference%20Tables.md)
 
The key comparison tables from each lesson, collected in one place.
 
## What's inside
 
350 flashcards in 25 decks, grouped by topic:

- **Foundations**: what ML is, unsupervised learning, algorithms, randomness
- **Preprocessing**: scaling, imputation, one-hot and ordinal encoding, `ColumnTransformer` and pipelines
- **Clustering**: K-means, evaluation, DBSCAN, HDBSCAN, hierarchical clustering, choosing a method
- **Dimensionality reduction**: PCA, t-SNE
- **Supervised learning and evaluation**: baselines, classification metrics, cross-validation and grid search, bias–variance and regularization
- **Classification algorithms**: decision trees, KNN, SVM, logistic regression, random forests
- **Regression**: regression metrics, linear regression, collinearity, Ridge and Lasso

Every card was checked against scikit-learn 1.9. New decks are added as the course progresses.
 
## How to study
 
1. Choose decks in the list on the left (on a phone, tap **Choose decks**), or click a cluster in the progress map at the top to study one deck.
2. Read the question and answer it in your head.
3. Tap the card (or press **Space**) to flip it.
4. Be honest: click **I knew it** or **Review again**.
5. Later, turn on **Only "review again"** to practise the cards you missed.
Keyboard shortcuts: **Space** flip · **← →** move · **1** review again · **2** I knew it
 
Your progress is saved in your own browser only. Use the same browser and device each time, or your marks won't be there.
 
## About the exam

*Last checked: 17 November 2026. Probabl has moved its certification pages to its learning platform **Skolar**. The old links (`certification.probabl.ai/scikit-learn-associate` and `probabl.ai/associate-certification`) no longer work.*

The **scikit-learn® Associate Practitioner** certification is the entry level of the official scikit-learn certification programme run by **Probabl**, the company that operates the scikit-learn brand. The programme has three levels: Associate, Professional and Expert. The certifications were developed with scikit-learn core contributors.

### At a glance

| | |
|---|---|
| **Level** | Associate: for aspiring machine learning practitioners with working knowledge of Python |
| **Price** | Starting from **299 USD** (shown on Skolar) |
| **Where to buy** | [Skolar](https://skolar.probabl.ai/en/), Probabl's training and certification platform |
| **Length** | 120 minutes |
| **Questions** | Theory questions and practical questions based on business use cases, including a programming exercise. Probabl's last published format: 30 to 35 multiple-choice questions, theory questions worth 1–2 points and practical questions worth 2–3 points |
| **Passing score** | 70% |
| **Delivery** | Online, watched live by a remote proctor, through Kryterion's Webassessor |
| **Allowed during the exam** | Only a JupyterLite notebook and the scikit-learn documentation. No other websites, software or help |
| **When you pass** | A digital badge on [Credly](https://www.credly.com/org/probabl/badge/scikit-learn-associate-practitioner-certification) that you can share on LinkedIn |

Probabl's public pages don't state the retake policy, how long the certificate stays valid, the ID requirements or the exam languages. Check these on Skolar or in your booking confirmation, or ask Probabl through its [contact page](https://www.probabl.ai/contact/).

### How to book

1. Go to [skolar.probabl.ai](https://skolar.probabl.ai/en/) and create a free account (**Sign in**, then **Sign up**).
2. Open **Scikit-learn Associate** in the menu and choose its last item, **Scikit-learn® Associate Practitioner Certification**. The page shows the price (from 299 USD); sign in to see the purchase options.
3. The exam itself runs on Kryterion's **Webassessor**. Follow the instructions you receive after buying to choose a date and time.
4. Before exam day, run Kryterion's [online exam system check](http://www.kryteriononline.com/systemcheck) on the computer you will use (webcam, microphone, browser).
5. Watch Skolar's short video **"What the official scikit-learn certification looks like"** (in the same menu). It walks through setup, the first questions, a programming exercise and the final review.
6. Read Probabl's [Exam Integrity Policy](https://certification.probabl.ai/exam-integrity-policy). Proctors can end the exam if they see unauthorised resources, and certifications can be withdrawn later for misconduct or for sharing exam content.

### What the exam covers

The official description on Credly says the certification "validates foundational skills in machine learning with scikit-learn". It covers supervised learning, train-test splits, overfitting, data preprocessing and building linear models, plus cross-validation, hyperparameter tuning and comparing models with baselines, applied to real-world datasets with scikit-learn pipelines.

The detailed list below follows Skolar's official **Associate** preparation track. It has five modules, and their content is public in Probabl's [course repository](https://github.com/probabl-ai/scikit-learn-course). After the modules come the models, metrics and tools you should know, and a final checklist from Probabl's earlier study guide.

#### Module 1. Machine learning concepts

- What machine learning is, and typical business and science applications
- **Memorization vs generalization**: why a model must be judged on data it has not seen (train/test split)
- **Learning from data vs hand-written rules**: a simple rule can be a baseline
- **Supervised vs unsupervised** learning. Supervised: data `X` with a target `y`. Unsupervised: structure without a target, such as clustering or dimensionality reduction
- **Classification vs regression**: category target vs number target
- Vocabulary: **samples** (rows, records), **features** (columns), **target**, the data matrix
- Typical exam task: given a business case, decide whether it is supervised or unsupervised, classification or regression, and name the features, the target and a record

#### Module 2. The predictive modeling pipeline

**Exploring tabular data**
- Loading data with pandas (`read_csv`; the earlier study guide also mentions parquet files)
- Telling numerical from categorical columns, and spotting wrongly encoded columns
- Looking at distributions and relationships with pandas, matplotlib and seaborn
- Checking the class balance of the target

**A first model with numerical data**
- The estimator API: create a model, `fit(X, y)`, `predict(X)`, `score(X, y)`
- `train_test_split`, with a fixed `random_state`
- First models: `KNeighborsClassifier` and `LogisticRegression`
- Comparing against a baseline: `DummyClassifier`
- Scaling numerical features with `StandardScaler`, and why scaling matters for some models (KNN, linear models) and not others (trees)
- Chaining steps with `make_pipeline` and `Pipeline`, and why a pipeline prevents data leakage
- A first look at cross-validation with `cross_validate`

**Categorical data**
- `OrdinalEncoder`: categories become integers, which implies an order
- `OneHotEncoder`: one 0/1 column per category; `handle_unknown="ignore"` for categories missing from the training data
- Which encoder suits which model: one-hot for linear models, ordinal for tree-based models

**Mixed data types**
- `ColumnTransformer` / `make_column_transformer`: different preprocessing for numerical and categorical columns
- `make_column_selector`: choosing columns by data type
- Filling missing values with `SimpleImputer`
- A complete pipeline: preprocessing + `LogisticRegression`, or ordinal encoding + `HistGradientBoostingClassifier`

#### Module 3. Selecting the best model

- **Overfitting**: the model learns noise. Low training error, high test error. Caused by a small training set, noisy data and very flexible models
- **Underfitting**: systematic errors from a model that is not flexible enough
- **Generalization**: the goal is the lowest test error, not the lowest training error
- **Cross-validation**: `cross_validate`, `cross_val_score`, `KFold` and `ShuffleSplit`; train scores vs test scores (`return_train_score=True`)
- **Validation curve** (`ValidationCurveDisplay`): train and test scores as one hyperparameter changes, such as `max_depth` of `DecisionTreeRegressor`
- **Learning curve** (`LearningCurveDisplay`): scores as the training set grows. More data reduces overfitting
- **Bias vs variance**: underfitting = high bias, overfitting = high variance. Squared error = bias² + variance + irreducible error (noise)
- **Stability**: how much results vary between folds

#### Module 4. Hyperparameter tuning

- **Hyperparameters vs parameters**: set before training (`n_neighbors`, `C`, `max_depth`, `learning_rate`) vs learned during training (`coef_`)
- Getting and setting hyperparameters: `get_params()`, `set_params()`, and the `step__parameter` names inside a pipeline
- Manual tuning with a loop and cross-validation
- **`GridSearchCV`**: every combination; `best_params_`, `best_score_`, `cv_results_`, refitting the best model
- **`RandomizedSearchCV`**: a fixed budget (`n_iter`), sampling from lists or distributions (for example log-uniform); better with many hyperparameters
- Reading search results, for example with a parallel coordinates plot
- **Nested cross-validation**: an outer loop to evaluate the tuned model without optimistic bias

#### Module 5. Unsupervised learning with clustering

- The clustering task: grouping samples without a target
- **K-means** (`KMeans`): `n_clusters`, centroids, `inertia_` (WCSS), the role of `random_state` and `n_init`, scaling before clustering
- Choosing the number of clusters: inertia (elbow) and the **silhouette score** (`silhouette_score`)
- **Cluster stability**: do the clusters change when the data is resampled?
- **Supervised clustering metrics** when true labels exist: adjusted Rand index (`adjusted_rand_score`), adjusted mutual information (`adjusted_mutual_info_score`), V-measure (`v_measure_score`)
- **Limits of k-means**: it assumes round (convex), similar-sized clusters
- **HDBSCAN** (`HDBSCAN`): non-convex shapes and noise points; `min_cluster_size`, `min_samples`
- K-means as a **feature-engineering step**: distances to centroids (`transform`) used as features in a supervised pipeline

#### Models you should be able to use

Credly lists **linear, tree-based, ensemble and neighbour-based models** among the skills tested. At Associate level you should know what each family does, when to choose it, and how to use it in a pipeline. The Professional level covers these families in depth.

| Family | Estimators | Key points |
|---|---|---|
| Baselines | `DummyClassifier`, `DummyRegressor` | Every model must beat them |
| Neighbours | `KNeighborsClassifier`, `KNeighborsRegressor` | `n_neighbors`; needs scaling |
| Linear | `LinearRegression`, `Ridge`, `LogisticRegression` | Coefficients (`coef_`); regularization (`alpha`, `C`); needs scaling |
| Tree-based | `DecisionTreeClassifier`, `DecisionTreeRegressor` | `max_depth`; no scaling needed; overfit easily |
| Ensembles | `RandomForestClassifier`/`Regressor`, `HistGradientBoostingClassifier`/`Regressor` | Many trees combined; strong default choice for tabular data |
| Clustering | `KMeans`, `HDBSCAN` | See module 5 |

#### Metrics you should know

- **Classification**: accuracy, precision, recall, F1 score, confusion matrix
- **Regression**: mean absolute error, mean squared error, R²
- **Clustering**: inertia, silhouette score, ARI, AMI, V-measure
- Scoring names in `cross_validate` and the searches, for example `scoring="neg_mean_absolute_error"`: errors are negated so that higher is always better

#### Checklist from Probabl's earlier study guide

Probabl's earlier study guide (online until autumn 2026) organised the exam into five sections. Everything in it is covered by the modules above:

| Section | Topics | Our flashcard decks |
|---|---|---|
| 1. Machine learning concepts | Supervised, unsupervised and semi-supervised learning; model families (linear, tree-based, ensemble, neighbours); features, labels, train and test sets; overfitting and underfitting; bias/variance trade-off | What is ML?, Unsupervised learning, Supervised learning basics, Bias–variance |
| 2. Data preprocessing | Loading parquet files; basic plots; spotting wrongly encoded columns; `SimpleImputer`; feature scaling (`StandardScaler`, `MinMaxScaler`); `OrdinalEncoder` and `OneHotEncoder`; `ColumnTransformer` | Scaling; Imputation, encoding and pipelines |
| 3. Model building and evaluation | `train_test_split`, `fit()`, `predict()`; accuracy, precision, recall, F1, confusion matrix, mean squared error, R²; comparing with dummy models | Supervised learning basics, Classification metrics, Regression basics, the classifier decks |
| 4. Model selection and validation | Cross-validation (`KFold`, `ShuffleSplit`); learning and validation curves; `GridSearchCV` and `RandomizedSearchCV`; stability of coefficients | Cross-validation and tuning, Bias–variance, Linear regression |
| 5. Interpretation and communication | Visualizing results with matplotlib and seaborn; explaining models and metrics to non-technical stakeholders | Practised in class projects |

**Tip:** clustering is part of the Associate track, so don't skip the clustering decks.

### Free preparation material

- **Skolar's Associate course**: the official preparation track; free with a Skolar account
- [Course content on GitHub](https://github.com/probabl-ai/scikit-learn-course): the material behind the Skolar course
- [Inria's scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/): the free course the Skolar track is based on
- [scikit-learn documentation](https://scikit-learn.org/stable/): practise finding things in it quickly, because it is the only reference allowed in the exam

## Sources

Checked on 17 November 2026:

- [Skolar](https://skolar.probabl.ai/en/): Associate track, certification page and price
- [Probabl's scikit-learn course repository](https://github.com/probabl-ai/scikit-learn-course): the content of the Skolar modules (last updated September 2026)
- [Credly badge page](https://www.credly.com/org/probabl/badge/scikit-learn-associate-practitioner-certification): description, exam length, passing score and Webassessor
- [Probabl Exam Integrity Policy](https://certification.probabl.ai/exam-integrity-policy): allowed resources and online proctoring
- [Probabl's certification launch post](https://www.probabl.ai/blog/official-scikit-learn-certification-launch/): the three levels, Kryterion, Credly
- [Probabl's 2024 certification challenge page](https://hello.probabl.ai/ai-action-summit-2024-scikit-learn-certification-challenge): question count and point values
- [Kryterion: scheduling an online proctored exam](https://support.kryterion.com/en_us/scheduleregister-an-onlineremote-proctor-exam-ByN6SIkjyx.md)

Exam details can change. Check the certification page on Skolar before you book.

These flashcards are a study aid and are not affiliated with or endorsed by Probabl or the scikit-learn project.

## Repository contents
 
| File | Purpose |
|---|---|
| `index.html` | The whole flashcard website: page, styles, cards and logic in one file |
| `Machine Learning Reference Tables.md` | The key comparison tables from the lessons |
| `README.md` | This file |
