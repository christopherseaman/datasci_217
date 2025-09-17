# DS-223: Data Science Applications - Complete Lecture Plan

*Health Data Science Masters Program | 15-20 Self-Contained Lectures*

---

## Core Foundation Lectures (Weeks 1-4)

### 1. 🌲 **Dev Environment Setup & Python Fundamentals** 
*Prerequisites: DS-217*
- **Focus:** Environment setup demonstration, Python review, standardized workflow
- **Libraries:** `python`, `git`
- **Methods:**
  - Virtual environments (`python -m venv`, mention `uv`)
  - Git workflows (init, clone, commit, branch)
  - VS Code with Python extension and Jupyter notebook support
  - GitHub Classroom workflow demonstration
- **Dataset:** Sample patient vitals CSV
- **Key Concepts:** Environment isolation, version control basics, reproducible workflows, standardized development setup

### 2. 🌲 **Handling Larger-than-Memory Data with Polars**
*Prerequisites: Lecture 1*
- **Focus:** Lazy evaluation and efficient processing of large tabular data
- **Libraries:** `polars`
- **Methods:**
  - Lazy DataFrames (`scan_csv`, `scan_parquet`)
  - Eager DataFrames (`read_csv`, `read_parquet`)  
  - Basic operations: select, filter, groupby, join
  - Memory optimization and chunking
- **Dataset:** Large health dataset in Parquet/CSV format
- **Key Concepts:** Out-of-core processing, eager vs lazy, performance tuning

### 3. 🌲 **SQL for Data Analysis**
*Prerequisites: Lecture 2*
- **Focus:** Core SQL for data wrangling and querying
- **Libraries:** `sqlite3`, `pandas`, `sqlalchemy`
- **Methods:**
  - SELECT, FROM, WHERE, GROUP BY, ORDER BY
  - JOIN types (INNER, LEFT, RIGHT)
  - Window functions and CTEs
- **Dataset:** Health-related database (patient records)
- **Key Concepts:** Relational model, query optimization, aggregations

### 4. 🌲 **Introduction to ML & Classification**
*Prerequisites: Lecture 3 | Recommended: Complete before Lecture 5*
- **Focus:** Classification task, workflow, basic models, feature engineering, evaluation
- **Libraries:** `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Methods:**
  - Data management: `train_test_split`, `StandardScaler`, `OneHotEncoder`, SMOTE
  - Model selection: `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier`
  - Evaluation: Accuracy/Precision/Recall/F1/AUC/Confusion Matrix, `cross_val_score`, `StratifiedKFold`
  - Hyperparameter tuning: `GridSearchCV`, `RandomizedSearchCV`
  - Feature engineering: `SelectKBest`, `RFE`, feature importance analysis
  - Explainability: Feature weights, SHAP values, eli5
- **Dataset:** Classification task on health data with class imbalance and missing values
- **Key Concepts:** Bias-variance tradeoff, dataset shift, Simpson's paradox in healthcare

---

## Advanced ML & Statistical Methods (Weeks 5-9)

### 5. 🌲 **Regression & Time-Series Forecasting**
*Prerequisites: Lecture 4*
- **Focus:** Supervised learning, regression tasks, time-based splits, statistical & ML models
- **Libraries:** `scikit-learn`, `statsmodels`, `lifelines`
- **Methods:**
  - Core ML/Eval: TimeSeriesSplit, MSE, MAE, R²
  - Regression models: LinearRegression, ElasticNet, RandomForestRegressor, GradientBoostingRegressor
  - TS feature engineering: `.shift()` (lags), `.rolling()` (windows), date/time components
  - Statistical models: ARIMA/SARIMA, VAR, State Space Models, Kalman Filters
  - Advanced: Survival Analysis (Cox, Kaplan-Meier), DTW, GAMs
- **Dataset:** Panel data, irregular time series, high-frequency sensor data
- **Key Concepts:** Time series cross-validation, temporal feature engineering, forecasting evaluation

### 6. 🌲 **Neural Network Fundamentals & Applications**
*Prerequisites: Lecture 4*
- **Focus:** Building basic NNs (MLP, CNN, RNN/LSTM) using PyTorch
- **Libraries:** `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `torchvision`
- **Methods:**
  - Architecture: Define layers (`Linear`, `Conv2d`, `LSTM`), activation functions, loss functions
  - Training: Optimizers (`Adam`), `Dataset`/`DataLoader`, backpropagation
  - Regularization: Dropout, batch normalization, early stopping
  - Monitoring: TensorBoard, gradient tracking
  - Deployment: Model saving/loading, inference optimization
- **Dataset:** Tabular (MLP), MNIST/CIFAR (CNN), time series (RNN/LSTM)
- **Key Concepts:** Biological inspiration, network depth, activation functions, regularization

### 7. **Bayesian Methods & Uncertainty Quantification**
*Prerequisites: Lecture 5*
- **Focus:** Bayesian inference, probabilistic modeling, uncertainty in healthcare
- **Libraries:** `pymc`, `arviz`, `scipy.stats`, `sklearn`
- **Methods:**
  - Bayesian linear/logistic regression
  - Markov Chain Monte Carlo (MCMC)
  - Variational inference
  - Bayesian neural networks
  - Credible intervals and posterior predictive checks
- **Dataset:** Clinical trial data, diagnostic test performance
- **Key Concepts:** Prior selection, posterior inference, model uncertainty, calibration

### 8. **Causal Inference & Observational Studies**
*Prerequisites: Lecture 4*
- **Focus:** Moving beyond correlation to causation in healthcare data
- **Libraries:** `dowhy`, `causalinference`, `econml`, `statsmodels`
- **Methods:**
  - Randomized controlled trials vs observational studies
  - Propensity score matching
  - Instrumental variables
  - Difference-in-differences
  - Regression discontinuity design
- **Dataset:** Electronic health records, insurance claims data
- **Key Concepts:** Confounding, selection bias, treatment effects, counterfactuals

### 9. **Survival Analysis & Longitudinal Data**
*Prerequisites: Lecture 5*
- **Focus:** Time-to-event analysis and repeated measures in healthcare
- **Libraries:** `lifelines`, `scikit-survival`, `statsmodels`
- **Methods:**
  - Kaplan-Meier survival curves
  - Cox proportional hazards model
  - Accelerated failure time models
  - Competing risks analysis
  - Mixed-effects models for longitudinal data
- **Dataset:** Patient survival data, longitudinal biomarker measurements
- **Key Concepts:** Censoring, hazard ratios, time-varying covariates

---

## Specialized Applications (Weeks 10-15)

### 10. 🌲 **Deep Learning: LLMs & Transformers**
*Prerequisites: Lecture 6*
- **Focus:** Practical NLP via APIs and Hugging Face, local LLM options
- **Libraries:** `transformers`, `openai`, `anthropic`, `cohere`, `ollama`
- **Methods:**
  - Core concepts: Tokenization, attention mechanisms, embeddings
  - Model usage: `pipeline`, `AutoTokenizer`, `AutoModelForSequenceClassification`
  - API integration: REST APIs, async calls, rate limiting
  - Fine-tuning: Transfer learning, prompt engineering
- **Dataset:** Medical text data, clinical notes, research papers
- **Key Concepts:** Transformer architecture, context windows, prompt engineering, PHI handling

### 11. 🌲 **Computer Vision for Healthcare**
*Prerequisites: Lecture 6*
- **Focus:** CNNs for medical imaging (PyTorch), classification, detection concepts
- **Libraries:** `torch`, `torchvision`, `PIL`, `opencv-python`, `monai`
- **Methods:**
  - Image processing: Loading, transforming, augmenting medical images
  - Model architecture: Pre-trained models, transfer learning, fine-tuning
  - Tasks: Classification, object detection, segmentation
  - Advanced: Feature visualization, attention maps, federated learning
- **Dataset:** Medical imaging (X-rays, MRIs, microscopy, pathology slides)
- **Key Concepts:** CNN architectures (ResNet, U-Net, YOLO), transfer learning, medical image preprocessing

### 12. **Genomics & Bioinformatics**
*Prerequisites: Lecture 2*
- **Focus:** Working with genomic data, sequence analysis, variant calling
- **Libraries:** `biopython`, `pysam`, `pandas`, `numpy`, `matplotlib`
- **Methods:**
  - Sequence manipulation and analysis
  - File format handling (FASTA, FASTQ, VCF, BAM)
  - Genome-wide association studies (GWAS)
  - Phylogenetic analysis
  - Gene expression analysis
- **Dataset:** Genomic sequences, variant call files, gene expression matrices
- **Key Concepts:** DNA/RNA/protein sequences, genetic variants, population genetics

### 13. **Epidemiological Modeling**
*Prerequisites: Lecture 8*
- **Focus:** Disease spread modeling, population health analytics
- **Libraries:** `scipy`, `numpy`, `matplotlib`, `networkx`
- **Methods:**
  - SIR/SEIR compartmental models
  - Agent-based modeling
  - Network epidemiology
  - Spatial epidemiology with GIS
  - Monte Carlo simulation
- **Dataset:** Disease surveillance data, contact networks, geographic health data
- **Key Concepts:** Basic reproduction number, herd immunity, outbreak investigation

### 14. **Health Economics & Outcomes Research**
*Prerequisites: Lecture 4*
- **Focus:** Economic evaluation, cost-effectiveness analysis, real-world evidence
- **Libraries:** `lifelines`, `scipy.stats`, `matplotlib`, `pandas`
- **Methods:**
  - Cost-effectiveness analysis
  - Markov modeling for health states
  - Budget impact analysis
  - Propensity score methods
  - Health technology assessment
- **Dataset:** Claims data, clinical trial economic endpoints
- **Key Concepts:** QALYs, ICERs, willingness-to-pay thresholds

### 15. **Digital Health & Wearables**
*Prerequisites: Lecture 5*
- **Focus:** Processing sensor data, mobile health applications, IoT in healthcare
- **Libraries:** `scipy.signal`, `numpy`, `pandas`, `matplotlib`
- **Methods:**
  - Signal processing for physiological data
  - Activity recognition from accelerometry
  - Heart rate variability analysis
  - Sleep stage classification
  - Anomaly detection in continuous monitoring
- **Dataset:** Wearable device data, smartphone sensor data, continuous glucose monitoring
- **Key Concepts:** Digital biomarkers, remote monitoring, patient engagement

---

## Integration & Communication (Weeks 16-20)

### 16. 🌲 **Data Visualization & Reporting**
*Prerequisites: Any 3 domain lectures*
- **Focus:** Communicating insights effectively using diagrams, interactive visualizations, reports, dashboards
- **Libraries:** `altair`, `mermaid`, `mkdocs`, `streamlit`, `plotly`
- **Methods:**
  - Diagramming as code (Mermaid)
  - Declarative visualization (Altair-Vega)
  - Automated report generation (MkDocs)
  - Interactive dashboards (Streamlit)
- **Dataset:** PhysioNet databases, adaptable for reports and dashboards
- **Key Concepts:** Data storytelling, static site generation, interactive web applications

### 17. 🌲 **Experimentation & A/B Testing**
*Prerequisites: Lecture 4*
- **Focus:** Experimental design, A/B testing, causal inference, variance reduction
- **Libraries:** `scipy.stats`, `statsmodels`, `pandas`, `numpy`
- **Methods:**
  - A/B testing design (`ttest_ind`, `proportions_ztest`)
  - CUPED variance reduction, sequential testing
  - GLM analysis, multiple comparison corrections
  - Power analysis and sample size calculation
- **Dataset:** Clinical trial data, digital health intervention studies
- **Key Concepts:** Experimental design principles, multiple testing, ethical considerations

### 18. **MLOps & Production Systems**
*Prerequisites: Multiple ML lectures*
- **Focus:** Deploying ML models, monitoring, CI/CD for data science
- **Libraries:** `mlflow`, `docker`, `fastapi`, `great_expectations`
- **Methods:**
  - Model versioning and experiment tracking
  - Containerization and deployment
  - Model monitoring and drift detection
  - Data quality testing and validation
  - CI/CD pipelines for ML
- **Dataset:** Production-ready healthcare ML pipeline
- **Key Concepts:** Model lifecycle management, reproducibility, scalability

### 19. **Ethics, Privacy & Regulatory Compliance**
*Prerequisites: Any 2 domain lectures*
- **Focus:** HIPAA, bias in healthcare AI, fairness, transparency
- **Libraries:** `fairlearn`, `aif360`, `privacy-preserving libraries`
- **Methods:**
  - Bias detection and mitigation
  - Differential privacy
  - Federated learning
  - Explainable AI techniques
  - De-identification methods
- **Dataset:** Synthetic healthcare data with bias examples
- **Key Concepts:** Healthcare data governance, algorithmic fairness, regulatory frameworks

### 20. **Capstone Project Presentation**
*Prerequisites: All previous lectures*
- **Focus:** End-to-end data science project lifecycle
- **Libraries:** Integration of tools from entire course
- **Methods:**
  - Project lifecycle stages: problem definition, EDA, preprocessing, modeling, evaluation, reporting
  - Applying research design principles
  - Version control and reproducibility
  - Professional presentation of results
- **Dataset:** Student-selected healthcare dataset (e.g., MIMIC-IV subset)
- **Key Concepts:** CRISP-DM process, reproducibility, ethical considerations, communication

---

## Lecture Dependencies & Recommendations

**Essential Prerequisites:**
- Lectures 1-4 should be completed before any specialized applications
- Lecture 4 (Intro to ML) is prerequisite for most advanced ML lectures (5-11)
- Lecture 6 (Neural Networks) is prerequisite for deep learning applications (10-11)

**Suggested Pedagogical Ordering:**
1. **Foundation Block:** Lectures 1-4 (sequential)
2. **Methods Block:** Lectures 5-9 (flexible order, but 5 before 7)
3. **Applications Block:** Lectures 10-15 (flexible based on student interests)
4. **Integration Block:** Lectures 16-20 (16-17 can be done earlier, 18-20 should be near end)