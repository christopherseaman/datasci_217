# Lecture 08: Statistical Analysis + Machine Learning Foundations

**Duration**: 5.5 hours  
**Level**: Advanced Professional Track  
**Prerequisites**: L06-L07 Advanced Data Processing & Visualization

## Professional Context: From Descriptive to Predictive Analytics

The transition from descriptive statistics to predictive modeling represents the evolution from "what happened?" to "what will happen?" in clinical and business contexts. Professional data scientists must master:

- **Statistical modeling**: Hypothesis testing, regression, survival analysis
- **Machine learning pipelines**: Feature engineering, model selection, validation  
- **Model interpretation**: Explaining complex models to clinical stakeholders
- **Production deployment**: Models that work reliably in real-world systems

In clinical research and healthcare analytics, this means building models that:
- **Predict patient outcomes** with quantified uncertainty
- **Identify treatment responders** before therapy initiation
- **Support clinical decision-making** with interpretable insights
- **Meet regulatory standards** for medical device approval

Today we master the statistical and ML foundations that transform data scientists into trusted analytical partners.

## Learning Objectives

By the end of this lecture, you will:

1. **Design and implement sophisticated statistical models** for clinical research
2. **Build end-to-end machine learning pipelines** with proper validation frameworks  
3. **Create interpretable models** that support clinical decision-making
4. **Apply time series analysis** for forecasting and trend detection
5. **Integrate domain expertise** with advanced analytical techniques
6. **Deploy models** using professional software engineering practices

---

## Part 1: Advanced Statistical Modeling (90 minutes)

### Professional Challenge: Clinical Research Statistical Analysis

Statistical modeling in clinical research requires rigorous methodology:
- **Study design considerations**: Power analysis, sample size, randomization
- **Multiple testing corrections**: Controlling family-wise error rates
- **Survival analysis**: Time-to-event outcomes with censoring
- **Hierarchical models**: Accounting for site effects, repeated measures

### Comprehensive Statistical Framework

#### 1. Advanced Regression Modeling

```python
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ClinicalStatisticalAnalyzer:
    """Advanced statistical analysis for clinical research."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        
    def perform_comprehensive_analysis(self, df: pd.DataFrame, 
                                     primary_endpoint: str,
                                     treatment_col: str,
                                     covariates: list) -> dict:
        """Perform comprehensive statistical analysis for clinical trial."""
        
        analysis_results = {}
        
        # 1. Descriptive statistics by treatment group
        analysis_results['descriptives'] = self._descriptive_analysis(
            df, primary_endpoint, treatment_col, covariates
        )
        
        # 2. Primary efficacy analysis
        analysis_results['primary_analysis'] = self._primary_endpoint_analysis(
            df, primary_endpoint, treatment_col
        )
        
        # 3. Adjusted analysis with covariates
        analysis_results['adjusted_analysis'] = self._adjusted_analysis(
            df, primary_endpoint, treatment_col, covariates
        )
        
        # 4. Subgroup analyses
        analysis_results['subgroup_analyses'] = self._subgroup_analyses(
            df, primary_endpoint, treatment_col, ['age_group', 'gender', 'site_id']
        )
        
        # 5. Safety analysis
        if 'adverse_event' in df.columns:
            analysis_results['safety_analysis'] = self._safety_analysis(df, treatment_col)
        
        return analysis_results
    
    def _descriptive_analysis(self, df: pd.DataFrame, 
                            endpoint: str, 
                            treatment_col: str, 
                            covariates: list) -> pd.DataFrame:
        """Comprehensive descriptive statistics by treatment group."""
        
        # Identify numeric and categorical variables
        numeric_vars = df[covariates + [endpoint]].select_dtypes(include=[np.number]).columns
        categorical_vars = df[covariates].select_dtypes(exclude=[np.number]).columns
        
        descriptives = []
        
        # Process numeric variables
        for var in numeric_vars:
            for treatment in df[treatment_col].unique():
                treatment_data = df[df[treatment_col] == treatment][var].dropna()
                
                descriptives.append({
                    'variable': var,
                    'treatment': treatment,
                    'n': len(treatment_data),
                    'mean': treatment_data.mean(),
                    'std': treatment_data.std(),
                    'median': treatment_data.median(),
                    'q25': treatment_data.quantile(0.25),
                    'q75': treatment_data.quantile(0.75),
                    'min': treatment_data.min(),
                    'max': treatment_data.max(),
                    'missing': df[df[treatment_col] == treatment][var].isna().sum(),
                    'variable_type': 'numeric'
                })
        
        # Process categorical variables
        for var in categorical_vars:
            for treatment in df[treatment_col].unique():
                treatment_data = df[df[treatment_col] == treatment]
                value_counts = treatment_data[var].value_counts()
                total_n = len(treatment_data[var].dropna())
                
                for value, count in value_counts.items():
                    descriptives.append({
                        'variable': f"{var}_{value}",
                        'treatment': treatment,
                        'n': total_n,
                        'count': count,
                        'percentage': (count / total_n * 100) if total_n > 0 else 0,
                        'missing': treatment_data[var].isna().sum(),
                        'variable_type': 'categorical'
                    })
        
        return pd.DataFrame(descriptives)
    
    def _primary_endpoint_analysis(self, df: pd.DataFrame, 
                                 endpoint: str, 
                                 treatment_col: str) -> dict:
        """Primary endpoint analysis with effect size and confidence intervals."""
        
        results = {}
        
        # Get treatment groups
        treatments = df[treatment_col].unique()
        
        if len(treatments) == 2:
            # Two-sample analysis
            group1_data = df[df[treatment_col] == treatments[0]][endpoint].dropna()
            group2_data = df[df[treatment_col] == treatments[1]][endpoint].dropna()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                (len(group2_data) - 1) * group2_data.var()) / 
                               (len(group1_data) + len(group2_data) - 2))
            
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            # Calculate confidence interval for mean difference
            mean_diff = group1_data.mean() - group2_data.mean()
            se_diff = pooled_std * np.sqrt(1/len(group1_data) + 1/len(group2_data))
            df_t = len(group1_data) + len(group2_data) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df_t)
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            results = {
                'test_type': 'two_sample_t_test',
                'group1_n': len(group1_data),
                'group1_mean': group1_data.mean(),
                'group1_std': group1_data.std(),
                'group2_n': len(group2_data),
                'group2_mean': group2_data.mean(),
                'group2_std': group2_data.std(),
                'mean_difference': mean_diff,
                'cohens_d': cohens_d,
                't_statistic': t_stat,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': p_value < self.alpha
            }
            
        else:
            # ANOVA for multiple groups
            groups = [df[df[treatment_col] == treatment][endpoint].dropna() 
                     for treatment in treatments]
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            results = {
                'test_type': 'one_way_anova',
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'groups': {treatment: {
                    'n': len(df[df[treatment_col] == treatment][endpoint].dropna()),
                    'mean': df[df[treatment_col] == treatment][endpoint].mean(),
                    'std': df[df[treatment_col] == treatment][endpoint].std()
                } for treatment in treatments}
            }
        
        return results
    
    def _adjusted_analysis(self, df: pd.DataFrame,
                         endpoint: str,
                         treatment_col: str, 
                         covariates: list) -> dict:
        """Perform adjusted analysis using multiple regression."""
        
        # Prepare data for regression
        analysis_data = df[[endpoint, treatment_col] + covariates].dropna()
        
        # Create dummy variables for categorical predictors
        analysis_data_encoded = pd.get_dummies(analysis_data, 
                                             columns=[treatment_col] + 
                                             [col for col in covariates 
                                              if analysis_data[col].dtype == 'object'])
        
        # Identify treatment columns after encoding
        treatment_cols = [col for col in analysis_data_encoded.columns 
                         if col.startswith(treatment_col + '_')]
        
        # Build regression model
        predictors = [col for col in analysis_data_encoded.columns 
                     if col != endpoint]
        
        X = analysis_data_encoded[predictors]
        y = analysis_data_encoded[endpoint]
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Extract results
        results = {
            'model_summary': model.summary(),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_p_value': model.f_pvalue,
            'treatment_effects': {}
        }
        
        # Extract treatment effects
        for col in treatment_cols:
            if col in model.params.index:
                results['treatment_effects'][col] = {
                    'coefficient': model.params[col],
                    'std_error': model.bse[col],
                    'p_value': model.pvalues[col],
                    'ci_lower': model.conf_int().loc[col, 0],
                    'ci_upper': model.conf_int().loc[col, 1]
                }
        
        return results
    
    def survival_analysis(self, df: pd.DataFrame,
                         duration_col: str,
                         event_col: str,
                         treatment_col: str) -> dict:
        """Comprehensive survival analysis."""
        
        results = {}
        
        # Kaplan-Meier analysis by treatment group
        kmf_results = {}
        
        for treatment in df[treatment_col].unique():
            treatment_data = df[df[treatment_col] == treatment]
            
            kmf = KaplanMeierFitter()
            kmf.fit(treatment_data[duration_col], 
                   treatment_data[event_col],
                   label=treatment)
            
            # Calculate median survival
            median_survival = kmf.median_survival_time_
            
            # Calculate survival at specific time points
            time_points = [30, 60, 90, 180, 365]  # days
            survival_probs = []
            
            for t in time_points:
                try:
                    prob = kmf.survival_function_at_times(t).iloc[0]
                except:
                    prob = np.nan
                survival_probs.append(prob)
            
            kmf_results[treatment] = {
                'median_survival': median_survival,
                'survival_at_timepoints': dict(zip(time_points, survival_probs)),
                'kmf_object': kmf
            }
        
        results['kaplan_meier'] = kmf_results
        
        # Log-rank test
        if len(df[treatment_col].unique()) == 2:
            treatments = df[treatment_col].unique()
            group1 = df[df[treatment_col] == treatments[0]]
            group2 = df[df[treatment_col] == treatments[1]]
            
            logrank_result = multivariate_logrank_test(
                [group1[duration_col], group2[duration_col]],
                [group1[event_col], group2[event_col]],
                [treatments[0], treatments[1]]
            )
            
            results['logrank_test'] = {
                'test_statistic': logrank_result.test_statistic,
                'p_value': logrank_result.p_value,
                'significant': logrank_result.p_value < self.alpha
            }
        
        # Cox proportional hazards model
        cox_data = df[[duration_col, event_col, treatment_col]].dropna()
        cox_data_encoded = pd.get_dummies(cox_data, columns=[treatment_col])
        
        cph = CoxPHFitter()
        cph.fit(cox_data_encoded, duration_col=duration_col, event_col=event_col)
        
        results['cox_regression'] = {
            'model_summary': cph.summary,
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'aic': cph.AIC_
        }
        
        return results
```

#### 2. Multiple Testing and Power Analysis

```python
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest

class MultipleTestingFramework:
    """Handle multiple testing corrections and power analysis."""
    
    def __init__(self):
        self.correction_methods = {
            'bonferroni': 'bonferroni',
            'holm': 'holm',
            'fdr_bh': 'fdr_bh',  # Benjamini-Hochberg
            'fdr_by': 'fdr_by'   # Benjamini-Yekutieli
        }
    
    def multiple_testing_correction(self, p_values: list, 
                                  method: str = 'fdr_bh',
                                  alpha: float = 0.05) -> dict:
        """Apply multiple testing correction to p-values."""
        
        # Perform correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        results = {
            'original_p_values': p_values,
            'corrected_p_values': p_corrected.tolist(),
            'rejected_hypotheses': rejected.tolist(),
            'method': method,
            'alpha_original': alpha,
            'alpha_corrected': alpha_bonf if method == 'bonferroni' else alpha,
            'n_significant_original': sum(p < alpha for p in p_values),
            'n_significant_corrected': sum(rejected)
        }
        
        return results
    
    def power_analysis(self, effect_sizes: list, 
                      sample_sizes: list,
                      alpha: float = 0.05) -> pd.DataFrame:
        """Comprehensive power analysis for different scenarios."""
        
        power_results = []
        
        for effect_size in effect_sizes:
            for n in sample_sizes:
                # Two-sided t-test power
                power = ttest_power(effect_size, n, alpha, alternative='two-sided')
                
                power_results.append({
                    'effect_size': effect_size,
                    'sample_size': n,
                    'power': power,
                    'alpha': alpha,
                    'adequate_power': power >= 0.8
                })
        
        return pd.DataFrame(power_results)
    
    def sequential_testing(self, df: pd.DataFrame,
                          endpoint: str, 
                          treatment_col: str,
                          interim_fractions: list = [0.5, 0.75, 1.0]) -> dict:
        """Simulate sequential testing with stopping boundaries."""
        
        results = {'interim_analyses': []}
        
        n_total = len(df)
        
        for fraction in interim_fractions:
            # Sample data up to current interim
            n_interim = int(n_total * fraction)
            interim_data = df.iloc[:n_interim]
            
            # Perform analysis
            treatments = interim_data[treatment_col].unique()
            if len(treatments) == 2:
                group1_data = interim_data[interim_data[treatment_col] == treatments[0]][endpoint].dropna()
                group2_data = interim_data[interim_data[treatment_col] == treatments[1]][endpoint].dropna()
                
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                
                # Calculate effect size
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                    (len(group2_data) - 1) * group2_data.var()) / 
                                   (len(group1_data) + len(group2_data) - 2))
                
                effect_size = abs(group1_data.mean() - group2_data.mean()) / pooled_std
                
                # Simple alpha spending (O'Brien-Fleming-like)
                alpha_interim = 0.05 * (1 / fraction**2)
                
                results['interim_analyses'].append({
                    'fraction': fraction,
                    'n': n_interim,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'alpha_threshold': alpha_interim,
                    'stop_for_efficacy': p_value < alpha_interim,
                    'group1_mean': group1_data.mean(),
                    'group2_mean': group2_data.mean()
                })
        
        return results
```

### Hands-On Exercise 1: Clinical Trial Statistical Analysis

**Scenario**: Phase III randomized controlled trial of a new diabetes medication.

**Primary Endpoint**: HbA1c reduction from baseline at 24 weeks  
**Secondary Endpoints**: Weight change, blood pressure reduction, time to glycemic target
**Safety Endpoints**: Adverse events, hypoglycemic episodes

**Your Task**: Perform comprehensive statistical analysis including:
- Descriptive statistics by treatment group
- Primary endpoint analysis with effect size
- Adjusted analysis controlling for baseline characteristics
- Multiple testing corrections for secondary endpoints
- Survival analysis for time-to-target achievement

---

## Part 2: Machine Learning Pipeline Development (105 minutes)

### Professional Challenge: Predictive Model Development

Building production-ready ML models requires systematic approaches:
- **Feature engineering**: Domain-informed variable creation
- **Model selection**: Comparing multiple algorithms systematically  
- **Validation frameworks**: Proper cross-validation preventing data leakage
- **Performance metrics**: Clinically relevant evaluation criteria

### End-to-End ML Pipeline

#### 1. Feature Engineering for Clinical Data

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class ClinicalFeatureEngineer:
    """Advanced feature engineering for clinical datasets."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated clinical features."""
        
        df_featured = df.copy()
        
        # 1. Derived clinical variables
        if 'age' in df.columns and 'gender' in df.columns:
            # Age-gender interaction (different risk profiles)
            df_featured['age_gender_interaction'] = (
                df['age'] * df['gender'].map({'Male': 1, 'Female': 0})
            )
        
        # 2. Laboratory ratio features (clinically meaningful)
        if all(col in df.columns for col in ['total_cholesterol', 'hdl_cholesterol']):
            df_featured['tc_hdl_ratio'] = (
                df['total_cholesterol'] / df['hdl_cholesterol']
            )
        
        if all(col in df.columns for col in ['ldl_cholesterol', 'hdl_cholesterol']):
            df_featured['ldl_hdl_ratio'] = (
                df['ldl_cholesterol'] / df['hdl_cholesterol']
            )
        
        # 3. Cardiovascular risk scores
        if all(col in df.columns for col in ['age', 'systolic_bp', 'total_cholesterol', 'smoking']):
            df_featured['framingham_risk_score'] = self._calculate_framingham_risk(df)
        
        # 4. Medication combination effects
        med_columns = [col for col in df.columns if 'on_' in col and col.startswith('on_')]
        if len(med_columns) >= 2:
            df_featured['polypharmacy_count'] = df[med_columns].sum(axis=1)
            
            # Specific drug combinations
            if 'on_statin' in df.columns and 'on_ace_inhibitor' in df.columns:
                df_featured['statin_ace_combo'] = (
                    df['on_statin'] & df['on_ace_inhibitor']
                ).astype(int)
        
        # 5. Temporal features (if longitudinal data available)
        if 'days_since_baseline' in df.columns:
            df_featured['treatment_duration_months'] = df['days_since_baseline'] / 30.44
            df_featured['long_term_treatment'] = (df['days_since_baseline'] > 180).astype(int)
        
        # 6. Missing data patterns as features
        df_featured['n_missing_labs'] = df.select_dtypes(include=[np.number]).isnull().sum(axis=1)
        df_featured['has_complete_lipid_panel'] = (
            df[['total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides']]
            .notnull().all(axis=1).astype(int)
        )
        
        # 7. Binned continuous variables for non-linear relationships
        if 'age' in df.columns:
            df_featured['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 40, 55, 70, 100], 
                labels=['Under_40', '40-55', '55-70', 'Over_70']
            )
        
        if 'bmi' in df.columns:
            df_featured['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 50],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
        
        return df_featured
    
    def _calculate_framingham_risk(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simplified Framingham Risk Score."""
        
        # Simplified scoring system (actual Framingham is more complex)
        risk_score = pd.Series(0, index=df.index, dtype=float)
        
        # Age points
        risk_score += np.where(df['age'] >= 45, 2, 0)
        risk_score += np.where(df['age'] >= 55, 1, 0)
        risk_score += np.where(df['age'] >= 65, 1, 0)
        
        # Blood pressure points  
        risk_score += np.where(df['systolic_bp'] >= 140, 2, 0)
        risk_score += np.where(df['systolic_bp'] >= 160, 1, 0)
        
        # Cholesterol points
        risk_score += np.where(df['total_cholesterol'] >= 240, 2, 0)
        risk_score += np.where(df['total_cholesterol'] >= 280, 1, 0)
        
        # Smoking points
        risk_score += np.where(df['smoking'] == 1, 2, 0)
        
        return risk_score

class MLPipelineBuilder:
    """Build comprehensive ML pipelines for clinical prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipelines = {}
        self.results = {}
        
    def create_preprocessing_pipeline(self, 
                                    numeric_features: list,
                                    categorical_features: list) -> ColumnTransformer:
        """Create preprocessing pipeline handling different feature types."""
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def build_model_pipelines(self, 
                            preprocessing_pipeline: ColumnTransformer) -> dict:
        """Build multiple ML model pipelines."""
        
        # Define models with hyperparameter grids
        models = {
            'logistic_regression': {
                'model': Pipeline([
                    ('preprocessor', preprocessing_pipeline),
                    ('classifier', LogisticRegression(random_state=self.random_state))
                ]),
                'params': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            
            'random_forest': {
                'model': Pipeline([
                    ('preprocessor', preprocessing_pipeline),
                    ('classifier', RandomForestClassifier(random_state=self.random_state))
                ]),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            },
            
            'gradient_boosting': {
                'model': Pipeline([
                    ('preprocessor', preprocessing_pipeline),
                    ('classifier', GradientBoostingClassifier(random_state=self.random_state))
                ]),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.05, 0.1, 0.15],
                    'classifier__max_depth': [3, 5, 7]
                }
            }
        }
        
        return models
    
    def train_and_evaluate_models(self, 
                                X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                models: dict,
                                cv_folds: int = 5) -> dict:
        """Train and evaluate multiple models with cross-validation."""
        
        results = {}
        
        for model_name, model_config in models.items():
            print(f"Training {model_name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit model
            grid_search.fit(X_train, y_train)
            
            # Best model predictions
            y_pred = grid_search.best_estimator_.predict(X_test)
            y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'test_auc': auc,
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix,
                'cv_scores': grid_search.cv_results_['mean_test_score'],
                'best_estimator': grid_search.best_estimator_,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            print(f"  Best CV AUC: {grid_search.best_score_:.4f}")
            print(f"  Test AUC: {auc:.4f}")
        
        return results
    
    def feature_importance_analysis(self, model_results: dict, 
                                  feature_names: list) -> pd.DataFrame:
        """Analyze feature importance across models."""
        
        importance_results = []
        
        for model_name, results in model_results.items():
            estimator = results['best_estimator']
            
            # Extract classifier from pipeline
            classifier = estimator.named_steps['classifier']
            
            # Get feature importance based on model type
            if hasattr(classifier, 'feature_importances_'):
                # Tree-based models
                importances = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                # Linear models
                importances = np.abs(classifier.coef_[0])
            else:
                continue
            
            # Get feature names after preprocessing
            preprocessor = estimator.named_steps['preprocessor']
            feature_names_transformed = (
                preprocessor.get_feature_names_out(feature_names)
            )
            
            for i, importance in enumerate(importances):
                importance_results.append({
                    'model': model_name,
                    'feature': feature_names_transformed[i],
                    'importance': importance,
                    'rank': None  # Will fill after sorting
                })
        
        importance_df = pd.DataFrame(importance_results)
        
        # Add ranks within each model
        for model in importance_df['model'].unique():
            mask = importance_df['model'] == model
            importance_df.loc[mask, 'rank'] = (
                importance_df.loc[mask, 'importance']
                .rank(ascending=False, method='dense')
            )
        
        return importance_df.sort_values(['model', 'rank'])
```

#### 2. Model Validation and Performance Assessment

```python
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt

class ModelValidationFramework:
    """Comprehensive model validation for clinical applications."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def comprehensive_validation(self, 
                               model, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               model_name: str = "Model") -> dict:
        """Perform comprehensive model validation."""
        
        validation_results = {}
        
        # 1. Learning curves
        validation_results['learning_curves'] = self._plot_learning_curves(
            model, X, y, model_name
        )
        
        # 2. Validation curves for key hyperparameters
        if hasattr(model.named_steps['classifier'], 'C'):  # Logistic Regression
            validation_results['validation_curves'] = self._plot_validation_curves(
                model, X, y, 'classifier__C', [0.01, 0.1, 1, 10, 100], model_name
            )
        
        # 3. Cross-validation stability
        validation_results['cv_stability'] = self._cv_stability_analysis(model, X, y)
        
        # 4. Bootstrap confidence intervals
        validation_results['bootstrap_ci'] = self._bootstrap_confidence_intervals(
            model, X, y
        )
        
        return validation_results
    
    def _plot_learning_curves(self, model, X, y, model_name):
        """Generate learning curves to assess bias-variance tradeoff."""
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='red')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('AUC Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return {
            'train_sizes': train_sizes,
            'train_scores_mean': train_mean,
            'train_scores_std': train_std,
            'validation_scores_mean': val_mean,
            'validation_scores_std': val_std,
            'final_training_score': train_mean[-1],
            'final_validation_score': val_mean[-1],
            'overfitting_gap': train_mean[-1] - val_mean[-1]
        }
    
    def clinical_performance_metrics(self, y_true, y_pred, y_pred_proba) -> dict:
        """Calculate clinically relevant performance metrics."""
        
        # Standard classification metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Clinical metrics
        sensitivity = tp / (tp + fn)  # True Positive Rate
        specificity = tn / (tn + fp)  # True Negative Rate
        ppv = tp / (tp + fp)  # Positive Predictive Value
        npv = tn / (tn + fn)  # Negative Predictive Value
        
        # Likelihood ratios
        lr_positive = sensitivity / (1 - specificity)
        lr_negative = (1 - sensitivity) / specificity
        
        # Number needed to treat (simplified calculation)
        # Assumes baseline event rate and treatment effect
        baseline_risk = y_true.mean()
        nnt = 1 / abs(ppv - baseline_risk)
        
        metrics = {
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1_score': 2 * (ppv * sensitivity) / (ppv + sensitivity),
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            'nnt_approximation': nnt,
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def calibration_analysis(self, y_true, y_pred_proba, n_bins: int = 10) -> dict:
        """Analyze model calibration - how well predicted probabilities match actual outcomes."""
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        # Brier score (lower is better)
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        
        # Plot calibration curve
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                label=f'Model Calibration (Brier Score: {brier_score:.3f})')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Calculate calibration slope and intercept
        from sklearn.linear_model import LinearRegression
        cal_model = LinearRegression()
        cal_model.fit(mean_predicted_value.reshape(-1, 1), fraction_of_positives)
        calibration_slope = cal_model.coef_[0]
        calibration_intercept = cal_model.intercept_
        
        return {
            'brier_score': brier_score,
            'calibration_slope': calibration_slope,
            'calibration_intercept': calibration_intercept,
            'mean_predicted_values': mean_predicted_value,
            'fraction_of_positives': fraction_of_positives,
            'well_calibrated': abs(calibration_slope - 1.0) < 0.1 and abs(calibration_intercept) < 0.05
        }
```

### Hands-On Exercise 2: End-to-End ML Pipeline

**Scenario**: Develop a model to predict 30-day hospital readmission risk.

**Features Available**:
- Patient demographics (age, gender, insurance)
- Admission characteristics (diagnosis, length of stay, discharge disposition)
- Clinical indicators (comorbidities, vital signs, lab values)
- Historical utilization (previous admissions, emergency visits)

**Your Task**: Build complete ML pipeline including:
- Advanced feature engineering with clinical domain knowledge
- Multiple model comparison with proper validation
- Feature importance analysis and model interpretation
- Clinical performance metrics and calibration analysis

---

## Part 3: Model Interpretation and Clinical Integration (75 minutes)

### Professional Challenge: Explainable AI for Clinical Decision Support

Clinical models must be interpretable to gain physician trust and regulatory approval:
- **Local explanations**: Why did the model make this specific prediction?
- **Global explanations**: What patterns did the model learn overall?
- **Feature interactions**: How do variables combine to influence predictions?
- **Uncertainty quantification**: How confident is the model in its predictions?

### Advanced Model Interpretation

#### 1. SHAP (SHapley Additive exPlanations) Analysis

```python
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ClinicalModelInterpreter:
    """Advanced model interpretation for clinical applications."""
    
    def __init__(self):
        self.explainers = {}
        self.explanations = {}
        
    def comprehensive_interpretation(self, 
                                   model, 
                                   X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame,
                                   feature_names: list,
                                   model_name: str = "Model") -> dict:
        """Perform comprehensive model interpretation analysis."""
        
        interpretation_results = {}
        
        # 1. SHAP Analysis
        interpretation_results['shap'] = self._shap_analysis(
            model, X_train, X_test, feature_names, model_name
        )
        
        # 2. LIME Analysis for local explanations
        interpretation_results['lime'] = self._lime_analysis(
            model, X_train, X_test, feature_names
        )
        
        # 3. Feature interaction analysis
        interpretation_results['interactions'] = self._feature_interaction_analysis(
            model, X_train, feature_names
        )
        
        # 4. Decision boundary analysis
        interpretation_results['decision_boundaries'] = self._analyze_decision_boundaries(
            model, X_test, feature_names
        )
        
        return interpretation_results
    
    def _shap_analysis(self, model, X_train, X_test, feature_names, model_name):
        """Comprehensive SHAP analysis."""
        
        # Get the actual sklearn classifier from the pipeline
        classifier = model.named_steps['classifier']
        preprocessor = model.named_steps['preprocessor']
        
        # Transform the data using the preprocessor
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get feature names after preprocessing
        feature_names_transformed = preprocessor.get_feature_names_out(feature_names)
        
        # Create SHAP explainer based on model type
        if hasattr(classifier, 'predict_proba'):
            explainer = shap.Explainer(classifier.predict_proba, X_train_transformed)
        else:
            explainer = shap.Explainer(classifier, X_train_transformed)
        
        # Calculate SHAP values
        shap_values = explainer(X_test_transformed)
        
        # For binary classification, use positive class SHAP values
        if len(shap_values.shape) == 3:  # Multi-output case
            shap_values_binary = shap_values[:, :, 1]
        else:
            shap_values_binary = shap_values
        
        # Global feature importance
        global_importance = np.abs(shap_values_binary.values).mean(axis=0)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names_transformed,
            'importance': global_importance
        }).sort_values('importance', ascending=False)
        
        results = {
            'shap_values': shap_values_binary,
            'global_importance': importance_df,
            'explainer': explainer,
            'feature_names': feature_names_transformed,
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
        }
        
        return results
    
    def create_shap_visualizations(self, shap_results: dict, 
                                 X_test_transformed: np.ndarray,
                                 n_features: int = 20) -> dict:
        """Create comprehensive SHAP visualizations."""
        
        visualizations = {}
        
        # 1. Summary plot (global importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_results['shap_values'].values,
            X_test_transformed,
            feature_names=shap_results['feature_names'],
            max_display=n_features,
            show=False
        )
        plt.title('SHAP Summary Plot - Global Feature Importance')
        plt.tight_layout()
        visualizations['summary_plot'] = plt.gcf()
        
        # 2. Waterfall plot for a single prediction
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap_results['shap_values'][0], max_display=10, show=False)
        plt.title('SHAP Waterfall Plot - Individual Prediction Explanation')
        plt.tight_layout()
        visualizations['waterfall_plot'] = plt.gcf()
        
        # 3. Force plot for multiple predictions
        force_plot = shap.force_plot(
            shap_results['base_value'],
            shap_results['shap_values'].values[:5],  # First 5 predictions
            X_test_transformed[:5],
            feature_names=shap_results['feature_names'],
            show=False
        )
        visualizations['force_plot'] = force_plot
        
        return visualizations
    
    def clinical_explanation_summary(self, 
                                   shap_results: dict,
                                   patient_index: int,
                                   X_test: pd.DataFrame,
                                   threshold: float = 0.5) -> dict:
        """Create clinical explanation summary for a specific patient."""
        
        # Get SHAP values for the specific patient
        patient_shap = shap_results['shap_values'][patient_index]
        patient_features = X_test.iloc[patient_index]
        
        # Get top contributing features
        feature_contributions = []
        
        for i, (feature, shap_val) in enumerate(zip(shap_results['feature_names'], 
                                                   patient_shap.values)):
            contribution_type = "Increases Risk" if shap_val > 0 else "Decreases Risk"
            
            feature_contributions.append({
                'feature': feature,
                'shap_value': shap_val,
                'feature_value': X_test.iloc[patient_index, i % len(X_test.columns)],
                'contribution_type': contribution_type,
                'magnitude': abs(shap_val)
            })
        
        # Sort by magnitude
        feature_contributions.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Create clinical summary
        top_risk_factors = [fc for fc in feature_contributions[:5] if fc['shap_value'] > 0]
        top_protective_factors = [fc for fc in feature_contributions[:5] if fc['shap_value'] < 0]
        
        summary = {
            'patient_index': patient_index,
            'prediction_probability': None,  # Will be filled by calling function
            'top_risk_factors': top_risk_factors,
            'top_protective_factors': top_protective_factors,
            'all_contributions': feature_contributions,
            'clinical_interpretation': self._generate_clinical_interpretation(
                top_risk_factors, top_protective_factors
            )
        }
        
        return summary
    
    def _generate_clinical_interpretation(self, 
                                        risk_factors: list, 
                                        protective_factors: list) -> str:
        """Generate human-readable clinical interpretation."""
        
        interpretation_parts = []
        
        if risk_factors:
            risk_names = [rf['feature'].replace('_', ' ').title() for rf in risk_factors[:3]]
            interpretation_parts.append(
                f"Primary risk factors: {', '.join(risk_names)}"
            )
        
        if protective_factors:
            protective_names = [pf['feature'].replace('_', ' ').title() for pf in protective_factors[:3]]
            interpretation_parts.append(
                f"Protective factors: {', '.join(protective_names)}"
            )
        
        if not risk_factors and not protective_factors:
            interpretation_parts.append("No strong contributing factors identified")
        
        return ". ".join(interpretation_parts) + "."
```

#### 2. Interactive Clinical Decision Support Dashboard

```python
class ClinicalDecisionSupportDashboard:
    """Interactive dashboard for clinical decision support."""
    
    def __init__(self):
        self.colors = {
            'high_risk': '#DC143C',      # Crimson
            'medium_risk': '#FF8C00',    # Dark Orange  
            'low_risk': '#32CD32',       # Lime Green
            'neutral': '#4682B4'         # Steel Blue
        }
    
    def create_patient_risk_dashboard(self, 
                                    model_results: dict,
                                    shap_results: dict,
                                    patient_data: pd.DataFrame,
                                    patient_index: int) -> go.Figure:
        """Create comprehensive patient risk assessment dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Risk Prediction',
                'Feature Contributions (SHAP)',
                'Similar Patient Outcomes',
                'Risk Trajectory Over Time'
            ],
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # Get patient prediction
        patient_prob = model_results['prediction_probabilities'][patient_index]
        
        # 1. Risk gauge
        risk_color = self._get_risk_color(patient_prob)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=patient_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Readmission Risk (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 20], 'color': self.colors['low_risk']},
                        {'range': [20, 50], 'color': self.colors['medium_risk']},
                        {'range': [50, 100], 'color': self.colors['high_risk']}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Feature contributions
        patient_shap = shap_results['shap_values'][patient_index]
        top_features = np.argsort(np.abs(patient_shap.values))[-10:]
        
        feature_names = [shap_results['feature_names'][i] for i in top_features]
        shap_values = [patient_shap.values[i] for i in top_features]
        colors = [self.colors['high_risk'] if sv > 0 else self.colors['low_risk'] 
                 for sv in shap_values]
        
        fig.add_trace(
            go.Bar(
                x=shap_values,
                y=feature_names,
                orientation='h',
                marker_color=colors,
                text=[f"{sv:.3f}" for sv in shap_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Similar patients (simulated data)
        # In practice, would use nearest neighbors or clustering
        similar_patients_risk = np.random.normal(patient_prob, 0.1, 20)
        similar_patients_outcomes = np.random.binomial(1, similar_patients_risk)
        
        fig.add_trace(
            go.Scatter(
                x=similar_patients_risk,
                y=similar_patients_outcomes,
                mode='markers',
                marker=dict(
                    size=8,
                    color=[self._get_risk_color(risk) for risk in similar_patients_risk]
                ),
                name='Similar Patients'
            ),
            row=2, col=1
        )
        
        # 4. Risk trajectory (simulated longitudinal data)
        days = np.arange(0, 31)  # 30 days
        risk_trajectory = patient_prob * (1 - np.exp(-days/10))  # Exponential approach
        risk_trajectory += np.random.normal(0, 0.02, len(days))  # Add noise
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=risk_trajectory * 100,
                mode='lines+markers',
                line=dict(color=risk_color, width=3),
                name='Risk Trajectory'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Patient Risk Assessment Dashboard",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="SHAP Value", row=1, col=2)
        fig.update_yaxes(title_text="Features", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Risk", row=2, col=1)
        fig.update_yaxes(title_text="Actual Outcome", row=2, col=1)
        fig.update_xaxes(title_text="Days", row=2, col=2)
        fig.update_yaxes(title_text="Risk (%)", row=2, col=2)
        
        return fig
    
    def _get_risk_color(self, risk_prob: float) -> str:
        """Get color based on risk probability."""
        if risk_prob < 0.2:
            return self.colors['low_risk']
        elif risk_prob < 0.5:
            return self.colors['medium_risk']
        else:
            return self.colors['high_risk']
```

### Hands-On Exercise 3: Model Interpretation and Clinical Integration

**Scenario**: Your readmission prediction model needs clinical validation.

**Tasks**:
1. **SHAP Analysis**: Generate comprehensive explanations for model predictions
2. **Clinical Dashboard**: Create interactive dashboard for physician review
3. **Patient-Specific Reports**: Generate interpretable reports for individual patients
4. **Model Validation**: Compare model explanations with clinical judgment

---

## Part 4: Time Series Analysis and Forecasting (75 minutes)

### Professional Applications: Temporal Modeling in Healthcare

Time series analysis enables:
- **Patient monitoring**: Predicting clinical deterioration from vital signs
- **Resource planning**: Forecasting bed utilization, staffing needs
- **Epidemiological surveillance**: Disease outbreak detection and modeling
- **Quality improvement**: Monitoring intervention effects over time

### Advanced Time Series Framework

#### 1. Clinical Time Series Modeling

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ClinicalTimeSeriesAnalyzer:
    """Advanced time series analysis for clinical applications."""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        
    def comprehensive_time_series_analysis(self, 
                                         df: pd.DataFrame,
                                         date_col: str,
                                         value_col: str,
                                         patient_col: str = None) -> dict:
        """Perform comprehensive time series analysis."""
        
        results = {}
        
        # 1. Data preprocessing and validation
        results['preprocessing'] = self._preprocess_time_series(
            df, date_col, value_col, patient_col
        )
        
        # 2. Exploratory time series analysis
        results['exploratory'] = self._exploratory_time_series_analysis(
            results['preprocessing']['clean_data'], value_col
        )
        
        # 3. Stationarity testing
        results['stationarity'] = self._test_stationarity(
            results['preprocessing']['clean_data'][value_col]
        )
        
        # 4. Seasonal decomposition
        results['decomposition'] = self._seasonal_decomposition(
            results['preprocessing']['clean_data'], value_col
        )
        
        # 5. Multiple forecasting models
        results['forecasts'] = self._multi_model_forecasting(
            results['preprocessing']['clean_data'], value_col
        )
        
        return results
    
    def _preprocess_time_series(self, df, date_col, value_col, patient_col):
        """Comprehensive time series preprocessing."""
        
        # Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date (and patient if applicable)
        sort_cols = [patient_col, date_col] if patient_col else [date_col]
        df_sorted = df.sort_values(sort_cols)
        
        # Handle missing values
        if patient_col:
            # Forward fill within each patient
            df_sorted[value_col] = (df_sorted.groupby(patient_col)[value_col]
                                   .fillna(method='ffill'))
        else:
            # Simple forward fill for single series
            df_sorted[value_col] = df_sorted[value_col].fillna(method='ffill')
        
        # Remove remaining missing values
        df_clean = df_sorted.dropna(subset=[value_col])
        
        # Set date as index for time series analysis
        if patient_col is None:
            df_clean = df_clean.set_index(date_col)
            # Ensure daily frequency by resampling
            df_clean = df_clean.resample('D')[value_col].mean()
            df_clean = df_clean.fillna(method='ffill')
        
        preprocessing_results = {
            'original_shape': df.shape,
            'clean_shape': df_clean.shape,
            'missing_values_removed': df[value_col].isna().sum(),
            'clean_data': df_clean,
            'date_range': (df_clean.index.min(), df_clean.index.max()) if patient_col is None else None,
            'frequency': 'Daily' if patient_col is None else 'Variable'
        }
        
        return preprocessing_results
    
    def _exploratory_time_series_analysis(self, data, value_col):
        """Exploratory analysis of time series data."""
        
        if isinstance(data, pd.DataFrame):
            series = data[value_col]
        else:
            series = data
        
        # Basic statistics
        basic_stats = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing'
        }
        
        # Autocorrelation analysis
        from statsmodels.tsa.stattools import acf, pacf
        
        autocorr = acf(series.dropna(), nlags=40)
        partial_autocorr = pacf(series.dropna(), nlags=40)
        
        # Identify potential seasonality
        seasonal_periods = []
        for lag in [7, 30, 365]:  # Weekly, monthly, yearly patterns
            if lag < len(autocorr) and abs(autocorr[lag]) > 0.3:
                seasonal_periods.append(lag)
        
        return {
            'basic_statistics': basic_stats,
            'autocorrelation': autocorr,
            'partial_autocorrelation': partial_autocorr,
            'potential_seasonal_periods': seasonal_periods,
            'series_length': len(series)
        }
    
    def _multi_model_forecasting(self, data, value_col, forecast_periods=30):
        """Compare multiple forecasting approaches."""
        
        if isinstance(data, pd.DataFrame):
            series = data[value_col]
        else:
            series = data
        
        forecasting_results = {}
        
        # Split data for validation
        train_size = int(len(series) * 0.8)
        train_data = series[:train_size]
        test_data = series[train_size:]
        
        # 1. ARIMA Model
        try:
            # Auto ARIMA (simplified)
            arima_model = ARIMA(train_data, order=(1, 1, 1))
            arima_fitted = arima_model.fit()
            
            arima_forecast = arima_fitted.forecast(steps=len(test_data))
            arima_future = arima_fitted.forecast(steps=forecast_periods)
            
            forecasting_results['arima'] = {
                'model': arima_fitted,
                'validation_forecast': arima_forecast,
                'future_forecast': arima_future,
                'aic': arima_fitted.aic,
                'validation_mae': np.mean(np.abs(test_data - arima_forecast))
            }
        except:
            forecasting_results['arima'] = {'error': 'ARIMA model failed to fit'}
        
        # 2. Exponential Smoothing
        try:
            exp_smooth = ExponentialSmoothing(train_data, seasonal=None)
            exp_fitted = exp_smooth.fit()
            
            exp_forecast = exp_fitted.forecast(steps=len(test_data))
            exp_future = exp_fitted.forecast(steps=forecast_periods)
            
            forecasting_results['exponential_smoothing'] = {
                'model': exp_fitted,
                'validation_forecast': exp_forecast,
                'future_forecast': exp_future,
                'validation_mae': np.mean(np.abs(test_data - exp_forecast))
            }
        except:
            forecasting_results['exponential_smoothing'] = {'error': 'Exponential smoothing failed'}
        
        # 3. Random Forest with time-based features
        try:
            # Create time-based features
            train_df = train_data.reset_index()
            train_df['day_of_year'] = train_df[train_df.columns[0]].dt.dayofyear
            train_df['day_of_week'] = train_df[train_df.columns[0]].dt.dayofweek
            train_df['month'] = train_df[train_df.columns[0]].dt.month
            train_df['lag_1'] = train_data.shift(1)
            train_df['lag_7'] = train_data.shift(7)
            train_df['rolling_mean_7'] = train_data.rolling(7).mean()
            
            # Remove missing values
            train_df = train_df.dropna()
            
            # Features and target
            features = ['day_of_year', 'day_of_week', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']
            X_train = train_df[features]
            y_train = train_df[train_df.columns[1]]  # Value column
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Create test features (simplified)
            test_features = []
            for i in range(len(test_data)):
                test_date = test_data.index[i]
                test_features.append([
                    test_date.dayofyear,
                    test_date.dayofweek,
                    test_date.month,
                    series.iloc[train_size + i - 1] if i > 0 else train_data.iloc[-1],
                    series.iloc[train_size + i - 7] if i >= 7 else train_data.iloc[-7],
                    series.iloc[max(0, train_size + i - 7):train_size + i].mean()
                ])
            
            X_test = pd.DataFrame(test_features, columns=features)
            rf_forecast = rf_model.predict(X_test)
            
            forecasting_results['random_forest'] = {
                'model': rf_model,
                'validation_forecast': rf_forecast,
                'feature_importance': dict(zip(features, rf_model.feature_importances_)),
                'validation_mae': np.mean(np.abs(test_data - rf_forecast))
            }
        except Exception as e:
            forecasting_results['random_forest'] = {'error': f'Random Forest failed: {str(e)}'}
        
        # 4. Model comparison
        mae_scores = {}
        for model_name, results in forecasting_results.items():
            if 'validation_mae' in results:
                mae_scores[model_name] = results['validation_mae']
        
        best_model = min(mae_scores, key=mae_scores.get) if mae_scores else None
        
        forecasting_results['model_comparison'] = {
            'mae_scores': mae_scores,
            'best_model': best_model,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
        
        return forecasting_results
    
    def clinical_forecasting_dashboard(self, forecasting_results: dict, 
                                     series_name: str = "Clinical Metric") -> go.Figure:
        """Create comprehensive forecasting dashboard."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Historical Data and Forecasts',
                'Model Performance Comparison',
                'Forecast Confidence Intervals',
                'Residual Analysis'
            ]
        )
        
        # 1. Historical data and forecasts
        # This would include plotting actual vs predicted values
        # Simplified for demonstration
        
        # 2. Model performance comparison
        if 'model_comparison' in forecasting_results:
            mae_scores = forecasting_results['model_comparison']['mae_scores']
            
            fig.add_trace(
                go.Bar(
                    x=list(mae_scores.keys()),
                    y=list(mae_scores.values()),
                    name='MAE Scores'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Time Series Forecasting Dashboard - {series_name}",
            showlegend=False
        )
        
        return fig
```

### Final Exercise: Comprehensive Clinical Analytics Project

**Scenario**: ICU patient deterioration prediction system.

**Data Components**:
- Continuous vital signs (heart rate, blood pressure, oxygen saturation)
- Laboratory values (hourly blood gases, daily chemistry panels)
- Medication administration records
- Clinical events and interventions

**Deliverables**:
1. **Time series preprocessing** and quality assessment
2. **Statistical models** for trend detection and change point analysis
3. **Machine learning models** for deterioration prediction
4. **Model interpretation** with SHAP and clinical explanations
5. **Interactive dashboard** for real-time clinical decision support

---

## Assessment and Professional Integration

### Capstone Project: End-to-End Clinical Analytics Pipeline

**Project Overview**: Comprehensive cardiovascular risk prediction system

**Dataset**: Simulated longitudinal cardiovascular registry with:
- **Demographics and comorbidities**: 1,000 patients
- **Laboratory monitoring**: Quarterly lipid panels, HbA1c, kidney function
- **Medication history**: Statin therapy, antihypertensives, diabetes medications  
- **Outcomes**: Major adverse cardiovascular events (MACE) over 5 years

**Deliverables** (100 points total):

1. **Statistical Analysis** (25 points)
   - Comprehensive descriptive analysis by risk factors
   - Time-to-event analysis for MACE outcomes
   - Multiple testing corrections for secondary endpoints
   - Power analysis and effect size calculations

2. **Machine Learning Pipeline** (25 points)
   - Advanced feature engineering with clinical rationale
   - Multiple model comparison with proper validation
   - Hyperparameter tuning and performance optimization
   - Clinical performance metrics and calibration analysis

3. **Model Interpretation** (25 points)
   - SHAP analysis for global and local explanations
   - Feature interaction analysis
   - Clinical decision support dashboard
   - Patient-specific risk reports with actionable insights

4. **Time Series Analysis** (25 points)
   - Longitudinal risk factor evolution analysis
   - Forecasting models for risk trajectory prediction
   - Change point detection for clinical intervention timing
   - Real-time monitoring dashboard design

### Professional Skills Mastery Assessment

After this lecture, you should demonstrate mastery in:

**Statistical Modeling**:
- [ ] Design and execute complex statistical analyses for clinical research
- [ ] Handle multiple testing problems with appropriate corrections
- [ ] Perform survival analysis and time-to-event modeling
- [ ] Calculate clinically meaningful effect sizes and confidence intervals

**Machine Learning**:
- [ ] Build end-to-end ML pipelines with proper validation frameworks
- [ ] Engineer domain-informed features for clinical applications
- [ ] Compare and select optimal models using clinical performance criteria
- [ ] Validate models using bootstrap and cross-validation techniques

**Model Interpretation**:
- [ ] Generate local and global model explanations using SHAP
- [ ] Create clinically interpretable model summaries
- [ ] Build interactive dashboards for clinical decision support
- [ ] Communicate model insights to clinical stakeholders

**Time Series Analysis**:
- [ ] Preprocess and analyze clinical time series data
- [ ] Build forecasting models for clinical applications
- [ ] Detect significant changes and trends in patient trajectories
- [ ] Create real-time monitoring and alerting systems

### Looking Ahead to L09

In our next lecture, we'll integrate these analytical foundations with production deployment:
- **Workflow automation** and reproducible analysis pipelines
- **Code quality practices** for collaborative development
- **Testing frameworks** for analytical code validation
- **Deployment strategies** for clinical decision support systems

The statistical modeling and ML competencies you've mastered today will serve as the analytical engine for the automated, production-ready systems we'll build next.

---

## Additional Resources

### Statistical Analysis
- **Clinical Trial Statistics**: Chow, S.C. "Design and Analysis of Clinical Trials"
- **Survival Analysis**: Klein & Moeschberger "Survival Analysis: Techniques for Censored and Truncated Data"
- **Multiple Testing**: Dmitrienko, A. "Multiple Testing Problems in Pharmaceutical Statistics"

### Machine Learning
- **Clinical ML**: Rajkomar, A. "Machine Learning in Medicine" (Nature Medicine)
- **Model Interpretation**: Molnar, C. "Interpretable Machine Learning"
- **Healthcare AI**: Topol, E. "Deep Medicine: How AI Can Make Healthcare Human Again"

### Time Series Analysis
- **Healthcare Time Series**: Chatfield, C. "The Analysis of Time Series: An Introduction"
- **Clinical Forecasting**: Montgomery, D.C. "Introduction to Time Series Analysis and Forecasting"

### Regulatory and Clinical Integration
- **FDA AI Guidance**: Software as Medical Device (SaMD) guidelines
- **Clinical Decision Support**: Sittig, D.F. "Ten Commandments for Effective Clinical Decision Support"
- **Health Informatics Standards**: HL7 FHIR, IHE profiles