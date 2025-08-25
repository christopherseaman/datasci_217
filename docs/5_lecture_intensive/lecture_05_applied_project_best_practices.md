# Lecture 05: Applied Project & Best Practices

*Integration Project, Statistical Methods, and Professional Development*

## Learning Objectives

By the end of this lecture, you will be able to:
- Apply all previous concepts in a comprehensive capstone project
- Understand fundamental statistical concepts and machine learning basics
- Implement professional debugging and error handling practices
- Deploy and share your data science work effectively
- Navigate the data science career landscape and continue learning

## Introduction: Bringing It All Together

Today we culminate our intensive journey through data science fundamentals by applying everything we've learned in a comprehensive project. We'll also explore the statistical and machine learning concepts that will guide your future learning, master professional debugging practices, and understand how to share and deploy your work.

This final lecture serves as both a capstone experience and a launching pad for your continued growth in data science. We'll bridge the gap between academic learning and professional practice, ensuring you're prepared for real-world data science challenges.

## Part 1: Statistical Methods and Machine Learning Overview

### Statistical Foundations

**Descriptive Statistics Review:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def comprehensive_stats_overview():
    """Demonstrate key statistical concepts for data science"""
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    # Create realistic dataset
    age = np.random.normal(35, 12, n).clip(18, 80)
    income = 30000 + age * 800 + np.random.normal(0, 5000, n)
    education_years = np.random.normal(14, 3, n).clip(8, 20)
    experience = np.maximum(0, age - education_years - 6 + np.random.normal(0, 2, n))
    
    # Add some relationships
    income = income + education_years * 2000 + experience * 1000
    income = np.maximum(income, 20000)  # Minimum wage floor
    
    # Create categorical variables
    job_satisfaction = np.random.choice(['Low', 'Medium', 'High'], n, p=[0.2, 0.5, 0.3])
    promoted = (income > np.percentile(income, 70)) & (np.random.random(n) > 0.3)
    
    df = pd.DataFrame({
        'age': age.round(1),
        'income': income.round(0),
        'education_years': education_years.round(1),
        'experience': experience.round(1),
        'job_satisfaction': job_satisfaction,
        'promoted': promoted
    })
    
    print("=== DESCRIPTIVE STATISTICS ===")
    print(df.describe().round(2))
    
    # Correlation analysis
    print("\n=== CORRELATION MATRIX ===")
    corr_matrix = df[['age', 'income', 'education_years', 'experience']].corr()
    print(corr_matrix.round(3))
    
    # Visualize correlations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
    axes[0,0].set_title('Correlation Matrix')
    
    # Distribution plots
    df['income'].hist(bins=30, ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title('Income Distribution')
    axes[0,1].set_xlabel('Income ($)')
    
    # Scatter plot with regression line
    sns.regplot(data=df, x='education_years', y='income', ax=axes[1,0])
    axes[1,0].set_title('Education vs Income')
    
    # Box plot by category
    sns.boxplot(data=df, x='job_satisfaction', y='income', ax=axes[1,1])
    axes[1,1].set_title('Income by Job Satisfaction')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Generate and analyze data
sample_df = comprehensive_stats_overview()
```

**Hypothesis Testing:**
```python
def demonstrate_hypothesis_testing(df):
    """Demonstrate common hypothesis tests"""
    
    print("\n=== HYPOTHESIS TESTING ===")
    
    # T-test: Compare income between promoted and non-promoted
    promoted_income = df[df['promoted']]['income']
    not_promoted_income = df[~df['promoted']]['income']
    
    t_stat, p_value = stats.ttest_ind(promoted_income, not_promoted_income)
    
    print(f"T-test: Promoted vs Not Promoted Income")
    print(f"Promoted mean: ${promoted_income.mean():,.0f}")
    print(f"Not promoted mean: ${not_promoted_income.mean():,.0f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Chi-square test: Job satisfaction and promotion
    contingency_table = pd.crosstab(df['job_satisfaction'], df['promoted'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nChi-square test: Job Satisfaction vs Promotion")
    print("Contingency Table:")
    print(contingency_table)
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Significant association: {'Yes' if p_val < 0.05 else 'No'}")
    
    # ANOVA: Income across job satisfaction levels
    low_income = df[df['job_satisfaction'] == 'Low']['income']
    med_income = df[df['job_satisfaction'] == 'Medium']['income']
    high_income = df[df['job_satisfaction'] == 'High']['income']
    
    f_stat, p_val = stats.f_oneway(low_income, med_income, high_income)
    
    print(f"\nANOVA: Income across Job Satisfaction Levels")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_val:.6f}")
    print(f"Significant difference: {'Yes' if p_val < 0.05 else 'No'}")

demonstrate_hypothesis_testing(sample_df)
```

### Linear Regression with statsmodels

**Comprehensive Regression Analysis:**
```python
def regression_analysis_statsmodels(df):
    """Demonstrate comprehensive regression analysis"""
    
    print("\n=== LINEAR REGRESSION ANALYSIS ===")
    
    # Prepare data
    y = df['income']
    X = df[['age', 'education_years', 'experience']]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Display results
    print(model.summary())
    
    # Model diagnostics
    print("\n=== MODEL DIAGNOSTICS ===")
    
    # Residual analysis
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs fitted values
    axes[0,0].scatter(fitted_values, residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Fitted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted Values')
    
    # Q-Q plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot (Normality Check)')
    
    # Histogram of residuals
    axes[1,0].hist(residuals, bins=30, alpha=0.7)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Residuals')
    
    # Scale-location plot
    standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
    axes[1,1].scatter(fitted_values, standardized_residuals, alpha=0.6)
    axes[1,1].set_xlabel('Fitted Values')
    axes[1,1].set_ylabel('√|Standardized Residuals|')
    axes[1,1].set_title('Scale-Location Plot')
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    print("\n=== MODEL INTERPRETATION ===")
    
    # Extract coefficients
    coeffs = model.params
    p_values = model.pvalues
    
    for var in X.columns:
        if var == 'const':
            continue
        coeff = coeffs[var]
        p_val = p_values[var]
        significance = "significant" if p_val < 0.05 else "not significant"
        
        print(f"{var}: ${coeff:,.0f} per unit increase ({significance}, p={p_val:.4f})")
    
    print(f"\nR-squared: {model.rsquared:.3f} ({model.rsquared*100:.1f}% of variance explained)")
    print(f"Adjusted R-squared: {model.rsquared_adj:.3f}")
    print(f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.2e})")
    
    return model

# Run regression analysis
regression_model = regression_analysis_statsmodels(sample_df)
```

### Machine Learning Introduction

**scikit-learn Basics:**
```python
def machine_learning_introduction(df):
    """Introduction to machine learning with scikit-learn"""
    
    print("\n=== MACHINE LEARNING OVERVIEW ===")
    
    # Prepare data for ML
    features = ['age', 'education_years', 'experience', 'income']
    X = df[features]
    y = df['promoted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Positive class proportion: {y_train.mean():.2%}")
    
    # Linear Regression for continuous outcome
    print("\n--- Linear Regression (Income Prediction) ---")
    
    X_reg = df[['age', 'education_years', 'experience']]
    y_reg = df['income']
    
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    lr_model = LinearRegression()
    lr_model.fit(X_reg_train, y_reg_train)
    
    y_reg_pred = lr_model.predict(X_reg_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mse)
    
    print(f"Root Mean Square Error: ${rmse:,.0f}")
    print(f"R-squared Score: {lr_model.score(X_reg_test, y_reg_test):.3f}")
    
    # Logistic Regression for classification
    print("\n--- Logistic Regression (Promotion Prediction) ---")
    
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train_scaled, y_train)
    
    y_pred = log_model.predict(X_test_scaled)
    y_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\nFeature Importance (Logistic Regression Coefficients):")
    feature_importance = pd.DataFrame({
        'feature': features,
        'coefficient': log_model.coef_[0],
        'abs_coefficient': np.abs(log_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print(feature_importance)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Actual vs Predicted (Regression)
    axes[0].scatter(y_reg_test, y_reg_pred, alpha=0.6)
    axes[0].plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Income')
    axes[0].set_ylabel('Predicted Income')
    axes[0].set_title(f'Linear Regression\nRMSE: ${rmse:,.0f}')
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}')
    
    # Feature importance
    feature_importance.plot(x='feature', y='abs_coefficient', kind='bar', ax=axes[2])
    axes[2].set_title('Feature Importance')
    axes[2].set_ylabel('Absolute Coefficient')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return lr_model, log_model, scaler

# Run machine learning introduction
lr_model, log_model, scaler = machine_learning_introduction(sample_df)
```

### Time Series Analysis

**Basic Time Series Operations:**
```python
def time_series_analysis():
    """Demonstrate time series analysis concepts"""
    
    print("\n=== TIME SERIES ANALYSIS ===")
    
    # Generate sample time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Create realistic time series with trend and seasonality
    trend = np.linspace(100, 200, 1000)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
    noise = np.random.normal(0, 10, 1000)
    values = trend + seasonal + noise
    
    ts_df = pd.DataFrame({
        'date': dates,
        'value': values
    }).set_index('date')
    
    # Basic time series operations
    ts_df['7_day_ma'] = ts_df['value'].rolling(7).mean()
    ts_df['30_day_ma'] = ts_df['value'].rolling(30).mean()
    ts_df['yearly_change'] = ts_df['value'].pct_change(365)
    
    # Seasonal decomposition (simplified)
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    decomposition = seasonal_decompose(ts_df['value'], model='additive', period=365)
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Original time series
    ts_df[['value', '7_day_ma', '30_day_ma']].plot(ax=axes[0])
    axes[0].set_title('Original Time Series with Moving Averages')
    axes[0].set_ylabel('Value')
    
    # Trend component
    decomposition.trend.plot(ax=axes[1], color='red')
    axes[1].set_title('Trend Component')
    axes[1].set_ylabel('Trend')
    
    # Seasonal component
    decomposition.seasonal.plot(ax=axes[2], color='green')
    axes[2].set_title('Seasonal Component')
    axes[2].set_ylabel('Seasonal')
    
    # Residual component
    decomposition.resid.plot(ax=axes[3], color='purple')
    axes[3].set_title('Residual Component')
    axes[3].set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Basic forecasting (simple trend projection)
    # Fit linear trend to last 180 days
    recent_data = ts_df['value'].iloc[-180:]
    x = np.arange(len(recent_data))
    coeffs = np.polyfit(x, recent_data, 1)
    
    # Project 30 days into future
    future_x = np.arange(len(recent_data), len(recent_data) + 30)
    forecast = np.polyval(coeffs, future_x)
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ts_df['value'].iloc[-180:].plot(ax=ax, label='Historical Data')
    
    future_dates = pd.date_range(ts_df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
    pd.Series(forecast, index=future_dates).plot(ax=ax, label='Forecast', color='red', linestyle='--')
    
    ax.set_title('Simple Time Series Forecast')
    ax.set_ylabel('Value')
    ax.legend()
    plt.show()
    
    print("Time series analysis complete!")
    print(f"Trend slope: {coeffs[0]:.2f} units per day")
    print(f"30-day forecast range: {forecast.min():.1f} to {forecast.max():.1f}")

# Run time series analysis
time_series_analysis()
```

## Part 2: Professional Debugging and Error Handling

### Advanced Debugging Techniques

**Comprehensive Error Handling Framework:**
```python
import logging
import traceback
import sys
from functools import wraps
from typing import Any, Callable, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_science.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DataScienceError(Exception):
    """Base exception for data science operations"""
    pass

class DataValidationError(DataScienceError):
    """Raised when data validation fails"""
    pass

class ModelError(DataScienceError):
    """Raised when model operations fail"""
    pass

def debug_decorator(func: Callable) -> Callable:
    """
    Decorator for debugging function calls with timing and error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func_name} in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func_name} after {execution_time:.2f} seconds: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    return wrapper

class DataValidator:
    """Comprehensive data validation class"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[list] = None,
                          min_rows: int = 1) -> None:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            
        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if len(df) < min_rows:
            raise DataValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            logger.warning(f"Found completely empty columns: {empty_columns}")
        
        # Check for high missing data percentage
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            logger.warning(f"Columns with >50% missing data: {dict(high_missing)}")
    
    @staticmethod
    def validate_numeric_column(series: pd.Series, 
                               column_name: str,
                               allow_negative: bool = True,
                               min_value: Optional[float] = None,
                               max_value: Optional[float] = None) -> None:
        """
        Validate numeric column constraints.
        """
        if not pd.api.types.is_numeric_dtype(series):
            raise DataValidationError(f"Column {column_name} must be numeric")
        
        if not allow_negative and (series < 0).any():
            raise DataValidationError(f"Column {column_name} contains negative values")
        
        if min_value is not None and (series < min_value).any():
            raise DataValidationError(f"Column {column_name} contains values below {min_value}")
        
        if max_value is not None and (series > max_value).any():
            raise DataValidationError(f"Column {column_name} contains values above {max_value}")

class RobustDataProcessor:
    """
    Data processor with comprehensive error handling and debugging.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.validator = DataValidator()
    
    @debug_decorator
    def load_and_validate_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data with comprehensive error handling and validation.
        """
        try:
            # Attempt to load data
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath, **kwargs)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath, **kwargs)
            elif filepath.endswith('.json'):
                df = pd.read_json(filepath, **kwargs)
            else:
                raise DataValidationError(f"Unsupported file format: {filepath}")
            
            self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic validation
            self.validator.validate_dataframe(df)
            
            return df
            
        except FileNotFoundError:
            raise DataValidationError(f"File not found: {filepath}")
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"File is empty: {filepath}")
        except Exception as e:
            raise DataValidationError(f"Error loading {filepath}: {str(e)}")
    
    @debug_decorator
    def clean_data(self, df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   handle_missing: str = 'drop') -> pd.DataFrame:
        """
        Clean data with error handling and logging.
        """
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        try:
            # Handle duplicates
            if drop_duplicates:
                duplicates = df_clean.duplicated().sum()
                if duplicates > 0:
                    df_clean = df_clean.drop_duplicates()
                    self.logger.info(f"Removed {duplicates} duplicate rows")
            
            # Handle missing values
            if handle_missing == 'drop':
                missing_rows = df_clean.isnull().any(axis=1).sum()
                df_clean = df_clean.dropna()
                if missing_rows > 0:
                    self.logger.info(f"Dropped {missing_rows} rows with missing values")
            
            elif handle_missing == 'fill':
                for column in df_clean.columns:
                    if df_clean[column].isnull().any():
                        if df_clean[column].dtype in ['int64', 'float64']:
                            fill_value = df_clean[column].median()
                            df_clean[column].fillna(fill_value, inplace=True)
                            self.logger.info(f"Filled missing values in {column} with median: {fill_value}")
                        else:
                            fill_value = df_clean[column].mode().iloc[0] if not df_clean[column].mode().empty else 'Unknown'
                            df_clean[column].fillna(fill_value, inplace=True)
                            self.logger.info(f"Filled missing values in {column} with mode: {fill_value}")
            
            final_shape = df_clean.shape
            self.logger.info(f"Data cleaning complete: {initial_shape} → {final_shape}")
            
            return df_clean
            
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}")
            raise DataValidationError(f"Data cleaning failed: {str(e)}")
    
    @debug_decorator
    def perform_analysis(self, df: pd.DataFrame, 
                        target_column: str,
                        feature_columns: list) -> dict:
        """
        Perform analysis with comprehensive error handling.
        """
        try:
            # Validate inputs
            self.validator.validate_dataframe(df, required_columns=[target_column] + feature_columns)
            
            results = {}
            
            # Descriptive statistics
            results['descriptive_stats'] = df[feature_columns + [target_column]].describe()
            
            # Correlation analysis
            numeric_columns = df[feature_columns + [target_column]].select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 1:
                results['correlation_matrix'] = df[numeric_columns].corr()
            
            # Simple regression if target is numeric
            if pd.api.types.is_numeric_dtype(df[target_column]):
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                X = df[feature_columns].select_dtypes(include=[np.number])
                y = df[target_column]
                
                if not X.empty:
                    # Handle missing values
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) > 10:  # Minimum sample size
                        model = LinearRegression()
                        model.fit(X_clean, y_clean)
                        y_pred = model.predict(X_clean)
                        
                        results['regression_results'] = {
                            'r2_score': r2_score(y_clean, y_pred),
                            'coefficients': dict(zip(X_clean.columns, model.coef_)),
                            'intercept': model.intercept_
                        }
            
            self.logger.info("Analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise ModelError(f"Analysis failed: {str(e)}")

# Example usage of robust data processing
def demonstrate_robust_processing():
    """Demonstrate robust data processing with error handling"""
    
    processor = RobustDataProcessor(debug_mode=True)
    
    # Create sample data with various issues
    problematic_data = pd.DataFrame({
        'id': [1, 2, 2, 4, 5, 6, 7, 8, 9, 10],  # Duplicate
        'age': [25, 30, 30, np.nan, 35, 40, -5, 200, 45, 50],  # Missing and outliers  
        'income': [50000, 60000, 60000, 70000, np.nan, 80000, 90000, 100000, 110000, 120000],
        'score': [85, 90, 90, 95, 88, np.nan, 92, 87, 89, 91]
    })
    
    try:
        # Clean the data
        cleaned_data = processor.clean_data(
            problematic_data, 
            drop_duplicates=True, 
            handle_missing='fill'
        )
        
        # Perform analysis
        results = processor.perform_analysis(
            cleaned_data,
            target_column='score',
            feature_columns=['age', 'income']
        )
        
        print("\n=== ANALYSIS RESULTS ===")
        if 'regression_results' in results:
            r2 = results['regression_results']['r2_score']
            print(f"R-squared: {r2:.3f}")
            
            coeffs = results['regression_results']['coefficients']
            for feature, coeff in coeffs.items():
                print(f"{feature} coefficient: {coeff:.3f}")
        
        print("\nRobust processing demonstration completed successfully!")
        
    except (DataValidationError, ModelError) as e:
        print(f"Processing error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

# Run the demonstration
demonstrate_robust_processing()
```

### Performance Monitoring and Optimization

**Memory and Performance Monitoring:**
```python
import psutil
import time
from memory_profiler import profile
import gc

class PerformanceMonitor:
    """Monitor and optimize data science operations"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        """Start monitoring performance"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / 1024**2  # MB
        gc.collect()  # Clean up garbage
    
    def stop_monitoring(self, operation_name="Operation"):
        """Stop monitoring and report results"""
        if self.start_time is None:
            print("Monitoring not started")
            return
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / 1024**2  # MB
        
        execution_time = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        
        print(f"\n=== PERFORMANCE REPORT: {operation_name} ===")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory usage: {psutil.virtual_memory().used / 1024**2:.2f} MB")
        
        # Reset for next measurement
        self.start_time = None
        self.start_memory = None

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    
    print("=== MEMORY OPTIMIZATION ===")
    print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if col_min >= 0:  # Unsigned integers
            if col_max < 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif col_max < 65535:
                df_optimized[col] = df_optimized[col].astype('uint16')
            elif col_max < 4294967295:
                df_optimized[col] = df_optimized[col].astype('uint32')
        else:  # Signed integers
            if col_min > -128 and col_max < 127:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df_optimized[col] = df_optimized[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df_optimized[col] = df_optimized[col].astype('int32')
    
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Optimize object columns to category where appropriate
    for col in df_optimized.select_dtypes(include=['object']).columns:
        unique_ratio = df_optimized[col].nunique() / len(df_optimized)
        if unique_ratio < 0.5:  # If less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    memory_reduction = ((df.memory_usage(deep=True).sum() - df_optimized.memory_usage(deep=True).sum()) 
                       / df.memory_usage(deep=True).sum() * 100)
    print(f"Memory reduction: {memory_reduction:.1f}%")
    
    return df_optimized

# Demonstrate performance monitoring
monitor = PerformanceMonitor()

# Create large dataset for testing
monitor.start_monitoring()
large_df = pd.DataFrame({
    'id': range(100000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
    'value1': np.random.randn(100000),
    'value2': np.random.randint(0, 1000, 100000),
    'text': ['item_' + str(i) for i in range(100000)]
})
monitor.stop_monitoring("Large DataFrame Creation")

# Optimize memory
monitor.start_monitoring()
optimized_df = optimize_dataframe_memory(large_df)
monitor.stop_monitoring("Memory Optimization")
```

## Part 3: Capstone Project Framework

### Project Structure and Planning

**Complete Project Template:**
```python
class DataScienceProject:
    """
    Template for a complete data science project with best practices.
    """
    
    def __init__(self, project_name, data_path, output_dir='results'):
        self.project_name = project_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.logger = self._setup_logging()
        self.results = {}
        
        # Create output directory structure
        import os
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    def _setup_logging(self):
        """Setup project-specific logging"""
        logger = logging.getLogger(f"{self.project_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(f'{self.project_name}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def load_data(self):
        """Load and validate project data"""
        self.logger.info("Loading project data")
        
        try:
            self.data = pd.read_csv(self.data_path)
            self.logger.info(f"Data loaded: {self.data.shape}")
            
            # Basic data quality check
            self.results['data_quality'] = {
                'shape': self.data.shape,
                'missing_values': self.data.isnull().sum().to_dict(),
                'duplicates': self.data.duplicated().sum(),
                'dtypes': self.data.dtypes.to_dict()
            }
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self):
        """Comprehensive exploratory data analysis"""
        self.logger.info("Starting exploratory data analysis")
        
        # Descriptive statistics
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        self.results['eda'] = {
            'numeric_summary': self.data[numeric_columns].describe().to_dict(),
            'categorical_summary': {col: self.data[col].value_counts().to_dict() 
                                  for col in categorical_columns}
        }
        
        # Create EDA visualizations
        n_numeric = len(numeric_columns)
        if n_numeric > 0:
            n_cols = min(3, n_numeric)
            n_rows = (n_numeric + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    self.data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Correlation analysis
        if len(numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.data[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.savefig(f'{self.output_dir}/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            self.results['eda']['correlations'] = correlation_matrix.to_dict()
    
    def build_model(self, target_column, feature_columns, model_type='regression'):
        """Build and evaluate model"""
        self.logger.info(f"Building {model_type} model")
        
        # Prepare data
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.median() if X.dtypes.name in ['int64', 'float64'] else X.mode().iloc[0])
        y = y.fillna(y.median() if y.dtype.name in ['int64', 'float64'] else y.mode().iloc[0])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features if needed
        if model_type in ['logistic', 'svm']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Build model
        if model_type == 'regression':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test_scaled, y_test)
            
            self.results['model'] = {
                'type': 'regression',
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'coefficients': dict(zip(feature_columns, model.coef_)),
                'intercept': model.intercept_
            }
            
            # Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Regression Results (R² = {r2:.3f})')
            plt.savefig(f'{self.output_dir}/figures/regression_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        elif model_type == 'classification':
            model = LogisticRegression(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results['model'] = {
                'type': 'classification',
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Accuracy = {accuracy:.3f})')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(f'{self.output_dir}/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        self.model = model
        return model
    
    def generate_report(self):
        """Generate comprehensive project report"""
        self.logger.info("Generating project report")
        
        report = f"""
# {self.project_name} - Data Science Project Report

## Executive Summary
This report presents the findings from the {self.project_name} data science project.

## Data Overview
- **Dataset shape**: {self.results['data_quality']['shape']}
- **Missing values**: {sum(self.results['data_quality']['missing_values'].values())} total
- **Duplicate rows**: {self.results['data_quality']['duplicates']}

## Key Findings

### Data Quality
"""
        
        if 'eda' in self.results:
            report += f"""
### Exploratory Data Analysis
- **Numeric variables**: {len(self.results['eda']['numeric_summary'])}
- **Categorical variables**: {len(self.results['eda']['categorical_summary'])}
"""
        
        if 'model' in self.results:
            model_results = self.results['model']
            if model_results['type'] == 'regression':
                report += f"""
### Model Performance (Regression)
- **R-squared**: {model_results['r2']:.3f}
- **RMSE**: {model_results['rmse']:.2f}
- **Key predictors**: {', '.join([k for k, v in model_results['coefficients'].items() if abs(v) > 0.1])}
"""
            elif model_results['type'] == 'classification':
                report += f"""
### Model Performance (Classification)
- **Accuracy**: {model_results['accuracy']:.3f}
- **F1-score**: {model_results['classification_report']['weighted avg']['f1-score']:.3f}
"""
        
        report += """
## Recommendations
1. Based on the analysis, consider focusing on the key predictors identified
2. Monitor data quality regularly to maintain model performance
3. Consider collecting additional data on underperforming segments

## Next Steps
1. Deploy model to production environment
2. Set up monitoring and alerting systems
3. Plan for regular model retraining

---
*Report generated automatically using DataScienceProject framework*
"""
        
        # Save report
        with open(f'{self.output_dir}/reports/{self.project_name}_report.md', 'w') as f:
            f.write(report)
        
        print("Report generated and saved!")
        print(report)
        
        return report

# Example usage of the project framework
def run_capstone_project():
    """Demonstrate the complete project framework"""
    
    # Create sample dataset for demonstration
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'age': np.random.normal(35, 10, 1000).clip(18, 70),
        'income': np.random.normal(50000, 15000, 1000).clip(20000, 150000),
        'education_years': np.random.normal(14, 3, 1000).clip(8, 20),
        'spending': np.random.normal(2000, 500, 1000).clip(500, 5000),
        'satisfaction_score': np.random.uniform(1, 10, 1000)
    })
    
    # Add relationships
    sample_data['spending'] = (sample_data['income'] * 0.02 + 
                              sample_data['age'] * 20 + 
                              np.random.normal(0, 200, 1000)).clip(500, 5000)
    
    # Save sample data
    sample_data.to_csv('sample_customer_data.csv', index=False)
    
    # Initialize project
    project = DataScienceProject(
        project_name="Customer_Spending_Analysis",
        data_path="sample_customer_data.csv"
    )
    
    # Run complete analysis
    project.load_data()
    project.explore_data()
    
    # Build regression model
    model = project.build_model(
        target_column='spending',
        feature_columns=['age', 'income', 'education_years', 'satisfaction_score'],
        model_type='regression'
    )
    
    # Generate report
    report = project.generate_report()
    
    print("\n=== CAPSTONE PROJECT COMPLETE ===")
    print("Check the 'results' directory for generated files!")

# Run the capstone project demonstration
run_capstone_project()
```

## Part 4: Deployment and Sharing

### Creating Shareable Data Science Work

**Jupyter Notebook Best Practices:**
```python
def create_professional_notebook():
    """Guidelines for professional Jupyter notebooks"""
    
    notebook_template = '''
# Professional Data Science Notebook Template

## 1. Executive Summary
- Clear problem statement
- Key findings in 2-3 bullet points
- Business impact/recommendations

## 2. Setup and Imports
```python
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_columns', 100)
np.random.seed(42)

# Custom functions
def load_and_validate_data(filepath):
    """Load data with basic validation"""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df
```

## 3. Data Loading and Initial Exploration
- Document data sources
- Show data quality assessment
- Include basic statistics

## 4. Exploratory Data Analysis
- Include meaningful visualizations
- Document insights found
- Show statistical relationships

## 5. Methodology
- Explain approach and rationale
- Document assumptions
- Show model selection process

## 6. Results and Evaluation
- Present key metrics
- Include visualizations
- Discuss limitations

## 7. Conclusions and Recommendations
- Summarize findings
- Provide actionable recommendations
- Suggest next steps

## 8. Appendix
- Additional technical details
- Code for complex functions
- Extended analysis
'''
    
    print("=== JUPYTER NOTEBOOK BEST PRACTICES ===")
    print("1. Start with executive summary")
    print("2. Use clear section headers")
    print("3. Document your thought process")
    print("4. Include meaningful visualizations")
    print("5. End with actionable insights")
    print("6. Clean up code before sharing")
    print("7. Test notebook from start to finish")
    
    return notebook_template

# Display the template
template = create_professional_notebook()
```

**Creating Interactive Dashboards:**
```python
def create_interactive_dashboard():
    """Example of creating an interactive dashboard with Plotly"""
    
    # Note: This requires plotly installation
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        # Sample data
        df = pd.DataFrame({
            'month': pd.date_range('2023-01-01', periods=12, freq='M'),
            'sales': [100, 120, 140, 130, 160, 180, 170, 190, 200, 180, 220, 250],
            'marketing_spend': [20, 25, 30, 28, 35, 40, 38, 42, 45, 40, 48, 55],
            'customer_count': [1000, 1100, 1250, 1200, 1400, 1500, 1450, 1600, 1700, 1550, 1800, 2000]
        })
        
        # Create subplot dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sales Trend', 'Marketing ROI', 'Customer Growth', 'Key Metrics'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Sales trend
        fig.add_trace(
            go.Scatter(x=df['month'], y=df['sales'], mode='lines+markers', name='Sales'),
            row=1, col=1
        )
        
        # Marketing ROI scatter
        fig.add_trace(
            go.Scatter(x=df['marketing_spend'], y=df['sales'], mode='markers', 
                      name='Marketing ROI', showlegend=False),
            row=1, col=2
        )
        
        # Customer growth bar
        fig.add_trace(
            go.Bar(x=df['month'], y=df['customer_count'], name='Customers', showlegend=False),
            row=2, col=1
        )
        
        # Key metric indicator
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=df['sales'].iloc[-1],
                delta={'reference': df['sales'].iloc[-2]},
                title={'text': "Current Month Sales"},
                gauge={'axis': {'range': [0, 300]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 150], 'color': "lightgray"},
                                {'range': [150, 250], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 200}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Business Dashboard")
        fig.show()
        
        print("Interactive dashboard created successfully!")
        print("To save as HTML: fig.write_html('dashboard.html')")
        
    except ImportError:
        print("Plotly not installed. Run: pip install plotly")
        print("Dashboard creation skipped.")

# Create the dashboard
create_interactive_dashboard()
```

### Version Control for Data Science

**Git Workflow for Data Science Projects:**
```python
def setup_data_science_git_workflow():
    """Best practices for Git in data science projects"""
    
    gitignore_template = '''
# Data Science .gitignore template

# Data files
*.csv
*.xlsx
*.json
*.parquet
*.h5
*.pkl
data/raw/
data/processed/
!data/sample/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
.venv/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Model files (optional - depends on size)
*.model
*.pkl
models/

# Results and outputs
results/
outputs/
figures/
reports/

# Secrets
.env
secrets.txt
api_keys.txt
'''
    
    best_practices = '''
=== GIT WORKFLOW FOR DATA SCIENCE ===

1. Repository Structure:
   - Keep raw data out of version control
   - Track code, not large files
   - Use Git LFS for necessary large files
   - Version control your environment (requirements.txt)

2. Commit Practices:
   - Commit early and often
   - Write meaningful commit messages
   - One logical change per commit
   - Clean notebooks before committing

3. Branching Strategy:
   - main: Production-ready code
   - develop: Integration branch
   - feature/analysis-name: Specific analyses
   - hotfix/issue-name: Critical fixes

4. Jupyter Notebook Management:
   - Clear outputs before committing
   - Use nbstripout to automate this
   - Consider using .py files for production code

5. Data Management:
   - Use DVC (Data Version Control) for large datasets
   - Document data sources and versions
   - Keep data processing scripts in version control

6. Collaboration:
   - Use pull requests for code review
   - Tag stable versions
   - Document changes in CHANGELOG.md
'''
    
    commands = '''
=== USEFUL GIT COMMANDS FOR DATA SCIENCE ===

# Setup
git init
git remote add origin <repository-url>

# Daily workflow
git status
git add <files>
git commit -m "Descriptive message"
git push origin <branch-name>

# Branching
git checkout -b feature/new-analysis
git merge main
git branch -d feature/completed-analysis

# Notebook management
pip install nbstripout
nbstripout --install  # Auto-strip notebook outputs

# Large files
git lfs track "*.csv"
git lfs track "*.model"
git add .gitattributes

# Tagging releases
git tag -a v1.0 -m "First stable model"
git push origin v1.0
'''
    
    print(best_practices)
    print(commands)
    
    # Save .gitignore template
    with open('.gitignore_template', 'w') as f:
        f.write(gitignore_template)
    
    print("\n.gitignore template saved as '.gitignore_template'")
    print("Copy this to your project root as '.gitignore'")

# Run the Git setup
setup_data_science_git_workflow()
```

## Part 5: Career Development and Next Steps

### Data Science Career Paths

**Understanding Different Roles:**
```python
def explore_data_science_careers():
    """Overview of data science career paths and requirements"""
    
    career_paths = {
        'Data Analyst': {
            'description': 'Focus on descriptive and diagnostic analytics',
            'key_skills': ['SQL', 'Excel', 'Python/R', 'Tableau/PowerBI', 'Statistics'],
            'typical_tasks': ['Creating dashboards', 'A/B testing', 'Reporting', 'Data cleaning'],
            'growth_path': 'Senior Analyst → Analytics Manager → Director of Analytics'
        },
        
        'Data Scientist': {
            'description': 'Build predictive models and extract insights',
            'key_skills': ['Python/R', 'Machine Learning', 'Statistics', 'SQL', 'Domain expertise'],
            'typical_tasks': ['Model building', 'Feature engineering', 'Hypothesis testing', 'Research'],
            'growth_path': 'Senior Data Scientist → Principal Data Scientist → Head of Data Science'
        },
        
        'Machine Learning Engineer': {
            'description': 'Deploy and maintain ML systems in production',
            'key_skills': ['Python/Java', 'ML frameworks', 'Cloud platforms', 'DevOps', 'Software engineering'],
            'typical_tasks': ['Model deployment', 'Pipeline automation', 'Performance monitoring', 'Scaling'],
            'growth_path': 'Senior MLE → Staff MLE → Engineering Manager'
        },
        
        'Data Engineer': {
            'description': 'Build and maintain data infrastructure',
            'key_skills': ['SQL', 'Python/Scala', 'Spark', 'Cloud platforms', 'ETL tools'],
            'typical_tasks': ['Data pipelines', 'Database design', 'Data warehousing', 'System optimization'],
            'growth_path': 'Senior Data Engineer → Principal Engineer → Head of Data Engineering'
        },
        
        'Research Scientist': {
            'description': 'Advance state-of-the-art in AI/ML',
            'key_skills': ['Advanced math', 'Deep learning', 'Research methodology', 'Publications', 'PhD often required'],
            'typical_tasks': ['Algorithm development', 'Research papers', 'Prototyping', 'Collaboration with academia'],
            'growth_path': 'Senior Researcher → Principal Researcher → Research Director'
        }
    }
    
    print("=== DATA SCIENCE CAREER PATHS ===\n")
    
    for role, details in career_paths.items():
        print(f"**{role}**")
        print(f"Description: {details['description']}")
        print(f"Key Skills: {', '.join(details['key_skills'])}")
        print(f"Growth Path: {details['growth_path']}")
        print()
    
    # Skills assessment
    skills_checklist = {
        'Programming': ['Python', 'R', 'SQL', 'Git'],
        'Statistics & ML': ['Descriptive statistics', 'Hypothesis testing', 'Regression', 'Classification', 'Clustering'],
        'Tools & Platforms': ['Jupyter', 'Pandas/NumPy', 'Scikit-learn', 'Matplotlib/Seaborn', 'Cloud platforms'],
        'Business Skills': ['Communication', 'Domain expertise', 'Problem-solving', 'Project management'],
        'Advanced (Optional)': ['Deep learning', 'Big data tools', 'Advanced math', 'Software engineering']
    }
    
    print("=== SKILLS SELF-ASSESSMENT ===")
    print("Rate yourself (1-5) on these skills:\n")
    
    for category, skills in skills_checklist.items():
        print(f"**{category}:**")
        for skill in skills:
            print(f"  □ {skill}: ___/5")
        print()
    
    return career_paths

# Explore career paths
career_info = explore_data_science_careers()
```

### Building a Portfolio

**Portfolio Project Guidelines:**
```python
def create_portfolio_guidelines():
    """Guidelines for building a data science portfolio"""
    
    portfolio_structure = '''
=== DATA SCIENCE PORTFOLIO STRUCTURE ===

1. **Personal Website/GitHub Profile**
   - Professional headshot and bio
   - Clear navigation
   - Contact information
   - Link to resume/CV

2. **3-5 Diverse Projects**
   - Different domains (business, healthcare, finance, etc.)
   - Various techniques (EDA, ML, time series, etc.)
   - Mix of supervised and unsupervised learning
   - At least one end-to-end deployed project

3. **Each Project Should Include:**
   - Clear problem statement
   - Data source and description
   - Methodology and approach
   - Results and insights
   - Code repository (clean and documented)
   - Optional: Blog post explaining the project

4. **Recommended Project Types:**
   - **Exploratory Analysis**: Deep dive into interesting dataset
   - **Predictive Modeling**: Classification or regression problem
   - **Business Case Study**: Real-world business problem
   - **Time Series Analysis**: Forecasting or trend analysis
   - **Web Scraping + Analysis**: Collect and analyze your own data
'''
    
    project_template = '''
=== PORTFOLIO PROJECT TEMPLATE ===

## Project Title: [Descriptive Name]

### Executive Summary
- Problem statement in 1-2 sentences
- Key findings and business impact
- Technologies used

### Data
- Source and collection method
- Size and scope
- Key variables and target

### Methodology
1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Model selection and training
5. Evaluation and validation

### Results
- Key metrics and performance
- Important insights
- Visualizations and tables

### Business Impact
- Recommendations
- Potential value/savings
- Next steps

### Technical Details
- **Languages**: Python, SQL
- **Libraries**: pandas, scikit-learn, matplotlib
- **Models**: Linear Regression, Random Forest
- **Deployment**: Streamlit app, GitHub Pages

### Code Repository
[Link to GitHub repository with clean, commented code]

### Live Demo
[Link to deployed application if applicable]
'''
    
    tips = '''
=== PORTFOLIO TIPS ===

**Technical Excellence:**
- Write clean, well-commented code
- Include proper error handling
- Use version control effectively
- Document your process thoroughly

**Storytelling:**
- Start with business context
- Walk through your thought process
- Explain technical choices
- End with actionable insights

**Presentation:**
- Use professional visualizations
- Include interactive elements where possible
- Make it mobile-friendly
- Optimize loading times

**Differentiation:**
- Choose unique/interesting datasets
- Show domain expertise
- Demonstrate end-to-end skills
- Include deployed applications

**Common Mistakes to Avoid:**
- Using overused datasets (Titanic, Iris, etc.)
- Showing only accuracy metrics
- Not explaining business context
- Having messy, uncommented code
- Not updating projects over time
'''
    
    print(portfolio_structure)
    print(project_template)
    print(tips)
    
    return portfolio_structure, project_template

# Create portfolio guidelines
portfolio_info = create_portfolio_guidelines()
```

### Continuous Learning Plan

**Learning Roadmap:**
```python
def create_learning_roadmap():
    """Structured plan for continued learning in data science"""
    
    learning_phases = {
        'Phase 1: Strengthen Foundations (Months 1-3)': {
            'focus': 'Master core skills and build confidence',
            'topics': [
                'Advanced pandas operations',
                'Statistical inference and hypothesis testing',
                'More machine learning algorithms (SVM, Random Forest, etc.)',
                'Feature engineering techniques',
                'Model evaluation and validation'
            ],
            'projects': [
                'Complete 2-3 end-to-end projects',
                'Participate in Kaggle competitions',
                'Build first portfolio website'
            ],
            'resources': [
                'Hands-On Machine Learning by Aurélien Géron',
                'Kaggle Learn courses',
                'Fast.ai Practical Deep Learning course'
            ]
        },
        
        'Phase 2: Specialization (Months 4-8)': {
            'focus': 'Choose and develop expertise in specific areas',
            'topics': [
                'Deep learning (if interested in AI/CV/NLP)',
                'Time series forecasting (for business analytics)',
                'A/B testing and causal inference (for product analytics)',
                'Big data tools (Spark, Hadoop) (for data engineering)',
                'Cloud platforms (AWS, GCP, Azure)'
            ],
            'projects': [
                'Build specialized projects in chosen area',
                'Contribute to open source projects',
                'Start a technical blog'
            ],
            'resources': [
                'Deep Learning by Ian Goodfellow (for AI track)',
                'Forecasting: Principles and Practice (for time series)',
                'Cloud provider documentation and tutorials'
            ]
        },
        
        'Phase 3: Professional Development (Months 9-12)': {
            'focus': 'Industry readiness and career advancement',
            'topics': [
                'MLOps and model deployment',
                'Data engineering and pipeline design',
                'Advanced business analytics',
                'Leadership and communication skills',
                'Industry-specific knowledge'
            ],
            'projects': [
                'Deploy a model to production',
                'Lead a team project or mentorship',
                'Present at meetups or conferences'
            ],
            'resources': [
                'Building Machine Learning Powered Applications',
                'Industry conferences and meetups',
                'Professional networking'
            ]
        },
        
        'Ongoing: Stay Current': {
            'focus': 'Keep up with rapidly evolving field',
            'activities': [
                'Read research papers (2-3 per month)',
                'Follow industry leaders on social media',
                'Attend virtual conferences and webinars',
                'Participate in online communities',
                'Experiment with new tools and techniques'
            ],
            'resources': [
                'arXiv.org for research papers',
                'Twitter/LinkedIn for industry updates',
                'Reddit r/MachineLearning, r/datascience',
                'Towards Data Science on Medium'
            ]
        }
    }
    
    practical_steps = '''
=== PRACTICAL NEXT STEPS ===

**This Week:**
1. Complete one of the practice exercises from this course
2. Set up a GitHub profile with your first project
3. Join 2-3 data science communities online
4. Start following 5 data science professionals on LinkedIn

**This Month:**
1. Complete a full end-to-end project
2. Write a blog post about your learning journey
3. Apply to 3 entry-level positions or internships
4. Network with local data science professionals

**Next 3 Months:**
1. Build 3 diverse portfolio projects
2. Take an advanced online course
3. Participate in a Kaggle competition
4. Attend a data science meetup or conference

**Resources for Continued Learning:**
- **Books**: "Hands-On Machine Learning", "Python Data Science Handbook"
- **Courses**: Coursera, edX, Udacity nanodegrees
- **Platforms**: Kaggle, GitHub, Stack Overflow
- **Communities**: Local meetups, Discord servers, Reddit
- **Conferences**: NeurIPS, ICML, KDD (virtual options available)
'''
    
    print("=== DATA SCIENCE LEARNING ROADMAP ===\n")
    
    for phase, details in learning_phases.items():
        print(f"**{phase}**")
        print(f"Focus: {details['focus']}")
        
        if 'topics' in details:
            print("Topics:")
            for topic in details['topics']:
                print(f"  • {topic}")
        
        if 'projects' in details:
            print("Projects:")
            for project in details['projects']:
                print(f"  • {project}")
        
        if 'resources' in details:
            print("Resources:")
            for resource in details['resources']:
                print(f"  • {resource}")
        
        if 'activities' in details:
            print("Activities:")
            for activity in details['activities']:
                print(f"  • {activity}")
        
        print()
    
    print(practical_steps)
    
    return learning_phases

# Create learning roadmap
roadmap = create_learning_roadmap()
```

## Final Project Demonstration

Let me create a complete, integrated example that brings together all concepts:

```python
def final_integrated_example():
    """
    Complete example integrating all course concepts:
    - Data loading and cleaning
    - Exploratory data analysis
    - Statistical testing
    - Machine learning
    - Visualization
    - Professional documentation
    """
    
    print("=== INTEGRATED DATA SCIENCE PROJECT ===")
    print("Analyzing Customer Behavior and Predicting Churn\n")
    
    # Step 1: Generate realistic customer data
    np.random.seed(42)
    n_customers = 2000
    
    # Create customer segments
    segments = ['Premium', 'Standard', 'Basic']
    segment_probs = [0.2, 0.5, 0.3]
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'segment': np.random.choice(segments, n_customers, p=segment_probs),
        'tenure_months': np.random.exponential(24, n_customers).astype(int).clip(1, 120),
        'monthly_charges': np.random.normal(70, 25, n_customers).clip(10, 200),
        'support_tickets': np.random.poisson(2, n_customers),
        'satisfaction_score': np.random.normal(7, 2, n_customers).clip(1, 10),
        'age': np.random.normal(40, 15, n_customers).astype(int).clip(18, 80),
        'num_services': np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    }
    
    # Create realistic relationships
    df = pd.DataFrame(data)
    
    # Premium customers tend to have higher charges and satisfaction
    premium_mask = df['segment'] == 'Premium'
    df.loc[premium_mask, 'monthly_charges'] *= 1.5
    df.loc[premium_mask, 'satisfaction_score'] += 1
    
    # Calculate churn probability based on multiple factors
    churn_prob = (
        0.1 +  # Base probability
        np.where(df['satisfaction_score'] < 5, 0.4, 0) +  # Low satisfaction
        np.where(df['support_tickets'] > 5, 0.3, 0) +     # High support usage
        np.where(df['tenure_months'] < 6, 0.3, 0) +       # New customers
        np.where(df['monthly_charges'] > 120, 0.2, 0) -   # High charges increase churn
        np.where(df['segment'] == 'Premium', 0.2, 0)      # Premium less likely to churn
    ).clip(0, 0.9)
    
    df['churned'] = np.random.binomial(1, churn_prob, n_customers)
    
    print(f"Dataset created: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    
    # Step 2: Exploratory Data Analysis
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # Basic statistics
    print("\nDescriptive Statistics:")
    print(df.describe().round(2))
    
    # Churn by segment
    print("\nChurn Rate by Segment:")
    churn_by_segment = df.groupby('segment')['churned'].agg(['count', 'sum', 'mean'])
    churn_by_segment.columns = ['Total', 'Churned', 'Churn_Rate']
    churn_by_segment['Churn_Rate'] = (churn_by_segment['Churn_Rate'] * 100).round(1)
    print(churn_by_segment)
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Distribution of tenure
    df['tenure_months'].hist(bins=30, ax=axes[0,0], alpha=0.7)
    axes[0,0].set_title('Distribution of Tenure (Months)')
    axes[0,0].set_xlabel('Tenure (Months)')
    
    # Churn by satisfaction score
    satisfaction_churn = df.groupby('satisfaction_score')['churned'].mean()
    satisfaction_churn.plot(kind='bar', ax=axes[0,1], alpha=0.7)
    axes[0,1].set_title('Churn Rate by Satisfaction Score')
    axes[0,1].set_ylabel('Churn Rate')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Monthly charges vs satisfaction (colored by churn)
    colors = ['blue' if x == 0 else 'red' for x in df['churned']]
    axes[0,2].scatter(df['monthly_charges'], df['satisfaction_score'], 
                      c=colors, alpha=0.6, s=20)
    axes[0,2].set_xlabel('Monthly Charges ($)')
    axes[0,2].set_ylabel('Satisfaction Score')
    axes[0,2].set_title('Charges vs Satisfaction (Red = Churned)')
    
    # Support tickets distribution
    df.boxplot(column='support_tickets', by='churned', ax=axes[1,0])
    axes[1,0].set_title('Support Tickets by Churn Status')
    
    # Tenure distribution by churn
    df[df['churned'] == 0]['tenure_months'].hist(bins=20, alpha=0.7, 
                                                  label='Retained', ax=axes[1,1])
    df[df['churned'] == 1]['tenure_months'].hist(bins=20, alpha=0.7, 
                                                  label='Churned', ax=axes[1,1])
    axes[1,1].set_title('Tenure Distribution by Churn')
    axes[1,1].legend()
    
    # Correlation heatmap
    numeric_cols = ['tenure_months', 'monthly_charges', 'support_tickets', 
                   'satisfaction_score', 'age', 'num_services', 'churned']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
    axes[1,2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Step 3: Statistical Testing
    print("\n=== STATISTICAL ANALYSIS ===")
    
    # T-test: Compare satisfaction scores between churned and retained
    retained = df[df['churned'] == 0]['satisfaction_score']
    churned = df[df['churned'] == 1]['satisfaction_score']
    
    t_stat, p_value = stats.ttest_ind(retained, churned)
    print(f"T-test: Satisfaction Score Difference")
    print(f"Retained mean: {retained.mean():.2f}")
    print(f"Churned mean: {churned.mean():.2f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Chi-square test: Segment and churn relationship
    contingency = pd.crosstab(df['segment'], df['churned'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test: Segment vs Churn")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"P-value: {p_val:.2e}")
    
    # Step 4: Machine Learning Model
    print("\n=== MACHINE LEARNING MODEL ===")
    
    # Prepare features
    feature_columns = ['tenure_months', 'monthly_charges', 'support_tickets', 
                      'satisfaction_score', 'age', 'num_services']
    
    # One-hot encode categorical variables
    df_ml = pd.get_dummies(df, columns=['segment'], prefix='segment')
    
    feature_columns.extend(['segment_Basic', 'segment_Premium', 'segment_Standard'])
    
    X = df_ml[feature_columns]
    y = df_ml['churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'model': model
        }
        
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
    
    # Feature importance (Random Forest)
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(feature_importance.round(3))
    
    # Step 5: Business Insights and Recommendations
    print("\n=== BUSINESS INSIGHTS ===")
    
    # Calculate customer lifetime value impact
    avg_monthly_revenue = df['monthly_charges'].mean()
    avg_tenure = df[df['churned'] == 0]['tenure_months'].mean()
    
    # Segment analysis
    segment_analysis = df.groupby('segment').agg({
        'churned': ['count', 'sum', 'mean'],
        'monthly_charges': 'mean',
        'satisfaction_score': 'mean',
        'tenure_months': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['Total_Customers', 'Churned_Count', 'Churn_Rate',
                               'Avg_Monthly_Charges', 'Avg_Satisfaction', 'Avg_Tenure']
    
    print("\nSegment Analysis:")
    print(segment_analysis)
    
    # High-risk customers
    high_risk_threshold = 0.7
    best_model = results['Random Forest']
    high_risk_customers = X_test[best_model['probabilities'] > high_risk_threshold]
    
    print(f"\nHigh-Risk Customers Identified: {len(high_risk_customers)}")
    print(f"Potential Monthly Revenue at Risk: ${len(high_risk_customers) * avg_monthly_revenue:,.0f}")
    
    recommendations = """
    BUSINESS RECOMMENDATIONS:
    
    1. IMMEDIATE ACTIONS:
       • Target {high_risk_count} high-risk customers with retention campaigns
       • Focus on customers with satisfaction scores < 5
       • Implement proactive support for customers with >5 tickets
    
    2. STRATEGIC INITIATIVES:
       • Improve satisfaction scores across all segments
       • Enhance onboarding for new customers (< 6 months tenure)
       • Review pricing strategy for high-charge customers
    
    3. MONITORING:
       • Track satisfaction scores monthly
       • Monitor support ticket volume trends
       • Implement churn prediction dashboard
    
    4. ESTIMATED IMPACT:
       • Reducing churn by 20% could save ${potential_savings:,.0f} in annual revenue
       • Improving satisfaction by 1 point could reduce churn by ~15%
    """.format(
        high_risk_count=len(high_risk_customers),
        potential_savings=len(high_risk_customers) * avg_monthly_revenue * 12 * 0.2
    )
    
    print(recommendations)
    
    return df, results, feature_importance

# Run the complete integrated example
final_data, model_results, importance = final_integrated_example()
```

## Summary: Your Data Science Journey

### What You've Accomplished

Over these five intensive lectures, you've built a comprehensive foundation in data science:

1. **Technical Mastery**: Python programming, command line proficiency, version control
2. **Data Skills**: NumPy arrays, Pandas DataFrames, data cleaning and validation
3. **Analysis Capabilities**: Statistical testing, machine learning, time series analysis
4. **Visualization Excellence**: Effective charts, design principles, storytelling
5. **Professional Practices**: Debugging, error handling, project organization, deployment

### Key Principles to Remember

- **Data Quality First**: Always validate and understand your data before analysis
- **Reproducible Research**: Document your process, version control your work
- **Clear Communication**: Your insights are only valuable if others can understand them
- **Continuous Learning**: The field evolves rapidly - stay curious and keep learning
- **Business Focus**: Technical skills serve business objectives and human needs

### Next Steps in Your Journey

1. **Immediate Actions** (Next 2 weeks):
   - Complete one comprehensive project using all course concepts
   - Set up your GitHub profile and upload your first project
   - Join 2-3 data science communities

2. **Short-term Goals** (Next 3 months):
   - Build a portfolio with 3-5 diverse projects
   - Apply to entry-level positions or internships
   - Network with professionals in your area of interest

3. **Long-term Development** (Next year):
   - Specialize in an area that interests you
   - Contribute to open source projects
   - Consider advanced education or certifications

### Final Thoughts

Data science is ultimately about using data to make better decisions and solve important problems. The technical skills you've learned are tools in service of this higher purpose. Whether you're helping businesses grow, improving healthcare outcomes, or addressing social challenges, your work as a data scientist can have real impact.

Remember that becoming proficient in data science is a journey, not a destination. The field is constantly evolving, with new techniques, tools, and applications emerging regularly. Embrace this continuous learning mindset, stay curious about the world around you, and always ask "What story is this data telling, and how can it help?"

Congratulations on completing this intensive data science program. You now have the foundation to tackle real-world data challenges and continue growing as a data scientist. The journey ahead is exciting - go forth and discover insights that matter!

---

*"In God we trust. All others must bring data."* - W. Edwards Deming

Your data science adventure begins now. Make it count.