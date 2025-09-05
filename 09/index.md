# Lecture 09: Automation + Professional Development Practices

**Duration**: 4.5 hours  
**Level**: Advanced Professional Track  
**Prerequisites**: L06-L08 Advanced Analytics Foundation

## Professional Context: From Analysis to Production Systems

The transition from working analytical code to production-ready systems represents a crucial professional milestone. In today's data-driven organizations, analysts and data scientists must deliver:

- **Reproducible workflows**: Analyses that produce consistent results across environments
- **Automated pipelines**: Systems that run reliably without manual intervention
- **Collaborative codebases**: Code that multiple team members can understand and maintain
- **Robust error handling**: Systems that gracefully handle edge cases and failures
- **Professional documentation**: Clear guidance for users and future developers

In clinical research and healthcare analytics, these practices become even more critical:
- **Regulatory compliance**: FDA/EMA validation requirements for analytical software
- **Patient safety**: Code errors can impact clinical decision-making
- **Audit trails**: Complete documentation of analytical processes
- **Scalability**: Systems that grow with organizational needs

Today we master the professional development practices that transform data scientists into trusted technology partners.

## Learning Objectives

By the end of this lecture, you will:

1. **Design and implement automated analytical workflows** using modern orchestration tools
2. **Apply professional code quality standards** including testing, documentation, and version control
3. **Build reproducible environments** using containerization and dependency management
4. **Create robust error handling** and logging systems for production analytics
5. **Implement continuous integration/deployment** for analytical workflows
6. **Establish professional development workflows** for collaborative data science teams

---

## Part 1: Workflow Automation and Orchestration (75 minutes)

### Professional Challenge: Automated Clinical Research Pipeline

Consider a multi-site clinical trial requiring:
- **Daily data ingestion** from multiple EMR systems
- **Real-time quality monitoring** with automated alerts
- **Weekly statistical reports** for data monitoring committees
- **Quarterly regulatory submissions** with validated analyses

Manual execution is error-prone and unsustainable. Professional automation requires:
- **Orchestration platforms**: Managing complex workflows with dependencies
- **Error recovery**: Handling failures gracefully without data loss
- **Monitoring and alerting**: Proactive notification of pipeline issues
- **Scalable architecture**: Growing with increasing data volumes

### Advanced Workflow Automation

#### 1. Professional Workflow Design with Apache Airflow

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import pandas as pd
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTrialPipeline:
    """Professional workflow automation for clinical trial data processing."""
    
    def __init__(self, study_id: str, base_path: str):
        self.study_id = study_id
        self.base_path = Path(base_path)
        self.config = self._load_study_config()
        
    def _load_study_config(self) -> dict:
        """Load study-specific configuration."""
        config_path = self.base_path / "configs" / f"{self.study_id}_config.json"
        
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration for clinical trial processing."""
        return {
            "data_sources": {
                "site_1": {"type": "csv", "path": "data/site_1/"},
                "site_2": {"type": "database", "connection": "clinical_db"},
                "central_lab": {"type": "api", "endpoint": "https://lab.example.com/api"}
            },
            "quality_thresholds": {
                "missing_data_threshold": 0.1,
                "outlier_threshold": 0.05,
                "duplicate_threshold": 0.01
            },
            "output_formats": ["csv", "parquet", "json"],
            "notification_emails": ["dm@clinicaltrial.org", "stats@clinicaltrial.org"]
        }

def create_clinical_trial_dag(study_id: str) -> DAG:
    """Create comprehensive clinical trial data processing DAG."""
    
    # DAG configuration
    default_args = {
        'owner': 'clinical-data-team',
        'depends_on_past': False,
        'start_date': days_ago(1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
        'max_active_runs': 1  # Prevent concurrent runs
    }
    
    dag = DAG(
        f'clinical_trial_{study_id}',
        default_args=default_args,
        description='Automated clinical trial data processing pipeline',
        schedule_interval='0 6 * * *',  # Daily at 6 AM
        catchup=False,
        tags=['clinical', 'data-processing', 'automated']
    )
    
    # Task 1: Data Validation and Ingestion
    with TaskGroup("data_ingestion", dag=dag) as data_ingestion_group:
        
        validate_sources = PythonOperator(
            task_id='validate_data_sources',
            python_callable=validate_data_sources,
            op_args=[study_id],
            dag=dag
        )
        
        ingest_site_data = PythonOperator(
            task_id='ingest_multi_site_data',
            python_callable=ingest_multi_site_data,
            op_args=[study_id],
            dag=dag
        )
        
        validate_sources >> ingest_site_data
    
    # Task 2: Data Quality Assessment
    with TaskGroup("quality_assessment", dag=dag) as quality_group:
        
        run_quality_checks = PythonOperator(
            task_id='comprehensive_quality_assessment',
            python_callable=run_comprehensive_quality_checks,
            op_args=[study_id],
            dag=dag
        )
        
        generate_quality_report = PythonOperator(
            task_id='generate_quality_dashboard',
            python_callable=generate_quality_dashboard,
            op_args=[study_id],
            dag=dag
        )
        
        run_quality_checks >> generate_quality_report
    
    # Task 3: Statistical Analysis
    with TaskGroup("statistical_analysis", dag=dag) as stats_group:
        
        primary_analysis = PythonOperator(
            task_id='primary_endpoint_analysis',
            python_callable=run_primary_analysis,
            op_args=[study_id],
            dag=dag
        )
        
        secondary_analysis = PythonOperator(
            task_id='secondary_endpoints_analysis',
            python_callable=run_secondary_analysis,
            op_args=[study_id],
            dag=dag
        )
        
        safety_analysis = PythonOperator(
            task_id='safety_analysis',
            python_callable=run_safety_analysis,
            op_args=[study_id],
            dag=dag
        )
        
        primary_analysis >> [secondary_analysis, safety_analysis]
    
    # Task 4: Report Generation
    generate_reports = PythonOperator(
        task_id='generate_automated_reports',
        python_callable=generate_automated_reports,
        op_args=[study_id],
        dag=dag
    )
    
    # Task 5: Notification and Archiving
    send_notifications = EmailOperator(
        task_id='send_completion_notification',
        to=['dm@clinicaltrial.org', 'stats@clinicaltrial.org'],
        subject=f'Clinical Trial {study_id} - Daily Processing Complete',
        html_content="""
        <h3>Clinical Trial Data Processing Complete</h3>
        <p>Study: {{ params.study_id }}</p>
        <p>Date: {{ ds }}</p>
        <p>Status: {{ ti.state }}</p>
        <p>Processing time: {{ ti.duration }} seconds</p>
        
        <p>Reports available at: /reports/{{ params.study_id }}/{{ ds }}/</p>
        """,
        params={'study_id': study_id},
        dag=dag
    )
    
    # Define task dependencies
    data_ingestion_group >> quality_group >> stats_group >> generate_reports >> send_notifications
    
    return dag

# Task implementations
def validate_data_sources(study_id: str, **context):
    """Validate that all required data sources are available."""
    
    pipeline = ClinicalTrialPipeline(study_id, "/opt/clinical_trials/")
    
    validation_results = {}
    
    for source_name, source_config in pipeline.config['data_sources'].items():
        try:
            if source_config['type'] == 'csv':
                source_path = Path(source_config['path'])
                available = source_path.exists() and any(source_path.glob('*.csv'))
                
            elif source_config['type'] == 'database':
                # Test database connection
                available = test_database_connection(source_config['connection'])
                
            elif source_config['type'] == 'api':
                # Test API endpoint
                available = test_api_endpoint(source_config['endpoint'])
            
            validation_results[source_name] = {
                'available': available,
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to validate {source_name}: {str(e)}")
            validation_results[source_name] = {
                'available': False,
                'error': str(e),
                'checked_at': datetime.now().isoformat()
            }
    
    # Check if all sources are available
    all_available = all(result['available'] for result in validation_results.values())
    
    if not all_available:
        failed_sources = [name for name, result in validation_results.items() 
                         if not result['available']]
        raise ValueError(f"Data source validation failed for: {failed_sources}")
    
    # Store validation results for downstream tasks
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    
    logger.info(f"All data sources validated successfully for study {study_id}")
    
    return validation_results

def ingest_multi_site_data(study_id: str, **context):
    """Ingest data from multiple sites with error handling."""
    
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    pipeline = ClinicalTrialPipeline(study_id, "/opt/clinical_trials/")
    
    ingestion_summary = {
        'total_records': 0,
        'successful_sources': 0,
        'failed_sources': 0,
        'processing_errors': []
    }
    
    all_data = []
    
    for source_name, source_config in pipeline.config['data_sources'].items():
        try:
            logger.info(f"Ingesting data from {source_name}")
            
            if source_config['type'] == 'csv':
                source_data = ingest_csv_data(source_config['path'])
            elif source_config['type'] == 'database':
                source_data = ingest_database_data(source_config['connection'])
            elif source_config['type'] == 'api':
                source_data = ingest_api_data(source_config['endpoint'])
            
            # Add source identifier
            source_data['data_source'] = source_name
            source_data['ingestion_timestamp'] = datetime.now().isoformat()
            
            all_data.append(source_data)
            ingestion_summary['total_records'] += len(source_data)
            ingestion_summary['successful_sources'] += 1
            
            logger.info(f"Successfully ingested {len(source_data)} records from {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to ingest data from {source_name}: {str(e)}")
            ingestion_summary['failed_sources'] += 1
            ingestion_summary['processing_errors'].append({
                'source': source_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    if ingestion_summary['successful_sources'] == 0:
        raise ValueError("Failed to ingest data from any source")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save to staging area
    staging_path = pipeline.base_path / "staging" / study_id / f"combined_data_{context['ds']}.parquet"
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_parquet(staging_path)
    
    # Store summary for downstream tasks
    context['task_instance'].xcom_push(key='ingestion_summary', value=ingestion_summary)
    context['task_instance'].xcom_push(key='staging_path', value=str(staging_path))
    
    logger.info(f"Data ingestion complete. Total records: {ingestion_summary['total_records']}")
    
    return ingestion_summary

def run_comprehensive_quality_checks(study_id: str, **context):
    """Run comprehensive data quality assessment."""
    
    staging_path = context['task_instance'].xcom_pull(key='staging_path')
    pipeline = ClinicalTrialPipeline(study_id, "/opt/clinical_trials/")
    
    # Load data
    df = pd.read_parquet(staging_path)
    
    # Import quality assessment tools from previous lectures
    from reports.utils.data_loader import DataQualityAssessor
    
    assessor = DataQualityAssessor(pipeline.config)
    quality_report = assessor.comprehensive_assessment(df, f"study_{study_id}")
    
    # Check against quality thresholds
    quality_alerts = []
    thresholds = pipeline.config['quality_thresholds']
    
    # Check missing data threshold
    high_missing_cols = [col for col, info in quality_report.missing_data.items() 
                        if info.get('percent', 0) > thresholds['missing_data_threshold'] * 100]
    
    if high_missing_cols:
        quality_alerts.append({
            'type': 'missing_data',
            'severity': 'warning',
            'message': f"High missing data in columns: {high_missing_cols}"
        })
    
    # Check outlier threshold
    high_outlier_cols = [col for col, info in quality_report.outliers.items() 
                        if info.get('percent', 0) > thresholds['outlier_threshold'] * 100]
    
    if high_outlier_cols:
        quality_alerts.append({
            'type': 'outliers',
            'severity': 'warning',
            'message': f"High outlier rate in columns: {high_outlier_cols}"
        })
    
    # Check duplicate threshold
    duplicate_rate = quality_report.duplicates / quality_report.n_rows
    if duplicate_rate > thresholds['duplicate_threshold']:
        quality_alerts.append({
            'type': 'duplicates',
            'severity': 'critical',
            'message': f"Duplicate rate ({duplicate_rate:.2%}) exceeds threshold"
        })
    
    # Store results
    quality_results = {
        'quality_report': quality_report,
        'quality_alerts': quality_alerts,
        'quality_score': quality_report.quality_score,
        'passed_thresholds': len(quality_alerts) == 0
    }
    
    context['task_instance'].xcom_push(key='quality_results', value=quality_results)
    
    # Fail task if critical quality issues
    critical_alerts = [alert for alert in quality_alerts if alert['severity'] == 'critical']
    if critical_alerts:
        raise ValueError(f"Critical quality issues detected: {critical_alerts}")
    
    logger.info(f"Quality assessment complete. Score: {quality_report.quality_score:.1f}")
    
    return quality_results
```

#### 2. Advanced Error Handling and Recovery

```python
from functools import wraps
import traceback
import json
from datetime import datetime
from typing import Callable, Any, Optional
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class ProductionErrorHandler:
    """Professional error handling and recovery for production systems."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 notification_config: Optional[dict] = None,
                 recovery_strategies: Optional[dict] = None):
        
        self.logger = self._setup_logger(log_level)
        self.notification_config = notification_config or {}
        self.recovery_strategies = recovery_strategies or {}
        self.error_history = []
        
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up comprehensive logging configuration."""
        
        logger = logging.getLogger('production_pipeline')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'logs/production_pipeline.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def handle_with_recovery(self, 
                           recovery_strategy: str = "retry",
                           max_retries: int = 3,
                           notify_on_failure: bool = True):
        """Decorator for functions with automated error recovery."""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        self.logger.info(f"Executing {func.__name__} (attempt {attempt + 1})")
                        result = func(*args, **kwargs)
                        
                        if attempt > 0:
                            self.logger.info(f"Recovery successful for {func.__name__} after {attempt} retries")
                        
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        error_details = self._capture_error_details(func, e, args, kwargs)
                        self.error_history.append(error_details)
                        
                        if attempt < max_retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying..."
                            )
                            
                            # Apply recovery strategy
                            self._apply_recovery_strategy(recovery_strategy, attempt, error_details)
                            
                        else:
                            self.logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}"
                            )
                            
                            if notify_on_failure:
                                self._send_failure_notification(func.__name__, error_details)
                
                # If we get here, all retries failed
                raise last_exception
            
            return wrapper
        return decorator
    
    def _capture_error_details(self, func: Callable, exception: Exception, 
                             args: tuple, kwargs: dict) -> dict:
        """Capture comprehensive error details for debugging."""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'function_name': func.__name__,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc(),
            'args': str(args),
            'kwargs': str(kwargs),
            'function_module': func.__module__,
            'function_file': func.__code__.co_filename,
            'function_line': func.__code__.co_firstlineno
        }
    
    def _apply_recovery_strategy(self, strategy: str, attempt: int, error_details: dict):
        """Apply recovery strategy based on error type."""
        
        if strategy == "retry":
            # Exponential backoff
            import time
            wait_time = (2 ** attempt) * 1  # 1, 2, 4, 8 seconds
            self.logger.info(f"Waiting {wait_time} seconds before retry")
            time.sleep(wait_time)
            
        elif strategy == "fallback_data":
            # Use cached or backup data
            self.logger.info("Attempting fallback to cached data")
            # Implementation would depend on specific use case
            
        elif strategy == "skip_and_continue":
            # Skip problematic records and continue
            self.logger.info("Skipping problematic data and continuing")
            # Implementation would mark data as skipped
            
        elif strategy == "manual_intervention":
            # Alert for manual intervention
            self.logger.warning("Manual intervention required")
            self._send_intervention_request(error_details)
    
    def _send_failure_notification(self, function_name: str, error_details: dict):
        """Send notification about critical failures."""
        
        if not self.notification_config.get('email_enabled', False):
            return
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.notification_config['sender_email']
            msg['To'] = ', '.join(self.notification_config['recipient_emails'])
            msg['Subject'] = f"Production Pipeline Failure: {function_name}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2>Production Pipeline Failure Alert</h2>
                <h3>Function: {function_name}</h3>
                <p><strong>Time:</strong> {error_details['timestamp']}</p>
                <p><strong>Error Type:</strong> {error_details['exception_type']}</p>
                <p><strong>Error Message:</strong> {error_details['exception_message']}</p>
                
                <h4>Technical Details:</h4>
                <pre>{error_details['traceback']}</pre>
                
                <h4>Error History (Last 5):</h4>
                <ul>
                {''.join(f"<li>{error['timestamp']}: {error['function_name']} - {error['exception_type']}</li>" 
                        for error in self.error_history[-5:])}
                </ul>
                
                <p><strong>Action Required:</strong> Please investigate and resolve this issue.</p>
            </body>
            </html>
            """
            
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.notification_config['smtp_server'], 
                             self.notification_config['smtp_port']) as server:
                if self.notification_config.get('use_tls', True):
                    server.starttls()
                if 'smtp_username' in self.notification_config:
                    server.login(self.notification_config['smtp_username'], 
                               self.notification_config['smtp_password'])
                server.send_message(msg)
            
            self.logger.info(f"Failure notification sent for {function_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
    
    def generate_error_report(self) -> dict:
        """Generate comprehensive error analysis report."""
        
        if not self.error_history:
            return {'message': 'No errors recorded', 'error_count': 0}
        
        # Error frequency analysis
        error_types = {}
        function_errors = {}
        
        for error in self.error_history:
            error_type = error['exception_type']
            function_name = error['function_name']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            function_errors[function_name] = function_errors.get(function_name, 0) + 1
        
        # Recent error trend
        recent_errors = [error for error in self.error_history 
                        if (datetime.now() - datetime.fromisoformat(error['timestamp'])).days <= 7]
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors_7_days': len(recent_errors),
            'most_common_error_types': sorted(error_types.items(), key=lambda x: x[1], reverse=True),
            'functions_with_most_errors': sorted(function_errors.items(), key=lambda x: x[1], reverse=True),
            'last_error': self.error_history[-1] if self.error_history else None,
            'error_rate_trend': self._calculate_error_rate_trend(),
            'recommendations': self._generate_error_recommendations()
        }
    
    def _calculate_error_rate_trend(self) -> dict:
        """Calculate error rate trend over time."""
        
        daily_errors = {}
        
        for error in self.error_history:
            date = datetime.fromisoformat(error['timestamp']).date().isoformat()
            daily_errors[date] = daily_errors.get(date, 0) + 1
        
        # Simple trend calculation (last 7 days vs previous 7 days)
        recent_dates = sorted(daily_errors.keys())[-7:]
        previous_dates = sorted(daily_errors.keys())[-14:-7]
        
        recent_avg = sum(daily_errors.get(date, 0) for date in recent_dates) / max(len(recent_dates), 1)
        previous_avg = sum(daily_errors.get(date, 0) for date in previous_dates) / max(len(previous_dates), 1)
        
        trend = "increasing" if recent_avg > previous_avg else "decreasing"
        
        return {
            'recent_avg_errors_per_day': recent_avg,
            'previous_avg_errors_per_day': previous_avg,
            'trend': trend,
            'daily_errors': daily_errors
        }
    
    def _generate_error_recommendations(self) -> list:
        """Generate recommendations based on error patterns."""
        
        recommendations = []
        
        # Analyze error patterns
        error_types = {}
        for error in self.error_history:
            error_type = error['exception_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Common recommendations based on error types
        if 'ConnectionError' in error_types:
            recommendations.append({
                'priority': 'high',
                'issue': 'Connection Errors',
                'recommendation': 'Implement connection pooling and retry logic with exponential backoff'
            })
        
        if 'FileNotFoundError' in error_types:
            recommendations.append({
                'priority': 'medium',
                'issue': 'File Access Errors',
                'recommendation': 'Add file existence validation before processing'
            })
        
        if 'KeyError' in error_types:
            recommendations.append({
                'priority': 'medium',
                'issue': 'Data Schema Errors',
                'recommendation': 'Implement schema validation for incoming data'
            })
        
        # General recommendations
        if len(self.error_history) > 50:
            recommendations.append({
                'priority': 'high',
                'issue': 'High Error Volume',
                'recommendation': 'Consider implementing circuit breaker pattern and improved input validation'
            })
        
        return recommendations

# Usage example
error_handler = ProductionErrorHandler(
    log_level="INFO",
    notification_config={
        'email_enabled': True,
        'sender_email': 'system@clinicaltrial.org',
        'recipient_emails': ['admin@clinicaltrial.org', 'dev@clinicaltrial.org'],
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'use_tls': True
    }
)

@error_handler.handle_with_recovery(recovery_strategy="retry", max_retries=3)
def process_clinical_data(file_path: str) -> pd.DataFrame:
    """Example function with error handling."""
    
    # Simulate potential failures
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("Empty dataset received")
    
    return df
```

### Hands-On Exercise 1: Production Workflow Design

**Scenario**: Design an automated workflow for a longitudinal diabetes management study.

**Requirements**:
- Daily ingestion from 5 clinic sites
- Real-time quality monitoring with alerts
- Weekly statistical reports for clinicians
- Monthly regulatory compliance reports

**Your Task**:
- Create Airflow DAG with proper dependencies
- Implement comprehensive error handling
- Design notification system for stakeholders
- Add monitoring and recovery mechanisms

---

## Part 2: Professional Code Quality Standards (75 minutes)

### Professional Challenge: Collaborative Data Science Development

Professional data science teams require:
- **Code standards**: Consistent style and documentation across team members
- **Testing frameworks**: Automated validation of analytical code
- **Version control workflows**: Managing changes and collaboration
- **Code review processes**: Ensuring quality and knowledge sharing

### Advanced Code Quality Framework

#### 1. Comprehensive Testing for Analytical Code

```python
import unittest
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TestClinicalDataProcessor(unittest.TestCase):
    """Comprehensive test suite for clinical data processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Create sample clinical data
        self.sample_data = pd.DataFrame({
            'patient_id': ['PAT001', 'PAT002', 'PAT003', 'PAT004', 'PAT005'],
            'age': [45, 62, 38, 71, 55],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'baseline_hba1c': [7.2, 8.9, 6.5, 9.8, 7.7],
            'followup_hba1c': [6.8, 7.2, 6.1, 8.1, 6.9],
            'treatment_group': ['Drug_A', 'Drug_B', 'Placebo', 'Drug_A', 'Drug_B'],
            'site_id': ['Site_01', 'Site_02', 'Site_01', 'Site_03', 'Site_02']
        })
        
        # Create sample data with missing values
        self.data_with_missing = self.sample_data.copy()
        self.data_with_missing.loc[1, 'followup_hba1c'] = np.nan
        self.data_with_missing.loc[3, 'baseline_hba1c'] = np.nan
        
        # Create sample data with outliers
        self.data_with_outliers = self.sample_data.copy()
        self.data_with_outliers.loc[0, 'baseline_hba1c'] = 25.0  # Extreme outlier
        
        # Import the functions to test
        from reports.utils.stats_utils import ClinicalStatisticalAnalyzer
        from reports.utils.data_loader import DataQualityAssessor
        
        self.stats_analyzer = ClinicalStatisticalAnalyzer()
        self.quality_assessor = DataQualityAssessor({})
    
    def test_primary_endpoint_analysis_two_groups(self):
        """Test primary endpoint analysis with two treatment groups."""
        
        # Filter to only two groups for simpler testing
        two_group_data = self.sample_data[
            self.sample_data['treatment_group'].isin(['Drug_A', 'Placebo'])
        ].copy()
        
        # Calculate change from baseline
        two_group_data['hba1c_change'] = (
            two_group_data['followup_hba1c'] - two_group_data['baseline_hba1c']
        )
        
        result = self.stats_analyzer._primary_endpoint_analysis(
            two_group_data, 'hba1c_change', 'treatment_group'
        )
        
        # Assertions
        self.assertEqual(result['test_type'], 'two_sample_t_test')
        self.assertIn('p_value', result)
        self.assertIn('cohens_d', result)
        self.assertIn('mean_difference', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIsInstance(result['p_value'], float)
        self.assertTrue(0 <= result['p_value'] <= 1)
    
    def test_primary_endpoint_analysis_multiple_groups(self):
        """Test primary endpoint analysis with multiple treatment groups."""
        
        # Calculate change from baseline
        test_data = self.sample_data.copy()
        test_data['hba1c_change'] = (
            test_data['followup_hba1c'] - test_data['baseline_hba1c']
        )
        
        result = self.stats_analyzer._primary_endpoint_analysis(
            test_data, 'hba1c_change', 'treatment_group'
        )
        
        # Assertions for ANOVA
        self.assertEqual(result['test_type'], 'one_way_anova')
        self.assertIn('f_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('groups', result)
        self.assertIsInstance(result['groups'], dict)
        
        # Check that all treatment groups are represented
        expected_groups = set(test_data['treatment_group'].unique())
        actual_groups = set(result['groups'].keys())
        self.assertEqual(expected_groups, actual_groups)
    
    def test_data_quality_assessment_basic(self):
        """Test basic data quality assessment functionality."""
        
        quality_report = self.quality_assessor.comprehensive_assessment(
            self.sample_data, "test_dataset"
        )
        
        # Basic structure assertions
        self.assertEqual(quality_report.dataset_name, "test_dataset")
        self.assertEqual(quality_report.n_rows, len(self.sample_data))
        self.assertEqual(quality_report.n_cols, len(self.sample_data.columns))
        self.assertIsInstance(quality_report.quality_score, float)
        self.assertTrue(0 <= quality_report.quality_score <= 100)
    
    def test_data_quality_assessment_with_missing_data(self):
        """Test data quality assessment with missing data."""
        
        quality_report = self.quality_assessor.comprehensive_assessment(
            self.data_with_missing, "test_dataset_missing"
        )
        
        # Should detect missing data
        self.assertTrue(len(quality_report.missing_data) > 0)
        
        # Check specific missing data detection
        missing_cols = list(quality_report.missing_data.keys())
        self.assertIn('followup_hba1c', missing_cols)
        self.assertIn('baseline_hba1c', missing_cols)
    
    def test_data_quality_assessment_with_outliers(self):
        """Test outlier detection in data quality assessment."""
        
        quality_report = self.quality_assessor.comprehensive_assessment(
            self.data_with_outliers, "test_dataset_outliers"
        )
        
        # Should detect outliers in baseline_hba1c
        self.assertTrue(len(quality_report.outliers) > 0)
        self.assertIn('baseline_hba1c', quality_report.outliers)
    
    def test_statistical_analysis_edge_cases(self):
        """Test statistical analysis with edge cases."""
        
        # Test with all same values (no variance)
        no_variance_data = self.sample_data.copy()
        no_variance_data['baseline_hba1c'] = 7.0  # All same value
        no_variance_data['hba1c_change'] = 0.0    # All same change
        
        # This should handle the zero variance case gracefully
        with self.assertLogs(level='WARNING') as log:
            try:
                result = self.stats_analyzer._primary_endpoint_analysis(
                    no_variance_data, 'hba1c_change', 'treatment_group'
                )
                # If it doesn't raise an exception, that's also acceptable
            except (ValueError, ZeroDivisionError):
                # Expected behavior for zero variance
                pass
    
    def test_data_processing_with_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        
        empty_df = pd.DataFrame()
        
        with self.assertRaises((ValueError, IndexError)):
            self.quality_assessor.comprehensive_assessment(empty_df, "empty_dataset")
    
    def test_data_processing_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        
        # Create larger dataset
        large_data = pd.concat([self.sample_data] * 1000, ignore_index=True)
        
        # Add some noise to make it realistic
        np.random.seed(42)
        large_data['baseline_hba1c'] += np.random.normal(0, 0.5, len(large_data))
        large_data['followup_hba1c'] += np.random.normal(0, 0.3, len(large_data))
        
        # Should process without memory errors
        try:
            quality_report = self.quality_assessor.comprehensive_assessment(
                large_data, "large_dataset"
            )
            self.assertIsInstance(quality_report.quality_score, float)
        except MemoryError:
            self.fail("Memory error with moderately large dataset")

class TestClinicalFeatureEngineering(unittest.TestCase):
    """Test suite for clinical feature engineering functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        
        # Import feature engineering classes
        from reports.utils.data_loader import ClinicalFeatureEngineer
        
        self.feature_engineer = ClinicalFeatureEngineer()
        
        # Sample clinical data with features needed for engineering
        self.clinical_data = pd.DataFrame({
            'patient_id': ['PAT001', 'PAT002', 'PAT003'],
            'age': [45, 62, 35],
            'gender': ['Male', 'Female', 'Male'],
            'total_cholesterol': [220, 280, 180],
            'hdl_cholesterol': [45, 55, 65],
            'ldl_cholesterol': [140, 180, 95],
            'triglycerides': [150, 200, 120],
            'systolic_bp': [140, 160, 120],
            'smoking': [1, 0, 0],
            'on_statin': [True, True, False],
            'on_ace_inhibitor': [True, False, False]
        })
    
    def test_create_clinical_features(self):
        """Test clinical feature creation."""
        
        featured_data = self.feature_engineer.create_clinical_features(self.clinical_data)
        
        # Check that new features are created
        expected_features = [
            'tc_hdl_ratio',
            'ldl_hdl_ratio', 
            'framingham_risk_score',
            'polypharmacy_count',
            'statin_ace_combo'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, featured_data.columns, f"Missing feature: {feature}")
        
        # Check feature calculations
        # TC/HDL ratio
        expected_tc_hdl = self.clinical_data['total_cholesterol'] / self.clinical_data['hdl_cholesterol']
        pd.testing.assert_series_equal(
            featured_data['tc_hdl_ratio'], 
            expected_tc_hdl, 
            check_names=False
        )
        
        # Drug combination feature
        expected_combo = (self.clinical_data['on_statin'] & 
                         self.clinical_data['on_ace_inhibitor']).astype(int)
        pd.testing.assert_series_equal(
            featured_data['statin_ace_combo'],
            expected_combo,
            check_names=False
        )
    
    def test_framingham_risk_calculation(self):
        """Test Framingham risk score calculation."""
        
        risk_scores = self.feature_engineer._calculate_framingham_risk(self.clinical_data)
        
        # Should return numeric values
        self.assertTrue(pd.api.types.is_numeric_dtype(risk_scores))
        
        # Should not have negative values
        self.assertTrue((risk_scores >= 0).all())
        
        # Older patient with risk factors should have higher score
        older_patient_score = risk_scores[self.clinical_data['age'] == 62].iloc[0]
        younger_patient_score = risk_scores[self.clinical_data['age'] == 35].iloc[0]
        
        self.assertGreaterEqual(older_patient_score, younger_patient_score)

# Property-based testing using Hypothesis (advanced testing technique)
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, column

class TestClinicalDataProperties(unittest.TestCase):
    """Property-based tests for clinical data functions."""
    
    @given(data_frames([
        column('age', elements=st.integers(min_value=18, max_value=100)),
        column('baseline_value', elements=st.floats(min_value=4.0, max_value=15.0, allow_nan=False)),
        column('followup_value', elements=st.floats(min_value=4.0, max_value=15.0, allow_nan=False))
    ], min_size=5, max_size=50))
    def test_change_calculation_properties(self, df):
        """Test properties of change calculations."""
        
        # Calculate change
        df['change'] = df['followup_value'] - df['baseline_value']
        
        # Properties that should always hold
        # 1. Change should be additive inverse
        df['negative_change'] = df['baseline_value'] - df['followup_value']
        self.assertTrue(np.allclose(df['change'], -df['negative_change'], rtol=1e-10))
        
        # 2. If followup equals baseline, change should be zero
        equal_values_mask = df['followup_value'] == df['baseline_value']
        if equal_values_mask.any():
            changes_for_equal = df.loc[equal_values_mask, 'change']
            self.assertTrue(np.allclose(changes_for_equal, 0.0, atol=1e-10))

# Integration tests
class TestClinicalPipelineIntegration(unittest.TestCase):
    """Integration tests for complete clinical data pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data files
        test_data = pd.DataFrame({
            'patient_id': [f'PAT{i:03d}' for i in range(1, 101)],
            'age': np.random.normal(55, 15, 100).astype(int),
            'baseline_hba1c': np.random.normal(8.0, 1.5, 100),
            'followup_hba1c': np.random.normal(7.2, 1.2, 100),
            'treatment_group': np.random.choice(['Drug_A', 'Drug_B', 'Placebo'], 100)
        })
        
        self.test_file = os.path.join(self.test_dir, 'test_clinical_data.csv')
        test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_complete_analysis_pipeline(self):
        """Test complete analysis pipeline from data loading to results."""
        
        # This would test the entire pipeline
        from reports.utils.data_loader import DataLoader
        from reports.utils.stats_utils import ClinicalStatisticalAnalyzer
        
        # Load data
        loader = DataLoader({})
        df = loader.load_with_schema('clinical_data', self.test_file)
        
        # Calculate derived variables
        df['hba1c_change'] = df['followup_hba1c'] - df['baseline_hba1c']
        
        # Run statistical analysis
        analyzer = ClinicalStatisticalAnalyzer()
        results = analyzer.perform_comprehensive_analysis(
            df, 'hba1c_change', 'treatment_group', ['age']
        )
        
        # Verify results structure
        self.assertIn('primary_analysis', results)
        self.assertIn('descriptives', results)
        
        # Verify statistical test was performed
        primary_results = results['primary_analysis']
        self.assertIn('test_type', primary_results)
        self.assertIn('p_value', primary_results)

# Performance tests
class TestPerformance(unittest.TestCase):
    """Performance benchmarking tests."""
    
    def test_large_dataset_processing_time(self):
        """Test processing time for large datasets."""
        import time
        
        # Create large dataset
        large_data = pd.DataFrame({
            'patient_id': [f'PAT{i:06d}' for i in range(10000)],
            'value': np.random.normal(100, 15, 10000)
        })
        
        # Time the quality assessment
        from reports.utils.data_loader import DataQualityAssessor
        
        assessor = DataQualityAssessor({})
        
        start_time = time.time()
        quality_report = assessor.comprehensive_assessment(large_data, "performance_test")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 10k records in reasonable time (< 30 seconds)
        self.assertLess(processing_time, 30.0, 
                       f"Processing took too long: {processing_time:.2f} seconds")

# Test runner configuration
if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestClinicalDataProcessor))
    suite.addTest(unittest.makeSuite(TestClinicalFeatureEngineering))
    suite.addTest(unittest.makeSuite(TestClinicalDataProperties))
    suite.addTest(unittest.makeSuite(TestClinicalPipelineIntegration))
    suite.addTest(unittest.makeSuite(TestPerformance))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
```

#### 2. Professional Documentation Standards

```python
"""
Clinical Data Analysis Pipeline
===============================

This module provides comprehensive tools for analyzing clinical trial data,
including statistical analysis, quality assessment, and automated reporting.

Author: Clinical Data Science Team
Version: 2.1.0
Last Updated: 2024-01-15

Dependencies:
    - pandas >= 1.5.0
    - numpy >= 1.21.0
    - scipy >= 1.9.0
    - statsmodels >= 0.13.0
    - scikit-learn >= 1.1.0

Usage:
    Basic usage for clinical trial analysis:
    
    >>> from clinical_analysis import ClinicalAnalyzer
    >>> analyzer = ClinicalAnalyzer(study_id='CARDIO_2024')
    >>> results = analyzer.run_primary_analysis(data, endpoint='ldl_change')
    
License:
    This software is proprietary and confidential. Unauthorized copying,
    distribution, or use is strictly prohibited.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

# Configure module logger
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Enumeration of supported clinical analysis types.
    
    Attributes:
        PRIMARY: Primary endpoint analysis (efficacy)
        SECONDARY: Secondary endpoint analysis
        SAFETY: Safety analysis and adverse event assessment
        SUBGROUP: Subgroup and subset analyses
        SENSITIVITY: Sensitivity and robustness analyses
    """
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SAFETY = "safety"
    SUBGROUP = "subgroup"
    SENSITIVITY = "sensitivity"

@dataclass
class StudyConfiguration:
    """Configuration parameters for clinical study analysis.
    
    This dataclass encapsulates all configuration parameters needed
    for a comprehensive clinical study analysis pipeline.
    
    Attributes:
        study_id (str): Unique identifier for the clinical study
        protocol_version (str): Version of the study protocol
        primary_endpoint (str): Column name of primary endpoint variable
        treatment_column (str): Column name indicating treatment assignment
        significance_level (float): Statistical significance threshold (default: 0.05)
        power_threshold (float): Minimum acceptable statistical power (default: 0.80)
        missing_data_threshold (float): Maximum acceptable proportion of missing data
        quality_score_minimum (float): Minimum acceptable data quality score
        
    Example:
        >>> config = StudyConfiguration(
        ...     study_id="CARDIO_2024",
        ...     protocol_version="3.0",
        ...     primary_endpoint="ldl_change",
        ...     treatment_column="treatment_group"
        ... )
    """
    study_id: str
    protocol_version: str
    primary_endpoint: str
    treatment_column: str
    significance_level: float = 0.05
    power_threshold: float = 0.80
    missing_data_threshold: float = 0.20
    quality_score_minimum: float = 75.0
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if not 0 < self.significance_level < 1:
            raise ValueError("Significance level must be between 0 and 1")
        if not 0 < self.power_threshold < 1:
            raise ValueError("Power threshold must be between 0 and 1")
        if not 0 < self.missing_data_threshold < 1:
            raise ValueError("Missing data threshold must be between 0 and 1")

class ClinicalAnalyzer:
    """Comprehensive clinical trial data analyzer.
    
    This class provides a complete framework for analyzing clinical trial data,
    including primary endpoint analysis, safety assessment, and regulatory
    reporting. It follows ICH E9 guidelines for statistical principles.
    
    The analyzer supports:
        - Multiple treatment group comparisons
        - Covariate adjustment using ANCOVA
        - Missing data handling strategies
        - Multiple testing corrections
        - Effect size calculations with confidence intervals
        - Regulatory-compliant reporting
    
    Attributes:
        config (StudyConfiguration): Study-specific configuration parameters
        results (Dict): Dictionary storing analysis results
        quality_metrics (Dict): Data quality assessment results
        
    Example:
        >>> config = StudyConfiguration(
        ...     study_id="DIABETES_2024",
        ...     primary_endpoint="hba1c_change",
        ...     treatment_column="treatment_group"
        ... )
        >>> analyzer = ClinicalAnalyzer(config)
        >>> results = analyzer.run_comprehensive_analysis(clinical_data)
    """
    
    def __init__(self, config: StudyConfiguration):
        """Initialize clinical analyzer with study configuration.
        
        Args:
            config (StudyConfiguration): Complete study configuration including
                endpoints, treatment assignments, and analysis parameters
                
        Raises:
            ValueError: If configuration parameters are invalid
            TypeError: If config is not a StudyConfiguration instance
        """
        if not isinstance(config, StudyConfiguration):
            raise TypeError("config must be a StudyConfiguration instance")
            
        self.config = config
        self.results: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, Any] = {}
        
        # Set up analysis-specific logging
        self.logger = logging.getLogger(f"{__name__}.{config.study_id}")
        self.logger.setLevel(logging.INFO)
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info(f"Initialized ClinicalAnalyzer for study {config.study_id}")
    
    def run_comprehensive_analysis(self, 
                                 data: pd.DataFrame,
                                 covariates: Optional[List[str]] = None,
                                 stratification_factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute comprehensive clinical trial analysis pipeline.
        
        This method orchestrates a complete clinical analysis including:
        1. Data quality assessment and validation
        2. Descriptive statistics by treatment group  
        3. Primary endpoint analysis with effect sizes
        4. Secondary endpoint analyses with multiplicity adjustment
        5. Safety analysis and adverse event assessment
        6. Subgroup analyses for key demographics
        7. Sensitivity analyses for robustness
        
        Args:
            data (pd.DataFrame): Clinical trial dataset containing all required variables.
                Must include columns specified in configuration (primary_endpoint,
                treatment_column) and any requested covariates.
            covariates (Optional[List[str]]): List of baseline covariates for adjustment.
                Common covariates include age, gender, baseline_value, site_id.
                If None, unadjusted analysis will be performed.
            stratification_factors (Optional[List[str]]): Variables used for 
                randomization stratification. These will be included in adjusted analyses.
                
        Returns:
            Dict[str, Any]: Comprehensive analysis results containing:
                - 'quality_assessment': Data quality metrics and validation results
                - 'descriptive_statistics': Summary statistics by treatment group
                - 'primary_analysis': Primary endpoint statistical results
                - 'secondary_analyses': Secondary endpoint results with multiplicity correction  
                - 'safety_analysis': Adverse event analysis and safety profile
                - 'subgroup_analyses': Treatment effect consistency across subgroups
                - 'sensitivity_analyses': Robustness testing results
                - 'regulatory_summary': Summary formatted for regulatory submission
                
        Raises:
            ValueError: If required columns are missing from data
            RuntimeError: If analysis fails due to insufficient data or other issues
            
        Example:
            >>> # Load clinical trial data
            >>> data = pd.read_csv('clinical_trial_data.csv')
            >>> 
            >>> # Define covariates for adjustment
            >>> covariates = ['age', 'gender', 'baseline_ldl', 'site_id']
            >>> 
            >>> # Run comprehensive analysis
            >>> results = analyzer.run_comprehensive_analysis(
            ...     data=data,
            ...     covariates=covariates,
            ...     stratification_factors=['site_id']
            ... )
            >>> 
            >>> # Access primary analysis results
            >>> primary_p_value = results['primary_analysis']['p_value']
            >>> treatment_effect = results['primary_analysis']['treatment_effect']
            
        Note:
            This method follows ICH E9 statistical principles and produces
            results suitable for regulatory submission. All analyses include
            appropriate confidence intervals and effect size measures.
            
            For large datasets (>10,000 patients), consider using chunked
            processing or the run_analysis_chunked() method for better
            memory efficiency.
        """
        
        self.logger.info("Starting comprehensive clinical analysis")
        
        try:
            # Step 1: Validate input data
            self._validate_input_data(data, covariates)
            
            # Step 2: Data quality assessment
            self.logger.info("Performing data quality assessment")
            quality_results = self._assess_data_quality(data)
            self.results['quality_assessment'] = quality_results
            
            # Check if data quality meets minimum standards
            if quality_results['overall_score'] < self.config.quality_score_minimum:
                self.logger.warning(
                    f"Data quality score ({quality_results['overall_score']:.1f}) "
                    f"below minimum threshold ({self.config.quality_score_minimum})"
                )
            
            # Step 3: Descriptive statistics
            self.logger.info("Generating descriptive statistics")
            descriptive_results = self._generate_descriptive_statistics(
                data, covariates or []
            )
            self.results['descriptive_statistics'] = descriptive_results
            
            # Step 4: Primary endpoint analysis
            self.logger.info("Performing primary endpoint analysis")
            primary_results = self._analyze_primary_endpoint(
                data, covariates, stratification_factors
            )
            self.results['primary_analysis'] = primary_results
            
            # Step 5: Secondary analyses (placeholder for implementation)
            self.logger.info("Performing secondary analyses")
            secondary_results = self._analyze_secondary_endpoints(
                data, covariates, stratification_factors
            )
            self.results['secondary_analyses'] = secondary_results
            
            # Step 6: Safety analysis
            if self._has_safety_data(data):
                self.logger.info("Performing safety analysis")
                safety_results = self._analyze_safety(data)
                self.results['safety_analysis'] = safety_results
            
            # Step 7: Subgroup analyses
            if stratification_factors:
                self.logger.info("Performing subgroup analyses")
                subgroup_results = self._analyze_subgroups(
                    data, stratification_factors, covariates
                )
                self.results['subgroup_analyses'] = subgroup_results
            
            # Step 8: Generate regulatory summary
            self.logger.info("Generating regulatory summary")
            regulatory_summary = self._generate_regulatory_summary()
            self.results['regulatory_summary'] = regulatory_summary
            
            self.logger.info("Comprehensive analysis completed successfully")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise RuntimeError(f"Clinical analysis failed: {str(e)}") from e
    
    def _validate_input_data(self, 
                           data: pd.DataFrame, 
                           covariates: Optional[List[str]]) -> None:
        """Validate input data structure and required columns.
        
        Args:
            data (pd.DataFrame): Input clinical data
            covariates (Optional[List[str]]): List of covariate columns
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        
        # Check if DataFrame is empty
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check required columns
        required_columns = [self.config.primary_endpoint, self.config.treatment_column]
        
        if covariates:
            required_columns.extend(covariates)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate treatment column
        treatment_groups = data[self.config.treatment_column].unique()
        
        if len(treatment_groups) < 2:
            raise ValueError(
                f"Treatment column must have at least 2 groups, found: {treatment_groups}"
            )
        
        # Check for sufficient data per group
        min_group_size = 10  # Minimum for statistical testing
        group_sizes = data.groupby(self.config.treatment_column).size()
        small_groups = group_sizes[group_sizes < min_group_size].index.tolist()
        
        if small_groups:
            self.logger.warning(
                f"Treatment groups with small sample sizes: {small_groups}. "
                f"Statistical power may be limited."
            )
        
        self.logger.info("Input data validation completed successfully")
    
    # Additional method implementations would continue here...
    # Each method would include comprehensive docstrings following this pattern
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess comprehensive data quality metrics.
        
        Detailed implementation would go here...
        """
        # Implementation details...
        pass
    
    def _generate_descriptive_statistics(self, 
                                       data: pd.DataFrame, 
                                       covariates: List[str]) -> Dict[str, Any]:
        """Generate comprehensive descriptive statistics by treatment group.
        
        Detailed implementation would go here...
        """
        # Implementation details...
        pass
    
    # Continue with other method implementations...


def create_analysis_report(results: Dict[str, Any], 
                         output_format: str = "html") -> str:
    """Generate formatted analysis report from results.
    
    Creates a comprehensive, publication-ready report from clinical analysis
    results. Supports multiple output formats including HTML, PDF, and Word.
    
    Args:
        results (Dict[str, Any]): Complete analysis results from ClinicalAnalyzer
        output_format (str): Output format ('html', 'pdf', 'docx')
        
    Returns:
        str: Path to generated report file
        
    Example:
        >>> results = analyzer.run_comprehensive_analysis(data)
        >>> report_path = create_analysis_report(results, output_format="pdf")
        >>> print(f"Report saved to: {report_path}")
    """
    # Implementation would generate formatted report...
    pass


# Module-level configuration and utilities
__version__ = "2.1.0"
__author__ = "Clinical Data Science Team"

# Export public interface
__all__ = [
    'ClinicalAnalyzer',
    'StudyConfiguration', 
    'AnalysisType',
    'create_analysis_report'
]
```

### Hands-On Exercise 2: Professional Code Quality Implementation

**Scenario**: Establish code quality standards for a clinical research analytics team.

**Requirements**:
- Comprehensive test suite covering edge cases
- Professional documentation with examples
- Code style guidelines and automated formatting
- Type hints and static analysis integration

**Your Task**:
- Create test suite for existing analytical functions
- Write comprehensive documentation following standards
- Set up automated code quality checks
- Design code review process and guidelines

---

## Part 3: Containerization and Environment Management (60 minutes)

### Professional Challenge: Reproducible Analytics Environments

Professional analytics requires:
- **Environment consistency**: Same results across development, testing, production
- **Dependency management**: Reliable package versions and compatibility
- **Deployment portability**: Easy movement between systems and cloud providers
- **Version control**: Track environment changes over time

### Advanced Environment Management

#### 1. Professional Docker Configuration

```dockerfile
# Multi-stage Docker build for clinical analytics
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 clinicaluser && \
    useradd --uid 1000 --gid clinicaluser --shell /bin/bash --create-home clinicaluser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy application code
COPY --chown=clinicaluser:clinicaluser . .

# Switch to non-root user
USER clinicaluser

# Default command for development
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage  
FROM base as production

# Install only production dependencies
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge

# Copy application code
COPY --chown=clinicaluser:clinicaluser src/ ./src/
COPY --chown=clinicaluser:clinicaluser configs/ ./configs/
COPY --chown=clinicaluser:clinicaluser scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/results /app/logs && \
    chown -R clinicaluser:clinicaluser /app

# Switch to non-root user
USER clinicaluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import pandas, numpy, scipy; print('Health check passed')" || exit 1

# Default command for production
CMD ["python", "src/orchestrator.py"]

# Testing stage
FROM development as testing

# Copy test files
COPY --chown=clinicaluser:clinicaluser tests/ ./tests/

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=html"]
```

```yaml
# docker-compose.yml for complete development environment
version: '3.8'

services:
  # Main application container
  clinical-analytics:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    container_name: clinical-analytics-dev
    volumes:
      - .:/app
      - clinical-data:/app/data
      - clinical-results:/app/results
    ports:
      - "8888:8888"  # Jupyter Lab
      - "8000:8000"  # FastAPI/Dashboard
    environment:
      - POSTGRES_URL=postgresql://clinical:password@postgres:5432/clinical_db
      - REDIS_URL=redis://redis:6379
      - ENVIRONMENT=development
    depends_on:
      - postgres
      - redis
    networks:
      - clinical-network

  # PostgreSQL database for clinical data
  postgres:
    image: postgres:14-alpine
    container_name: clinical-postgres
    environment:
      POSTGRES_DB: clinical_db
      POSTGRES_USER: clinical
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - clinical-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U clinical"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Redis for caching and task queues
  redis:
    image: redis:7-alpine
    container_name: clinical-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - clinical-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Apache Airflow for workflow orchestration
  airflow:
    image: apache/airflow:2.7.0-python3.10
    container_name: clinical-airflow
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://clinical:password@postgres/airflow_db
      AIRFLOW__CORE__FERNET_KEY: 'your-fernet-key-here'
      AIRFLOW__WEBSERVER__SECRET_KEY: 'your-secret-key-here'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
      - airflow-logs:/opt/airflow/logs
    ports:
      - "8080:8080"
    depends_on:
      - postgres
    networks:
      - clinical-network
    command: >
      bash -c "airflow db init &&
               airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email admin@example.com &&
               airflow webserver"

  # MLflow for experiment tracking
  mlflow:
    image: python:3.10-slim
    container_name: clinical-mlflow
    working_dir: /app
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://clinical:password@postgres:5432/mlflow_db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
    volumes:
      - mlflow-artifacts:/app/artifacts
    ports:
      - "5000:5000"
    depends_on:
      - postgres
    networks:
      - clinical-network
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --backend-store-uri postgresql://clinical:password@postgres:5432/mlflow_db --default-artifact-root /app/artifacts --host 0.0.0.0"

  # Testing environment
  clinical-test:
    build:
      context: .
      dockerfile: Dockerfile
      target: testing
    container_name: clinical-test
    volumes:
      - .:/app
      - test-results:/app/test-results
    environment:
      - ENVIRONMENT=testing
    networks:
      - clinical-network
    profiles:
      - testing

volumes:
  clinical-data:
  clinical-results:
  postgres-data:
  redis-data:
  airflow-logs:
  mlflow-artifacts:
  test-results:

networks:
  clinical-network:
    driver: bridge
```

#### 2. Advanced Environment Configuration

```python
# environment_manager.py - Professional environment management
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import logging

class EnvironmentType(Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class EnvironmentConfig:
    """Environment configuration specification."""
    name: str
    type: EnvironmentType
    python_version: str
    dependencies: Dict[str, str]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, Any]
    security_settings: Dict[str, Any]

class ProfessionalEnvironmentManager:
    """Manage professional analytics environments."""
    
    def __init__(self, config_path: str = "configs/environments.yaml"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        self.environments = self._load_environment_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for environment management."""
        logger = logging.getLogger("environment_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_environment_configs(self) -> Dict[str, EnvironmentConfig]:
        """Load environment configurations from YAML."""
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        environments = {}
        for env_name, env_config in config_data.get('environments', {}).items():
            environments[env_name] = EnvironmentConfig(
                name=env_name,
                type=EnvironmentType(env_config['type']),
                python_version=env_config['python_version'],
                dependencies=env_config.get('dependencies', {}),
                environment_variables=env_config.get('environment_variables', {}),
                resource_limits=env_config.get('resource_limits', {}),
                security_settings=env_config.get('security_settings', {})
            )
        
        return environments
    
    def create_environment(self, env_name: str, force: bool = False) -> bool:
        """Create a new environment based on configuration."""
        if env_name not in self.environments:
            self.logger.error(f"Environment '{env_name}' not found in configuration")
            return False
        
        env_config = self.environments[env_name]
        self.logger.info(f"Creating environment: {env_name}")
        
        try:
            # Create Docker image for environment
            dockerfile_content = self._generate_dockerfile(env_config)
            dockerfile_path = Path(f"docker/Dockerfile.{env_name}")
            dockerfile_path.parent.mkdir(exist_ok=True)
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            # Build Docker image
            image_tag = f"clinical-analytics:{env_name}"
            build_cmd = [
                "docker", "build",
                "-f", str(dockerfile_path),
                "-t", image_tag,
                "."
            ]
            
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to build environment: {result.stderr}")
                return False
            
            # Create environment-specific docker-compose file
            compose_content = self._generate_docker_compose(env_config, image_tag)
            compose_path = Path(f"docker/docker-compose.{env_name}.yml")
            
            with open(compose_path, 'w') as f:
                f.write(compose_content)
            
            self.logger.info(f"Successfully created environment: {env_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create environment {env_name}: {str(e)}")
            return False
    
    def _generate_dockerfile(self, env_config: EnvironmentConfig) -> str:
        """Generate Dockerfile for specific environment configuration."""
        
        base_image = f"python:{env_config.python_version}-slim"
        
        dockerfile_content = f"""
# Generated Dockerfile for {env_config.name} environment
FROM {base_image}

# Environment metadata
LABEL environment="{env_config.name}"
LABEL type="{env_config.type.value}"
LABEL created="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1
"""
        
        # Add custom environment variables
        for key, value in env_config.environment_variables.items():
            dockerfile_content += f"ENV {key}={value}\n"
        
        # Add system dependencies based on environment type
        if env_config.type == EnvironmentType.DEVELOPMENT:
            dockerfile_content += """
# Development tools
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*
"""
        else:
            dockerfile_content += """
# Minimal production dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*
"""
        
        # Add Python dependencies
        dockerfile_content += """
# Install Python dependencies
WORKDIR /app
COPY requirements.txt ./
"""
        
        # Add environment-specific dependencies
        if env_config.dependencies:
            deps_content = "\n".join([f"{pkg}=={version}" 
                                    for pkg, version in env_config.dependencies.items()])
            dockerfile_content += f"""
# Environment-specific dependencies
RUN echo "{deps_content}" >> requirements.txt
"""
        
        dockerfile_content += """
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd --gid 1000 appuser && \\
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser && \\
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD python -c "import sys; print(f'Python {sys.version}'); exit(0)" || exit 1
"""
        
        return dockerfile_content
    
    def _generate_docker_compose(self, env_config: EnvironmentConfig, image_tag: str) -> str:
        """Generate docker-compose configuration for environment."""
        
        compose_config = {
            'version': '3.8',
            'services': {
                env_config.name: {
                    'image': image_tag,
                    'container_name': f"clinical-{env_config.name}",
                    'volumes': [
                        '.:/app',
                        f"clinical-data-{env_config.name}:/app/data",
                        f"clinical-results-{env_config.name}:/app/results"
                    ],
                    'environment': env_config.environment_variables,
                    'networks': ['clinical-network']
                }
            },
            'volumes': {
                f'clinical-data-{env_config.name}': None,
                f'clinical-results-{env_config.name}': None
            },
            'networks': {
                'clinical-network': {'driver': 'bridge'}
            }
        }
        
        # Add resource limits for production environments
        if env_config.resource_limits:
            compose_config['services'][env_config.name]['deploy'] = {
                'resources': {
                    'limits': env_config.resource_limits
                }
            }
        
        # Add ports for development environments
        if env_config.type == EnvironmentType.DEVELOPMENT:
            compose_config['services'][env_config.name]['ports'] = [
                '8888:8888',  # Jupyter
                '8000:8000'   # FastAPI
            ]
        
        return yaml.dump(compose_config, default_flow_style=False, indent=2)
    
    def validate_environment(self, env_name: str) -> Dict[str, Any]:
        """Validate environment setup and dependencies."""
        validation_results = {
            'environment': env_name,
            'valid': True,
            'issues': [],
            'recommendations': []
        }
        
        if env_name not in self.environments:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Environment '{env_name}' not configured")
            return validation_results
        
        env_config = self.environments[env_name]
        
        # Check if Docker image exists
        image_tag = f"clinical-analytics:{env_name}"
        check_cmd = ["docker", "images", "-q", image_tag]
        
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if not result.stdout.strip():
                validation_results['issues'].append(f"Docker image '{image_tag}' not found")
                validation_results['valid'] = False
        except Exception as e:
            validation_results['issues'].append(f"Failed to check Docker image: {str(e)}")
            validation_results['valid'] = False
        
        # Validate configuration completeness
        if not env_config.dependencies:
            validation_results['recommendations'].append("Consider specifying explicit dependency versions")
        
        if env_config.type == EnvironmentType.PRODUCTION and not env_config.resource_limits:
            validation_results['recommendations'].append("Production environment should have resource limits")
        
        return validation_results
    
    def list_environments(self) -> Dict[str, Dict[str, Any]]:
        """List all configured environments with their status."""
        environment_list = {}
        
        for env_name, env_config in self.environments.items():
            validation = self.validate_environment(env_name)
            
            environment_list[env_name] = {
                'type': env_config.type.value,
                'python_version': env_config.python_version,
                'status': 'valid' if validation['valid'] else 'invalid',
                'issues': validation['issues'],
                'dependency_count': len(env_config.dependencies)
            }
        
        return environment_list
```

### Hands-On Exercise 3: Containerized Analytics Environment

**Scenario**: Create a complete containerized environment for a clinical research team.

**Requirements**:
- Development, testing, and production configurations
- Database integration with PostgreSQL
- Jupyter Lab for interactive analysis
- Automated testing and quality checks
- Resource management and security policies

**Your Task**:
- Design multi-stage Dockerfile with proper security
- Create docker-compose configuration for full stack
- Implement environment validation and monitoring
- Document deployment and maintenance procedures

---

## Part 4: Continuous Integration and Deployment (60 minutes)

### Professional Challenge: Automated Quality Assurance

Modern analytics teams require:
- **Automated testing**: Every code change validated before deployment
- **Quality gates**: Preventing low-quality code from reaching production
- **Automated deployment**: Consistent, error-free releases
- **Monitoring and rollback**: Quick response to production issues

### Advanced CI/CD Pipeline

#### 1. GitHub Actions Workflow

```yaml
# .github/workflows/clinical-analytics-ci.yml
name: Clinical Analytics CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly run on Mondays

env:
  PYTHON_VERSION: '3.10'
  POETRY_VERSION: '1.6.1'

jobs:
  # Code quality and linting
  quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}
        
    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev
        
    - name: Run Black formatting check
      run: |
        black --check --diff src/ tests/
        
    - name: Run isort import sorting check  
      run: |
        isort --check-only --diff src/ tests/
        
    - name: Run flake8 linting
      run: |
        flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
        
    - name: Run mypy type checking
      run: |
        mypy src/ --ignore-missing-imports
        
    - name: Run security analysis with bandit
      run: |
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Upload security report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-report
        path: bandit-report.json

  # Unit and integration tests
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
        
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: testpassword
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install Poetry
      uses: snok/install-poetry@v1
      with:
        version: ${{ env.POETRY_VERSION }}
        
    - name: Install dependencies
      run: |
        poetry config virtualenvs.create false
        poetry install --with dev
        
    - name: Create test data
      run: |
        python scripts/generate_test_data.py
        
    - name: Run pytest with coverage
      env:
        DATABASE_URL: postgresql://postgres:testpassword@localhost:5432/testdb
      run: |
        pytest tests/ \
          --cov=src \
          --cov-report=xml \
          --cov-report=html \
          --cov-report=term \
          --cov-fail-under=80 \
          --junitxml=test-results.xml \
          -v
          
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          test-results.xml
          htmlcov/

  # Documentation build and validation
  docs:
    name: Documentation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        pip install -r docs/requirements.txt
        
    - name: Build documentation
      run: |
        cd docs && make html
        
    - name: Check documentation links
      run: |
        cd docs && make linkcheck
        
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/

  # Docker image build and security scan
  docker:
    name: Docker Build and Scan
    runs-on: ubuntu-latest
    needs: [quality, test]
    steps:
    - uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile
        target: production
        tags: clinical-analytics:${{ github.sha }}
        load: true
        
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'clinical-analytics:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  # Deployment to staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [quality, test, docker]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
      
    - name: Deploy to staging environment
      env:
        STAGING_SERVER: ${{ secrets.STAGING_SERVER }}
        DEPLOY_KEY: ${{ secrets.STAGING_DEPLOY_KEY }}
      run: |
        echo "Deploying to staging environment..."
        # Deployment script would go here
        
    - name: Run smoke tests
      env:
        STAGING_URL: ${{ secrets.STAGING_URL }}
      run: |
        python scripts/smoke_tests.py --url $STAGING_URL
        
    - name: Notify team of staging deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#clinical-analytics'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}

  # Production deployment (manual approval required)
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
      
    - name: Deploy to production environment
      env:
        PROD_SERVER: ${{ secrets.PROD_SERVER }}
        DEPLOY_KEY: ${{ secrets.PROD_DEPLOY_KEY }}
      run: |
        echo "Deploying to production environment..."
        # Production deployment script
        
    - name: Run production health checks
      env:
        PROD_URL: ${{ secrets.PROD_URL }}
      run: |
        python scripts/health_checks.py --url $PROD_URL --timeout 300
        
    - name: Create GitHub release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ github.run_number }}
        release_name: Release v${{ github.run_number }}
        draft: false
        prerelease: false
        
    - name: Notify team of production deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#clinical-analytics-prod'
        webhook_url: ${{ secrets.SLACK_WEBHOOK_PROD }}
```

#### 2. Advanced Monitoring and Alerting

```python
# monitoring_system.py - Professional monitoring for analytics pipelines
import time
import logging
import json
import psutil
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"

class AlertChannel(Enum):
    """Available alert channels."""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]

@dataclass
class AnalyticsMetrics:
    """Analytics-specific metrics."""
    timestamp: datetime
    pipeline_runs: int
    successful_runs: int
    failed_runs: int
    average_runtime: float
    data_quality_score: float
    records_processed: int
    error_rate: float

@dataclass
class Alert:
    """System alert definition."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source_system: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ProfessionalMonitoringSystem:
    """Comprehensive monitoring system for analytics pipelines."""
    
    def __init__(self, config_path: str = "configs/monitoring.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_history: List[Dict] = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default monitoring configuration."""
        return {
            "thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0,
                "error_rate": 5.0,
                "data_quality_score": 75.0
            },
            "alert_channels": {
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "recipients": ["admin@clinicaltrial.org"]
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": ""
                }
            },
            "collection_interval": 60,
            "retention_days": 30
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up monitoring system logging."""
        logger = logging.getLogger("monitoring_system")
        logger.setLevel(logging.INFO)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            'logs/monitoring.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network_io = psutil.net_io_counters()._asdict()
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix systems)
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            load_average = [0.0, 0.0, 0.0]  # Windows fallback
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io=network_io,
            process_count=process_count,
            load_average=load_average
        )
    
    def collect_analytics_metrics(self) -> AnalyticsMetrics:
        """Collect analytics pipeline metrics."""
        
        # This would integrate with your pipeline monitoring
        # For demonstration, using placeholder values
        
        return AnalyticsMetrics(
            timestamp=datetime.now(),
            pipeline_runs=24,
            successful_runs=22,
            failed_runs=2,
            average_runtime=1800.0,  # 30 minutes
            data_quality_score=82.5,
            records_processed=50000,
            error_rate=8.3
        )
    
    def analyze_metrics(self, 
                       system_metrics: SystemMetrics,
                       analytics_metrics: AnalyticsMetrics) -> List[Alert]:
        """Analyze metrics and generate alerts."""
        
        alerts = []
        thresholds = self.config['thresholds']
        
        # System alerts
        if system_metrics.cpu_percent > thresholds['cpu_percent']:
            alerts.append(Alert(
                id=f"cpu_high_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title="High CPU Usage",
                description=f"CPU usage at {system_metrics.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                source_system="system_monitor",
                metrics={"cpu_percent": system_metrics.cpu_percent}
            ))
        
        if system_metrics.memory_percent > thresholds['memory_percent']:
            alerts.append(Alert(
                id=f"memory_high_{int(time.time())}",
                severity=AlertSeverity.CRITICAL if system_metrics.memory_percent > 95 else AlertSeverity.WARNING,
                title="High Memory Usage",
                description=f"Memory usage at {system_metrics.memory_percent:.1f}%",
                timestamp=datetime.now(),
                source_system="system_monitor",
                metrics={"memory_percent": system_metrics.memory_percent}
            ))
        
        if system_metrics.disk_percent > thresholds['disk_percent']:
            alerts.append(Alert(
                id=f"disk_high_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                title="High Disk Usage",
                description=f"Disk usage at {system_metrics.disk_percent:.1f}%",
                timestamp=datetime.now(),
                source_system="system_monitor",
                metrics={"disk_percent": system_metrics.disk_percent}
            ))
        
        # Analytics alerts
        if analytics_metrics.error_rate > thresholds['error_rate']:
            alerts.append(Alert(
                id=f"error_rate_high_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title="High Pipeline Error Rate",
                description=f"Error rate at {analytics_metrics.error_rate:.1f}%",
                timestamp=datetime.now(),
                source_system="analytics_monitor",
                metrics={"error_rate": analytics_metrics.error_rate}
            ))
        
        if analytics_metrics.data_quality_score < thresholds['data_quality_score']:
            alerts.append(Alert(
                id=f"quality_low_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                title="Low Data Quality Score",
                description=f"Data quality score at {analytics_metrics.data_quality_score:.1f}",
                timestamp=datetime.now(),
                source_system="analytics_monitor",
                metrics={"data_quality_score": analytics_metrics.data_quality_score}
            ))
        
        return alerts
    
    def send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        
        self.logger.warning(f"ALERT: {alert.title} - {alert.description}")
        
        # Store active alert
        self.active_alerts[alert.id] = alert
        
        # Send through email if configured
        email_config = self.config['alert_channels'].get('email', {})
        if email_config.get('enabled', False):
            self._send_email_alert(alert, email_config)
        
        # Send through Slack if configured
        slack_config = self.config['alert_channels'].get('slack', {})
        if slack_config.get('enabled', False):
            self._send_slack_alert(alert, slack_config)
    
    def _send_email_alert(self, alert: Alert, email_config: Dict):
        """Send alert via email."""
        
        try:
            msg = MimeMultipart()
            msg['From'] = email_config.get('sender', 'monitoring@system.local')
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Severity: {alert.severity.value.upper()}
            - Source: {alert.source_system}
            - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            - Description: {alert.description}
            
            Metrics:
            {json.dumps(alert.metrics, indent=2)}
            
            Please investigate and resolve this issue promptly.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                if 'username' in email_config:
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Email alert sent for: {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    def _send_slack_alert(self, alert: Alert, slack_config: Dict):
        """Send alert via Slack webhook."""
        
        try:
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500", 
                AlertSeverity.CRITICAL: "#ff0000",
                AlertSeverity.FATAL: "#8B0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Source", "value": alert.source_system, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            response = requests.post(slack_config['webhook_url'], json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for: {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def run_monitoring_loop(self):
        """Main monitoring loop."""
        
        self.logger.info("Starting monitoring system")
        
        while True:
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                analytics_metrics = self.collect_analytics_metrics()
                
                # Store metrics for historical analysis
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'system': asdict(system_metrics),
                    'analytics': asdict(analytics_metrics)
                })
                
                # Analyze and generate alerts
                new_alerts = self.analyze_metrics(system_metrics, analytics_metrics)
                
                # Send new alerts
                for alert in new_alerts:
                    self.send_alert(alert)
                
                # Clean up resolved alerts
                self._cleanup_resolved_alerts()
                
                # Sleep until next collection
                time.sleep(self.config['collection_interval'])
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(30)  # Shorter sleep on error
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.resolution_time and alert.resolution_time < cutoff_time
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health summary."""
        
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest_metrics = self.metrics_history[-1]
        active_alerts_count = len([a for a in self.active_alerts.values() if not a.resolved])
        
        # Determine overall health status
        critical_alerts = len([a for a in self.active_alerts.values() 
                              if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.FATAL] and not a.resolved])
        
        if critical_alerts > 0:
            status = "critical"
        elif active_alerts_count > 0:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "timestamp": latest_metrics['timestamp'],
            "active_alerts": active_alerts_count,
            "critical_alerts": critical_alerts,
            "system_metrics": latest_metrics['system'],
            "analytics_metrics": latest_metrics['analytics'],
            "uptime": self._calculate_uptime()
        }
    
    def _calculate_uptime(self) -> Dict[str, float]:
        """Calculate system uptime metrics."""
        
        if len(self.metrics_history) < 2:
            return {"uptime_hours": 0, "availability_percent": 100}
        
        total_time = (datetime.now() - 
                     datetime.fromisoformat(self.metrics_history[0]['timestamp'])).total_seconds()
        
        # Calculate downtime based on critical alerts
        downtime = sum([
            (alert.resolution_time or datetime.now()) - alert.timestamp 
            for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL
        ], timedelta()).total_seconds()
        
        uptime_hours = (total_time - downtime) / 3600
        availability_percent = ((total_time - downtime) / total_time) * 100 if total_time > 0 else 100
        
        return {
            "uptime_hours": uptime_hours,
            "availability_percent": availability_percent
        }

# Usage example
if __name__ == "__main__":
    monitoring = ProfessionalMonitoringSystem()
    
    try:
        monitoring.run_monitoring_loop()
    except KeyboardInterrupt:
        print("\nMonitoring system stopped")
```

### Final Exercise: Complete CI/CD Pipeline

**Scenario**: Implement complete CI/CD pipeline for clinical analytics platform.

**Requirements**:
- Automated testing with multiple Python versions
- Security scanning and vulnerability assessment
- Documentation validation and generation
- Staging and production deployment with approval gates
- Comprehensive monitoring and alerting

**Your Task**:
- Create GitHub Actions workflow with all quality gates
- Implement monitoring system with multiple alert channels
- Design rollback and recovery procedures
- Document deployment processes and troubleshooting guides

---

## Assessment and Professional Integration

### Capstone Project: Production-Ready Analytics Platform

**Project Overview**: Complete productionization of clinical analytics system

**Components to Deliver**:

1. **Automated Workflow Pipeline** (25 points)
   - Apache Airflow DAG with error handling and recovery
   - Multi-source data ingestion with validation
   - Automated quality assessment and reporting
   - Email and Slack notifications with proper escalation

2. **Professional Code Quality** (25 points)
   - Comprehensive test suite with >80% coverage
   - Professional documentation with examples
   - Type hints and static analysis compliance
   - Code style standardization with automated formatting

3. **Containerized Environment** (25 points)
   - Multi-stage Docker builds for dev/test/prod
   - Docker Compose orchestration with services
   - Environment-specific configurations
   - Security best practices and resource management

4. **CI/CD Implementation** (25 points)
   - GitHub Actions workflow with quality gates
   - Automated testing and security scanning
   - Staging and production deployment pipelines
   - Monitoring system with comprehensive alerting

### Professional Development Competencies

After this lecture, you should demonstrate mastery in:

**Workflow Automation**:
- [ ] Design complex automated workflows using orchestration platforms
- [ ] Implement robust error handling with automatic recovery mechanisms
- [ ] Create comprehensive monitoring and alerting systems
- [ ] Build scalable data processing pipelines

**Code Quality Standards**:
- [ ] Write comprehensive test suites for analytical code
- [ ] Create professional documentation following industry standards
- [ ] Implement automated code quality checks and enforcement
- [ ] Design effective code review processes

**Environment Management**:
- [ ] Create reproducible environments using containerization
- [ ] Design multi-stage Docker builds for different deployment targets
- [ ] Implement proper security practices and resource management
- [ ] Manage environment-specific configurations effectively

**CI/CD Implementation**:
- [ ] Build automated testing and deployment pipelines
- [ ] Implement quality gates preventing low-quality code deployment
- [ ] Design monitoring and alerting for production systems
- [ ] Create rollback and recovery procedures for production issues

### Looking Ahead to L10

In our final lecture, we'll integrate all professional competencies into comprehensive applied projects:
- **End-to-end clinical research projects** demonstrating full analytical lifecycle
- **Professional communication** with stakeholders at all levels
- **Integration with clinical workflows** and decision support systems
- **Career development** strategies for senior data science roles

The professional development practices you've mastered today provide the foundation for delivering reliable, scalable analytics solutions that meet enterprise and regulatory standards.

---

## Additional Resources

### Automation and Orchestration
- **Apache Airflow Documentation**: Complete workflow management guide
- **Prefect**: Modern workflow orchestration alternative
- **Dagster**: Data-aware orchestration for analytics

### Code Quality and Testing
- **pytest Documentation**: Advanced testing patterns
- **Black**: Python code formatting standard
- **mypy**: Static type checking for Python
- **pre-commit**: Git hooks for code quality

### Containerization
- **Docker Best Practices**: Security and optimization guides
- **Kubernetes**: Container orchestration for production scale
- **Docker Compose**: Multi-container application management

### CI/CD and DevOps
- **GitHub Actions**: Complete automation platform
- **GitLab CI/CD**: Alternative CI/CD platform
- **Jenkins**: Traditional CI/CD server
- **Terraform**: Infrastructure as code

### Monitoring and Observability
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboarding
- **ELK Stack**: Centralized logging and analysis
- **DataDog**: Comprehensive monitoring platform

### Professional Development
- **Clean Code**: Robert Martin's software craftsmanship principles
- **The DevOps Handbook**: Cultural and technical transformation
- **Continuous Delivery**: Reliable software releases through automation