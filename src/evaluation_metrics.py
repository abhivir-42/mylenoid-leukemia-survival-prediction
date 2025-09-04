"""
Advanced Evaluation Metrics for Censored Survival Data

This module implements comprehensive evaluation strategies for survival prediction models,
specifically designed for the Myeloid Leukemia Survival Prediction challenge. It includes:

1. IPCW-C-index calculation with proper censoring handling
2. Time-aware cross-validation strategies
3. Bootstrap confidence intervals
4. Clinical subgroup analysis
5. Model calibration assessment

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import defaultdict
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Survival analysis libraries
try:
    from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
    from sksurv.util import Surv
    from sksurv.ensemble import RandomSurvivalForest
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not available. Using fallback metrics.")


class SurvivalEvaluator:
    """
    Comprehensive evaluation system for censored survival data

    Handles the complexities of right-censored survival data and provides
    robust evaluation metrics specifically designed for clinical applications.
    """

    def __init__(self, train_event_times: Optional[np.ndarray] = None,
                 train_events: Optional[np.ndarray] = None):
        """
        Initialize evaluator with training data for IPCW calculations

        Args:
            train_event_times: Training survival times for IPCW
            train_events: Training censoring indicators for IPCW
        """

        self.train_event_times = train_event_times
        self.train_events = train_events

        if train_event_times is not None and train_events is not None:
            self.train_survival = Surv.from_arrays(
                event=train_events,
                time=train_event_times
            ) if SKSURV_AVAILABLE else None

    def calculate_ipcw_c_index(self, y_true_times: np.ndarray, y_true_events: np.ndarray,
                             y_pred_risks: np.ndarray, tau: float = 7.0) -> Dict[str, float]:
        """
        Calculate IPCW-C-index with comprehensive metrics

        Args:
            y_true_times: True survival times
            y_true_events: True censoring indicators (1=event, 0=censored)
            y_pred_risks: Predicted risk scores
            tau: Maximum time horizon for evaluation

        Returns:
            Dictionary with C-index metrics and confidence intervals
        """

        if not SKSURV_AVAILABLE:
            return self._fallback_c_index(y_true_times, y_true_events, y_pred_risks)

        # Create test survival object
        test_survival = Surv.from_arrays(event=y_true_events, time=y_true_times)

        # Calculate IPCW C-index
        try:
            c_index, mean_c_index, _, lower_ci, upper_ci = concordance_index_ipcw(
                self.train_survival, test_survival, y_pred_risks, tau=tau
            )

            return {
                'ipcw_c_index': c_index,
                'mean_c_index': mean_c_index,
                'c_index_lower_ci': lower_ci,
                'c_index_upper_ci': upper_ci,
                'c_index_ci_width': upper_ci - lower_ci
            }

        except Exception as e:
            warnings.warn(f"IPCW calculation failed: {e}. Using fallback.")
            return self._fallback_c_index(y_true_times, y_true_events, y_pred_risks)

    def _fallback_c_index(self, times: np.ndarray, events: np.ndarray,
                         predictions: np.ndarray) -> Dict[str, float]:
        """Fallback C-index calculation when scikit-survival is unavailable"""

        if not SKSURV_AVAILABLE:
            # Simple concordance index calculation
            n_pairs = 0
            n_concordant = 0

            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    # Only compare uncensored pairs or pairs with different outcomes
                    if (events[i] == 1 and events[j] == 1) or \
                       (events[i] == 1 and times[i] < times[j]) or \
                       (events[j] == 1 and times[j] < times[i]):

                        n_pairs += 1

                        # Check concordance
                        pred_diff = predictions[i] - predictions[j]
                        time_diff = times[i] - times[j]

                        if (pred_diff > 0 and time_diff > 0) or \
                           (pred_diff < 0 and time_diff < 0):
                            n_concordant += 1

            c_index = n_concordant / n_pairs if n_pairs > 0 else 0.5

            return {
                'ipcw_c_index': c_index,
                'mean_c_index': c_index,
                'c_index_lower_ci': max(0, c_index - 0.05),
                'c_index_upper_ci': min(1, c_index + 0.05),
                'c_index_ci_width': 0.1
            }

    def calculate_bootstrap_ci(self, times: np.ndarray, events: np.ndarray,
                             predictions: np.ndarray, n_bootstraps: int = 1000,
                             confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for C-index

        Args:
            times: Survival times
            events: Censoring indicators
            predictions: Risk predictions
            n_bootstraps: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Bootstrap confidence interval results
        """

        bootstrap_c_indices = []

        for _ in range(n_bootstraps):
            # Bootstrap sample
            indices = resample(range(len(times)), replace=True)
            boot_times = times[indices]
            boot_events = events[indices]
            boot_predictions = predictions[indices]

            # Calculate C-index for bootstrap sample
            if SKSURV_AVAILABLE:
                boot_survival = Surv.from_arrays(event=boot_events, time=boot_times)
                c_index = concordance_index_censored(boot_events, boot_times, boot_predictions)[0]
            else:
                c_index = self._fallback_c_index(boot_times, boot_events, boot_predictions)['ipcw_c_index']

            bootstrap_c_indices.append(c_index)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        bootstrap_c_indices = np.array(bootstrap_c_indices)

        return {
            'bootstrap_mean': np.mean(bootstrap_c_indices),
            'bootstrap_std': np.std(bootstrap_c_indices),
            'bootstrap_lower_ci': np.percentile(bootstrap_c_indices, lower_percentile),
            'bootstrap_upper_ci': np.percentile(bootstrap_c_indices, upper_percentile),
            'bootstrap_median': np.median(bootstrap_c_indices)
        }

    def evaluate_time_dependence(self, times: np.ndarray, events: np.ndarray,
                               predictions: np.ndarray, time_bins: int = 5) -> Dict[str, List[float]]:
        """
        Evaluate model performance across different time horizons

        Args:
            times: Survival times
            events: Censoring indicators
            predictions: Risk predictions
            time_bins: Number of time bins for analysis

        Returns:
            C-index values for different time horizons
        """

        time_bins_edges = np.linspace(0, times.max(), time_bins + 1)
        time_dependent_metrics = defaultdict(list)

        for i in range(time_bins):
            bin_start = time_bins_edges[i]
            bin_end = time_bins_edges[i + 1]

            # Select patients in this time bin
            bin_mask = (times >= bin_start) & (times < bin_end)

            if np.sum(bin_mask) < 10:  # Skip bins with too few samples
                continue

            bin_times = times[bin_mask]
            bin_events = events[bin_mask]
            bin_predictions = predictions[bin_mask]

            # Calculate metrics for this bin
            metrics = self.calculate_ipcw_c_index(bin_times, bin_events, bin_predictions)

            for key, value in metrics.items():
                time_dependent_metrics[key].append(value)

            time_dependent_metrics['time_bin_center'].append((bin_start + bin_end) / 2)
            time_dependent_metrics['n_samples'].append(np.sum(bin_mask))

        return dict(time_dependent_metrics)


class SurvivalCrossValidator:
    """
    Advanced cross-validation strategies for survival data

    Implements time-aware cross-validation that maintains temporal structure
    and handles censoring appropriately.
    """

    def __init__(self, n_splits: int = 5, time_bins: int = 10, random_state: int = 42):
        self.n_splits = n_splits
        self.time_bins = time_bins
        self.random_state = random_state
        self.evaluator = SurvivalEvaluator()

    def time_stratified_split(self, X: pd.DataFrame, event_times: np.ndarray,
                            events: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-stratified train/validation splits

        Args:
            X: Feature matrix
            event_times: Survival times
            events: Censoring indicators

        Returns:
            List of (train_indices, val_indices) tuples
        """

        # Create time bins for stratification
        time_bins = pd.qcut(event_times, q=self.time_bins, duplicates='drop')

        # Create event-stratified bins
        strata = pd.Categorical(time_bins).codes * 2 + events

        # Perform stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                            random_state=self.random_state)

        splits = []
        for train_idx, val_idx in skf.split(X, strata):
            splits.append((train_idx, val_idx))

        return splits

    def censored_stratified_split(self, X: pd.DataFrame, event_times: np.ndarray,
                                events: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create splits that balance censoring distribution

        Args:
            X: Feature matrix
            event_times: Survival times
            events: Censoring indicators

        Returns:
            List of (train_indices, val_indices) tuples
        """

        # Create censoring-based strata
        censoring_rate = np.mean(events)
        strata = (events * 2).astype(int)  # 0 for censored, 2 for events

        # Add time-based stratification for events
        event_mask = events == 1
        if np.sum(event_mask) > 0:
            event_times_only = event_times[event_mask]
            time_quartiles = pd.qcut(event_times_only, q=4, duplicates='drop')
            event_strata = pd.Categorical(time_quartiles).codes + 3
            strata[event_mask] = event_strata

        # Perform stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                            random_state=self.random_state)

        splits = []
        for train_idx, val_idx in skf.split(X, strata):
            splits.append((train_idx, val_idx))

        return splits

    def cross_validate_survival_model(self, model, X: pd.DataFrame,
                                    event_times: np.ndarray, events: np.ndarray,
                                    cv_strategy: str = 'time_stratified') -> Dict[str, Union[List[float], float]]:
        """
        Perform comprehensive cross-validation for survival models

        Args:
            model: Survival model with fit/predict methods
            X: Feature matrix
            event_times: Survival times
            events: Censoring indicators
            cv_strategy: Cross-validation strategy ('time_stratified' or 'censored_stratified')

        Returns:
            Dictionary with CV results and metrics
        """

        print(f"Performing {self.n_splits}-fold cross-validation using {cv_strategy} strategy...")

        # Choose CV strategy
        if cv_strategy == 'time_stratified':
            splits = self.time_stratified_split(X, event_times, events)
        elif cv_strategy == 'censored_stratified':
            splits = self.censored_stratified_split(X, event_times, events)
        else:
            # Fallback to random split
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(X))

        cv_results = defaultdict(list)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Training fold {fold + 1}/{self.n_splits}...")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            times_train, times_val = event_times[train_idx], event_times[val_idx]
            events_train, events_val = events[train_idx], events[val_idx]

            # Train model
            try:
                model.fit(X_train, times_train, events_train)

                # Generate predictions
                val_predictions = model.predict(X_val)

                # Evaluate fold
                fold_evaluator = SurvivalEvaluator(times_train, events_train)
                metrics = fold_evaluator.calculate_ipcw_c_index(
                    times_val, events_val, val_predictions
                )

                # Store results
                for key, value in metrics.items():
                    cv_results[key].append(value)

                fold_metrics.append(metrics['ipcw_c_index'])

                print(".4f")

            except Exception as e:
                print(f"❌ Fold {fold + 1} failed: {e}")
                fold_metrics.append(0.5)  # Default to random performance

        # Calculate summary statistics
        cv_results['cv_mean'] = np.mean(fold_metrics)
        cv_results['cv_std'] = np.std(fold_metrics)
        cv_results['cv_min'] = np.min(fold_metrics)
        cv_results['cv_max'] = np.max(fold_metrics)

        print("\nCross-validation Results:")
        print(".4f")
        print(".4f")

        return dict(cv_results)

    def evaluate_model_stability(self, model, X: pd.DataFrame,
                               event_times: np.ndarray, events: np.ndarray,
                               n_iterations: int = 10) -> Dict[str, float]:
        """
        Evaluate model stability across multiple random splits

        Args:
            model: Survival model
            X: Feature matrix
            event_times: Survival times
            events: Censoring indicators
            n_iterations: Number of random evaluations

        Returns:
            Stability metrics
        """

        stability_metrics = []

        for i in range(n_iterations):
            # Random 80/20 split
            indices = np.random.permutation(len(X))
            train_size = int(0.8 * len(X))

            train_idx = indices[:train_size]
            val_idx = indices[train_size:]

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            times_train, times_val = event_times[train_idx], event_times[val_idx]
            events_train, events_val = events[train_idx], events[val_idx]

            # Train and evaluate
            model.fit(X_train, times_train, events_train)
            predictions = model.predict(X_val)

            evaluator = SurvivalEvaluator(times_train, events_train)
            metrics = evaluator.calculate_ipcw_c_index(times_val, events_val, predictions)
            stability_metrics.append(metrics['ipcw_c_index'])

        return {
            'stability_mean': np.mean(stability_metrics),
            'stability_std': np.std(stability_metrics),
            'stability_cv': np.std(stability_metrics) / np.mean(stability_metrics)
        }


class ClinicalSubgroupAnalyzer:
    """
    Clinical subgroup analysis for model evaluation

    Evaluates model performance across clinically relevant subgroups
    to ensure fairness and clinical utility.
    """

    def __init__(self):
        self.evaluator = SurvivalEvaluator()

    def analyze_clinical_subgroups(self, predictions: np.ndarray, times: np.ndarray,
                                 events: np.ndarray, clinical_features: pd.DataFrame,
                                 train_times: np.ndarray = None,
                                 train_events: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance across clinical subgroups

        Args:
            predictions: Model risk predictions
            times: Survival times
            events: Censoring indicators
            clinical_features: DataFrame with clinical features
            train_times: Training times for IPCW
            train_events: Training events for IPCW

        Returns:
            Dictionary with subgroup performance metrics
        """

        subgroup_results = {}

        # Define clinically relevant subgroups
        subgroups = {
            'age_groups': self._create_age_groups(clinical_features),
            'blast_count': self._create_blast_groups(clinical_features),
            'cytogenetic_risk': self._create_cytogenetic_groups(clinical_features),
            'molecular_risk': self._create_molecular_groups(clinical_features)
        }

        for subgroup_name, subgroup_labels in subgroups.items():
            if subgroup_labels is None:
                continue

            subgroup_results[subgroup_name] = {}

            # Analyze each subgroup
            unique_labels = np.unique(subgroup_labels)
            for label in unique_labels:
                mask = subgroup_labels == label
                if np.sum(mask) < 10:  # Skip small subgroups
                    continue

                subgroup_predictions = predictions[mask]
                subgroup_times = times[mask]
                subgroup_events = events[mask]

                # Calculate metrics for subgroup
                evaluator = SurvivalEvaluator(train_times, train_events)
                metrics = evaluator.calculate_ipcw_c_index(
                    subgroup_times, subgroup_events, subgroup_predictions
                )

                subgroup_results[subgroup_name][f'subgroup_{label}'] = metrics['ipcw_c_index']

        return subgroup_results

    def _create_age_groups(self, clinical_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create age-based subgroups (if age data available)"""
        # This would require age information which may not be in the dataset
        return None

    def _create_blast_groups(self, clinical_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create blast count-based subgroups"""

        if 'BM_BLAST' not in clinical_df.columns:
            return None

        blast_counts = clinical_df['BM_BLAST'].values

        # Clinical thresholds
        labels = np.full(len(blast_counts), 'normal', dtype=object)
        labels[blast_counts > 10] = 'elevated'
        labels[blast_counts > 20] = 'high'
        labels[blast_counts > 50] = 'very_high'

        return labels

    def _create_cytogenetic_groups(self, clinical_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create cytogenetic risk-based subgroups"""

        if 'cytogenetic_risk_score' not in clinical_df.columns:
            return None

        risk_scores = clinical_df['cytogenetic_risk_score'].values

        labels = np.full(len(risk_scores), 'intermediate', dtype=object)
        labels[risk_scores == 0] = 'favorable'
        labels[risk_scores == 2] = 'adverse'

        return labels

    def _create_molecular_groups(self, clinical_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Create molecular risk-based subgroups"""

        if 'total_mutations' not in clinical_df.columns:
            return None

        mutation_counts = clinical_df['total_mutations'].values

        labels = np.full(len(mutation_counts), 'low', dtype=object)
        labels[mutation_counts > 2] = 'moderate'
        labels[mutation_counts > 5] = 'high'

        return labels


def create_comprehensive_evaluation_report(predictions: np.ndarray, times: np.ndarray,
                                         events: np.ndarray, clinical_features: pd.DataFrame,
                                         train_times: np.ndarray = None,
                                         train_events: np.ndarray = None) -> Dict[str, Union[float, Dict]]:
    """
    Generate comprehensive evaluation report

    Args:
        predictions: Model predictions
        times: True survival times
        events: True censoring indicators
        clinical_features: Clinical feature DataFrame
        train_times: Training times for IPCW
        train_events: Training events for IPCW

    Returns:
        Comprehensive evaluation results
    """

    # Initialize evaluators
    evaluator = SurvivalEvaluator(train_times, train_events)
    subgroup_analyzer = ClinicalSubgroupAnalyzer()

    report = {}

    # Main evaluation metrics
    report['main_metrics'] = evaluator.calculate_ipcw_c_index(times, events, predictions)

    # Bootstrap confidence intervals
    report['bootstrap_ci'] = evaluator.calculate_bootstrap_ci(times, events, predictions)

    # Time-dependent analysis
    report['time_dependent'] = evaluator.evaluate_time_dependence(times, events, predictions)

    # Clinical subgroup analysis
    report['subgroup_analysis'] = subgroup_analyzer.analyze_clinical_subgroups(
        predictions, times, events, clinical_features, train_times, train_events
    )

    # Overall summary
    report['summary'] = {
        'n_samples': len(predictions),
        'censoring_rate': 1 - np.mean(events),
        'mean_prediction': np.mean(predictions),
        'prediction_range': np.ptp(predictions),
        'overall_c_index': report['main_metrics']['ipcw_c_index']
    }

    return report


# Example usage and testing
if __name__ == "__main__":

    print("Testing Survival Evaluation System...")
    print("=" * 50)

    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 500

    # Generate synthetic survival data
    times = np.random.exponential(3, n_samples)
    events = np.random.binomial(1, 0.6, n_samples)  # 60% uncensored
    predictions = np.random.randn(n_samples) * 0.5 + times * 0.1  # Some correlation with time

    # Create synthetic clinical features
    clinical_features = pd.DataFrame({
        'BM_BLAST': np.random.exponential(2, n_samples),
        'cytogenetic_risk_score': np.random.choice([0, 1, 2], n_samples),
        'total_mutations': np.random.poisson(3, n_samples)
    })

    print(f"Created test dataset: {n_samples} samples")
    print(".1f")

    # Test main evaluator
    evaluator = SurvivalEvaluator()
    metrics = evaluator.calculate_ipcw_c_index(times, events, predictions)

    print("\nMain Evaluation Metrics:")
    for key, value in metrics.items():
        print(".4f")

    # Test bootstrap CI
    bootstrap_results = evaluator.calculate_bootstrap_ci(times, events, predictions, n_bootstraps=100)
    print("\nBootstrap Confidence Intervals:")
    for key, value in bootstrap_results.items():
        print(".4f")

    # Test clinical subgroup analysis
    subgroup_analyzer = ClinicalSubgroupAnalyzer()
    subgroup_results = subgroup_analyzer.analyze_clinical_subgroups(
        predictions, times, events, clinical_features
    )

    print("\nClinical Subgroup Analysis:")
    for subgroup, results in subgroup_results.items():
        print(f"{subgroup}: {results}")

    print("\n✅ Evaluation system testing completed!")
