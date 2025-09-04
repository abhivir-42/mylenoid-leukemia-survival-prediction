"""
Advanced Ensemble Survival Modeling for Myeloid Leukemia

This module implements a comprehensive ensemble survival prediction system that combines:
1. Traditional survival models (Cox PH, Random Survival Forest)
2. Deep learning survival models with FastAI integration
3. Advanced ensemble techniques with stacking
4. Custom loss functions for censored survival data

"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import pickle
from dataclasses import dataclass

# Survival analysis libraries
try:
    from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    print("Warning: scikit-survival not available. Traditional survival models disabled.")

# FastAI imports
try:
    from fastai.tabular.all import *
    from fastai.callback.all import *
    FASTAI_AVAILABLE = True
except ImportError:
    FASTAI_AVAILABLE = False
    print("Warning: FastAI not available. Deep learning survival models disabled.")


@dataclass
class ModelConfig:
    """Configuration for survival model training"""
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    hidden_layers: List[int] = None
    dropout: float = 0.3
    weight_decay: float = 1e-4

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]


class DeepSurvivalModel(nn.Module):
    """Custom deep neural network for survival prediction"""

    def __init__(self, n_features: int, config: ModelConfig):
        super().__init__()

        layers = []
        in_features = n_features

        # Build hidden layers
        for out_features in config.hidden_layers:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_features = out_features

        # Survival head
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CoxLoss(nn.Module):
    """Custom Cox Proportional Hazards loss for censored survival data"""

    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, times: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative partial log-likelihood for Cox model

        Args:
            predictions: Model risk predictions (higher = higher risk)
            times: Survival times
            events: Censoring indicators (1 = event, 0 = censored)

        Returns:
            Negative partial log-likelihood
        """

        # Sort by time (descending for risk set calculation)
        sorted_indices = torch.argsort(times, descending=True)
        predictions = predictions[sorted_indices]
        times = times[sorted_indices]
        events = events[sorted_indices]

        # Calculate risk scores (exp of predictions for Cox model)
        risk_scores = torch.exp(predictions)

        # Calculate cumulative risk for each time point
        n_samples = len(predictions)
        loss = 0.0

        for i in range(n_samples):
            if events[i] == 1:  # Only uncensored observations contribute
                # Risk set: all individuals with time >= current time
                risk_set_mask = times >= times[i]
                risk_set_sum = torch.sum(risk_scores[risk_set_mask])

                # Log partial likelihood contribution
                if risk_set_sum > 0:
                    loss -= predictions[i] - torch.log(risk_set_sum)

        return loss / torch.sum(events)  # Average over uncensored observations


class LeukemiaSurvivalPredictor:
    """
    Advanced ensemble survival prediction system for myeloid leukemia

    Combines multiple survival modeling approaches:
    1. Traditional statistical models (Cox PH, Random Survival Forest)
    2. Deep learning models with FastAI integration
    3. Ensemble stacking with meta-learning
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.meta_model = None
        self.feature_scaler = None
        self.is_trained = False

        # Initialize available models
        self._init_models()

    def _init_models(self):
        """Initialize all available survival models"""

        if SKSURV_AVAILABLE:
            # Traditional survival models
            self.models['cox_ph'] = CoxPHSurvivalAnalysis(alpha=0.01)
            self.models['rsf'] = RandomSurvivalForest(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                random_state=42
            )
            self.models['extra_trees'] = ExtraSurvivalTrees(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )

        if FASTAI_AVAILABLE:
            # Deep learning model will be initialized during training
            self.models['deep_survival'] = None

    def fit(self, X: pd.DataFrame, y_train_times: np.ndarray,
            y_train_events: np.ndarray, validation_data: Tuple = None):
        """
        Train the ensemble survival prediction system

        Args:
            X: Feature matrix
            y_train_times: Survival times
            y_train_events: Censoring indicators
            validation_data: Optional (X_val, times_val, events_val) tuple
        """

        print("Training Leukemia Survival Prediction Ensemble...")
        print(f"Training on {len(X)} patients with {X.shape[1]} features")

        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create structured survival data
        if SKSURV_AVAILABLE:
            y_train_structured = Surv.from_arrays(
                event=y_train_events,
                time=y_train_times
            )

        # Train base models
        base_predictions = self._train_base_models(
            X_scaled, y_train_structured, y_train_times, y_train_events
        )

        # Train deep learning model
        if FASTAI_AVAILABLE:
            deep_predictions = self._train_deep_model(
                X_scaled, y_train_times, y_train_events, validation_data
            )
            base_predictions['deep_survival'] = deep_predictions

        # Train meta-model
        self._train_meta_model(base_predictions, y_train_times, y_train_events)

        self.is_trained = True
        print("✅ Ensemble training completed!")

    def _train_base_models(self, X_scaled: pd.DataFrame, y_structured,
                          times: np.ndarray, events: np.ndarray) -> Dict[str, np.ndarray]:
        """Train traditional survival models"""

        base_predictions = {}

        for model_name, model in self.models.items():
            if model is None or model_name == 'deep_survival':
                continue

            print(f"Training {model_name}...")
            try:
                model.fit(X_scaled, y_structured)
                predictions = model.predict(X_scaled)
                base_predictions[model_name] = predictions
                print(".4f")
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {e}")

        return base_predictions

    def _train_deep_model(self, X_scaled: pd.DataFrame, times: np.ndarray,
                         events: np.ndarray, validation_data: Tuple = None) -> np.ndarray:
        """Train deep learning survival model with FastAI"""

        print("Training Deep Survival Model...")

        # Prepare data
        train_df = X_scaled.copy()
        train_df['time'] = times
        train_df['event'] = events

        # Create continuous and categorical splits
        cont_names = X_scaled.columns.tolist()
        cat_names = []

        # Create DataLoaders
        if validation_data is not None:
            X_val, times_val, events_val = validation_data
            val_df = X_val.copy()
            val_df['time'] = times_val
            val_df['event'] = events_val

            dls = TabularDataLoaders.from_df(
                train_df, path='.', cat_names=cat_names, cont_names=cont_names,
                y_names=['time', 'event'], valid_df=val_df, bs=self.config.batch_size
            )
        else:
            dls = TabularDataLoaders.from_df(
                train_df, path='.', cat_names=cat_names, cont_names=cont_names,
                y_names=['time', 'event'], valid_pct=0.2, bs=self.config.batch_size
            )

        # Create model
        n_features = len(cont_names)
        deep_model = DeepSurvivalModel(n_features, self.config)

        # Custom loss function
        cox_loss = CoxLoss()

        # Create learner
        learn = Learner(
            dls, deep_model,
            loss_func=cox_loss,
            metrics=[self._c_index_metric],
            opt_func=ranger,
            wd=self.config.weight_decay
        )

        # Training callbacks
        callbacks = [
            EarlyStoppingCallback(patience=10, monitor='c_index', min_delta=0.001),
            SaveModelCallback(monitor='c_index', fname='best_deep_model'),
            GradientClip(max_norm=1.0)
        ]

        # Train model
        learn.fit_one_cycle(
            self.config.epochs,
            lr_max=self.config.learning_rate,
            cbs=callbacks
        )

        # Store trained model
        self.models['deep_survival'] = learn

        # Get training predictions
        train_dl = dls.train
        predictions = []

        learn.model.eval()
        with torch.no_grad():
            for batch in train_dl:
                pred = learn.model(batch[0])  # Only features, not targets
                predictions.extend(pred.squeeze().cpu().numpy())

        return np.array(predictions)

    def _train_meta_model(self, base_predictions: Dict[str, np.ndarray],
                         times: np.ndarray, events: np.ndarray):
        """Train meta-model for ensemble stacking"""

        print("Training Meta-Model...")

        # Stack base model predictions
        meta_features = np.column_stack(list(base_predictions.values()))

        # Use Cox PH as meta-model for survival prediction
        if SKSURV_AVAILABLE:
            meta_y = Surv.from_arrays(event=events, time=times)
            self.meta_model = CoxPHSurvivalAnalysis(alpha=0.01)
            self.meta_model.fit(meta_features, meta_y)

            # Evaluate meta-model
            meta_predictions = self.meta_model.predict(meta_features)
            c_index = concordance_index_censored(events, times, meta_predictions)[0]
            print(".4f")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble survival predictions

        Args:
            X: Feature matrix for prediction

        Returns:
            Risk scores (higher = higher risk of death)
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Get base model predictions
        base_predictions = []

        for model_name, model in self.models.items():
            if model is None:
                continue

            if model_name == 'deep_survival':
                # Deep model prediction
                pred = self._predict_deep_model(X_scaled)
            else:
                # Traditional model prediction
                pred = model.predict(X_scaled)

            base_predictions.append(pred)

        # Stack predictions for meta-model
        if len(base_predictions) > 1 and self.meta_model is not None:
            meta_features = np.column_stack(base_predictions)
            final_predictions = self.meta_model.predict(meta_features)
        else:
            # Simple averaging if no meta-model
            final_predictions = np.mean(base_predictions, axis=0)

        return final_predictions

    def _predict_deep_model(self, X_scaled: np.ndarray) -> np.ndarray:
        """Generate predictions from deep learning model"""

        if self.models['deep_survival'] is None:
            raise ValueError("Deep model not trained")

        learn = self.models['deep_survival']

        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        learn.model.eval()
        with torch.no_grad():
            predictions = learn.model(X_tensor).squeeze().cpu().numpy()

        return predictions

    def _c_index_metric(self, preds, targets):
        """Custom C-index metric for FastAI training"""

        # Extract predictions and targets
        predictions = preds.squeeze()
        times, events = targets[:, 0], targets[:, 1]

        # Calculate concordance index
        try:
            c_index = concordance_index_censored(
                events.numpy(), times.numpy(), predictions.numpy()
            )[0]
            return torch.tensor(c_index, dtype=torch.float32)
        except:
            return torch.tensor(0.5, dtype=torch.float32)  # Default to random

    def evaluate(self, X: pd.DataFrame, times: np.ndarray, events: np.ndarray,
                train_times: np.ndarray = None, train_events: np.ndarray = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of the ensemble model

        Returns:
            Dictionary with various evaluation metrics
        """

        predictions = self.predict(X)

        metrics = {}

        # Basic C-index
        if SKSURV_AVAILABLE:
            c_index = concordance_index_censored(events, times, predictions)[0]
            metrics['c_index'] = c_index

            # IPCW C-index if training data available
            if train_times is not None and train_events is not None:
                train_surv = Surv.from_arrays(event=train_events, time=train_times)
                test_surv = Surv.from_arrays(event=events, time=times)

                ipcw_c_index, _, _, _, _ = concordance_index_ipcw(
                    train_surv, test_surv, predictions, tau=7.0
                )
                metrics['ipcw_c_index'] = ipcw_c_index

        # Additional metrics
        metrics['mean_prediction'] = np.mean(predictions)
        metrics['std_prediction'] = np.std(predictions)
        metrics['prediction_range'] = np.ptp(predictions)

        return metrics

    def save_model(self, path: str = './models'):
        """Save trained model to disk"""

        Path(path).mkdir(exist_ok=True)

        model_dict = {
            'config': self.config,
            'models': {},
            'meta_model': self.meta_model,
            'feature_scaler': self.feature_scaler,
            'is_trained': self.is_trained
        }

        # Save individual models
        for name, model in self.models.items():
            if name == 'deep_survival' and model is not None:
                model.save(f'{path}/deep_survival_model')
            elif hasattr(model, 'save'):
                # For models with save method
                pass
            else:
                model_dict['models'][name] = model

        # Save dictionary
        with open(f'{path}/ensemble_model.pkl', 'wb') as f:
            pickle.dump(model_dict, f)

        print(f"✅ Model saved to {path}")

    def load_model(self, path: str = './models'):
        """Load trained model from disk"""

        with open(f'{path}/ensemble_model.pkl', 'rb') as f:
            model_dict = pickle.load(f)

        self.config = model_dict['config']
        self.models = model_dict['models']
        self.meta_model = model_dict['meta_model']
        self.feature_scaler = model_dict['feature_scaler']
        self.is_trained = model_dict['is_trained']

        # Load deep model if available
        if FASTAI_AVAILABLE:
            try:
                learn = load_learner(f'{path}/deep_survival_model.pkl')
                self.models['deep_survival'] = learn
            except:
                print("Warning: Could not load deep survival model")

        print(f"✅ Model loaded from {path}")


def create_survival_ensemble(config: ModelConfig = None) -> LeukemiaSurvivalPredictor:
    """
    Convenience function to create a configured survival ensemble

    Args:
        config: Model configuration

    Returns:
        Configured LeukemiaSurvivalPredictor instance
    """

    return LeukemiaSurvivalPredictor(config)


# Example usage and testing
if __name__ == "__main__":

    print("Testing Leukemia Survival Predictor...")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # Generate synthetic features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate synthetic survival data
    times = np.random.exponential(2, n_samples)  # Survival times
    events = np.random.binomial(1, 0.7, n_samples)  # 70% uncensored

    print(f"Created synthetic dataset: {n_samples} samples, {n_features} features")
    print(".1f")

    # Create and train model
    config = ModelConfig(
        learning_rate=1e-3,
        epochs=10,  # Reduced for testing
        hidden_layers=[64, 32],
        batch_size=32
    )

    predictor = LeukemiaSurvivalPredictor(config)

    # Split data for validation
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    times_train, times_val = times[:train_size], times[train_size:]
    events_train, events_val = events[:train_size], events[train_size:]

    # Train model
    predictor.fit(
        X_train, times_train, events_train,
        validation_data=(X_val, times_val, events_val)
    )

    # Evaluate model
    metrics = predictor.evaluate(X_val, times_val, events_val, times_train, events_train)

    print("\nModel Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n✅ Survival model testing completed!")
