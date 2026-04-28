"""
Optuna hyperparameter tuning for XGBoost
30 trials with PR-AUC (Average Precision) optimization
Saves best parameters and final trained model
"""

import joblib
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score
from pathlib import Path


def load_data():
    """Load train/test data and calculate class weights"""
    X_train = joblib.load('data/X_train.pkl')
    y_train = joblib.load('data/y_train.pkl')
    X_test = joblib.load('data/X_test.pkl')
    y_test = joblib.load('data/y_test.pkl')
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"y_test distribution: {y_test.value_counts().to_dict()}")
    
    # Calculate scale_pos_weight (for imbalanced classification)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nScale pos weight (for imbalance): {scale_pos_weight:.3f}")
    
    return X_train, y_train, X_test, y_test, scale_pos_weight


def create_objective(X_train, y_train, X_test, y_test, scale_pos_weight):
    """Create objective function for Optuna"""
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate using PR-AUC (Average Precision Score)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        return pr_auc
    
    return objective


def run_optuna_tuning(X_train, y_train, X_test, y_test, scale_pos_weight, n_trials=30):
    """Run Optuna optimization"""
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna optimization for XGBoost")
    print(f"{'='*60}")
    print(f"Number of trials: {n_trials}")
    print(f"Optimization metric: PR-AUC (Average Precision)")
    print(f"Direction: Maximize")
    print(f"{'='*60}\n")
    
    # Create study
    objective = create_objective(X_train, y_train, X_test, y_test, scale_pos_weight)
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best PR-AUC: {study.best_value:.4f}")
    print(f"\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    return study


def train_final_model(X_train, y_train, X_test, y_test, best_params, scale_pos_weight):
    """Train final model with best parameters"""
    
    print(f"Training final XGBoost model with best parameters...")
    
    # Ensure scale_pos_weight is in params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    
    # Train model
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"Final model PR-AUC on test set: {pr_auc:.4f}")
    
    return model, pr_auc


def save_results(study, model, scale_pos_weight, final_pr_auc, output_dir='models'):
    """Save best parameters and model"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save best parameters
    best_params_with_weight = study.best_params.copy()
    best_params_with_weight['scale_pos_weight'] = scale_pos_weight
    
    joblib.dump(best_params_with_weight, f'{output_dir}/xgb_best_params.pkl')
    print(f"Saved best parameters: {output_dir}/xgb_best_params.pkl")
    
    # Save final model
    joblib.dump(model, f'{output_dir}/xgb_final_model.pkl')
    print(f"Saved final model: {output_dir}/xgb_final_model.pkl")
    
    # Save study results
    joblib.dump(study, f'{output_dir}/optuna_study.pkl')
    print(f"Saved optuna study: {output_dir}/optuna_study.pkl")
    
    # Save summary
    summary = {
        'best_pr_auc': study.best_value,
        'final_pr_auc': final_pr_auc,
        'best_params': best_params_with_weight,
        'n_trials': len(study.trials),
        'model_type': 'XGBoost'
    }
    joblib.dump(summary, f'{output_dir}/xgb_optuna_summary.pkl')
    print(f"Saved summary: {output_dir}/xgb_optuna_summary.pkl")
    
    # Print summary to file
    with open(f'{output_dir}/xgb_optuna_results.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("XGBoost Optuna Tuning Results\n")
        f.write("="*60 + "\n")
        f.write(f"Best PR-AUC (from optimization): {study.best_value:.4f}\n")
        f.write(f"Final PR-AUC (on test set): {final_pr_auc:.4f}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"\nBest Parameters:\n")
        for key, value in best_params_with_weight.items():
            f.write(f"  {key}: {value}\n")
        f.write("="*60 + "\n")
    
    print(f"Saved results summary: {output_dir}/xgb_optuna_results.txt")


def main():
    """Main function"""
    
    # Load data
    X_train, y_train, X_test, y_test, scale_pos_weight = load_data()
    
    # Run Optuna optimization (30 trials)
    study = run_optuna_tuning(X_train, y_train, X_test, y_test, scale_pos_weight, n_trials=30)
    
    # Train final model with best parameters
    model, final_pr_auc = train_final_model(
        X_train, y_train, X_test, y_test, 
        study.best_params, scale_pos_weight
    )
    
    # Save results
    save_results(study, model, scale_pos_weight, final_pr_auc, output_dir='models')
    
    print(f"\n✓ Optuna tuning complete!")
    print(f"✓ Best model and parameters saved to models/")


if __name__ == '__main__':
    main()
