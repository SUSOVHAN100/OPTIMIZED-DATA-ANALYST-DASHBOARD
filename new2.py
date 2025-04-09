import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from tkinter import scrolledtext
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import shap
from pandastable import Table

# Modern color scheme
BACKGROUND_COLOR = "#263238"
PRIMARY_COLOR = "#37474F"
SECONDARY_COLOR = "#455A64"
ACCENT_COLOR = "#7E57C2"
HOVER_COLOR = "#5E35B1"
TEXT_COLOR = "#FFFFFF"
DANGER_COLOR = "#D32F2F"
SUCCESS_COLOR = "#388E3C"
INFO_COLOR = "#1976D2"


def load_file():
    """Function to load the file."""
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
    if file_path:
        try:
            global df, original_df
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            original_df = df.copy()  # Keep a copy of original data
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            update_data_preview()
            enable_buttons()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

def enable_buttons():
    """Enable all analysis buttons after data is loaded."""
    prepare_data_btn.config(state=tk.NORMAL)
    visualize_btn.config(state=tk.NORMAL)
    analyze_btn.config(state=tk.NORMAL)
    corr_matrix_btn.config(state=tk.NORMAL)
    boosting_btn.config(state=tk.NORMAL)
    advanced_btn.config(state=tk.NORMAL)

def show_regression_options():
    """Show regression options popup."""
    popup = tk.Toplevel(root)
    popup.title("Regression Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Regression Type", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    linear_btn = tk.Button(popup, text="Linear Regression", 
                         command=lambda: [popup.destroy(), show_linear_regression_interface()], 
                         bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                         relief=tk.FLAT, padx=30, pady=10)
    linear_btn.pack(pady=10)
    add_hover_effect(linear_btn)

    logistic_btn = tk.Button(popup, text="Logistic Regression", 
                           command=lambda: [popup.destroy(), show_data_type_popup("Classification")], 
                           bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=10)
    logistic_btn.pack(pady=10)
    add_hover_effect(logistic_btn)

def show_data_type_popup(analysis_type):
    """Show data type selection popup."""
    popup = tk.Toplevel(root)
    popup.title("Data Type Selection")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text=f"Select Data Type for {analysis_type}", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    if analysis_type == "Classification":
        discrete_btn = tk.Button(popup, text="Discrete Data", 
                               command=lambda: [popup.destroy(), show_classification_interface()], 
                               bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                               relief=tk.FLAT, padx=30, pady=10)
        discrete_btn.pack(pady=10)
        add_hover_effect(discrete_btn)
    else:
        continuous_btn = tk.Button(popup, text="Continuous Data", 
                                 command=lambda: [popup.destroy(), show_linear_regression_interface()], 
                                 bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                                 relief=tk.FLAT, padx=30, pady=10)
        continuous_btn.pack(pady=10)
        add_hover_effect(continuous_btn)

def add_hover_effect(button):
    """Function to add hover effect to a button."""
    def on_enter(e):
        button.config(bg=HOVER_COLOR)

    def on_leave(e):
        button.config(bg=ACCENT_COLOR)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

def show_classification_interface():
    """Display interface for classification analysis."""
    try:
        popup = tk.Toplevel(root)
        popup.title("Classification Analysis")
        popup.geometry("1000x800")
        popup.configure(bg=BACKGROUND_COLOR)

        # Main container frame
        main_frame = tk.Frame(popup, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(main_frame, text="Classification Analysis Results", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))

        global X, y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Generate metrics
        report = classification_report(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Results frame
        results_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Text results
        result_text = tk.Text(results_frame, wrap=tk.WORD, height=12, bg=PRIMARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10))
        result_text.insert(tk.END, f"Classification Report:\n{report}\n")
        result_text.insert(tk.END, f"\nAccuracy: {acc:.2f}\n")
        result_text.config(state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix", fontsize=12)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")

        canvas = FigureCanvasTkAgg(fig, master=results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(20, 0))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {e}")

def show_linear_regression_interface():
    """Display interface for linear regression analysis."""
    try:
        popup = tk.Toplevel(root)
        popup.title("Regression Analysis")
        popup.geometry("1100x900")
        popup.configure(bg=BACKGROUND_COLOR)

        # Main container frame
        main_frame = tk.Frame(popup, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(main_frame, text="Regression Analysis Results", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))

        global X, y
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coefficients = model.coef_
        intercept = model.intercept_

        # Results frame
        results_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Text results
        result_text = tk.Text(results_frame, wrap=tk.WORD, height=15, bg=PRIMARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10))
        result_text.insert(tk.END, "Regression Model Summary:\n\n")
        result_text.insert(tk.END, f"Formula: y = {intercept:.2f} ")
        for i, coef in enumerate(coefficients, 1):
            result_text.insert(tk.END, f"+ {coef:.2f}x{i} ")
        result_text.insert(tk.END, "\n\n")
        result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.4f}\n")
        result_text.insert(tk.END, f"R² Score: {r2:.4f}\n\n")
        result_text.insert(tk.END, "Feature Coefficients:\n")
        for i, coef in enumerate(coefficients, 1):
            result_text.insert(tk.END, f"  x{i}: {coef:.4f}\n")
        result_text.config(state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        # Plot frame
        plot_frame = tk.Frame(results_frame, bg=BACKGROUND_COLOR)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Plotting True vs Predicted values
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                color='red', linestyle='--', linewidth=2)
        ax.set_title("True vs Predicted Values", fontsize=12)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True, linestyle='--', alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=popup.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(20, 0))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Error during analysis: {e}")

def update_data_preview():
    """Update the data preview text widget."""
    data_preview.config(state=tk.NORMAL)
    data_preview.delete(1.0, tk.END)
    
    if df.empty:
        data_preview.insert(tk.END, "No data loaded")
    else:
        data_preview.insert(tk.END, f"Dataset Preview (First 5 rows):\n\n")
        data_preview.insert(tk.END, df.head().to_string())
        data_preview.insert(tk.END, "\n\nDataset Information:\n\n")
        
        # Capture df.info() output
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        data_preview.insert(tk.END, info_str)
        
        # Add null value counts
        data_preview.insert(tk.END, "\n\nNull Value Counts:\n")
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                data_preview.insert(tk.END, f"{col}: {count} null values\n")
    
    data_preview.config(state=tk.DISABLED)

def show_correlation_matrix():
    """Display correlation matrix in a separate window."""
    if df.empty:
        messagebox.showerror("Error", "No data loaded!")
        return
    
    try:
        corr_window = tk.Toplevel(root)
        corr_window.title("Correlation Matrix")
        corr_window.geometry("1000x800")
        corr_window.configure(bg=BACKGROUND_COLOR)
        
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showerror("Error", "No numeric columns to calculate correlation!")
            corr_window.destroy()
            return
            
        corr_matrix = numeric_df.corr()
        
        # Create figure with larger size
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                   annot_kws={"size": 10}, fmt=".2f", linewidths=.5)
        ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=corr_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Close button
        close_btn = tk.Button(corr_window, text="Close", command=corr_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=10)
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create correlation matrix: {e}")

def show_regression_options():
    """Show regression options popup."""
    popup = tk.Toplevel(root)
    popup.title("Regression Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Regression Type", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    linear_btn = tk.Button(popup, text="Linear Regression", 
                         command=lambda: [popup.destroy(), show_linear_regression_interface()], 
                         bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                         relief=tk.FLAT, padx=30, pady=10)
    linear_btn.pack(pady=10)
    add_hover_effect(linear_btn)

    logistic_btn = tk.Button(popup, text="Logistic Regression", 
                           command=lambda: [popup.destroy(), show_data_type_popup("Classification")], 
                           bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=10)
    logistic_btn.pack(pady=10)
    add_hover_effect(logistic_btn)

# ... [previous functions remain the same until show_regression_options] ...

def show_boosting_options():
    """Show boosting algorithm options popup."""
    popup = tk.Toplevel(root)
    popup.title("Boosting Algorithm Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Boosting Algorithm", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    xgb_btn = tk.Button(popup, text="XGBoost", 
                       command=lambda: [popup.destroy(), run_boosting_algorithm("XGBoost")], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=10)
    xgb_btn.pack(pady=10)
    add_hover_effect(xgb_btn)

    cat_btn = tk.Button(popup, text="CatBoost", 
                       command=lambda: [popup.destroy(), run_boosting_algorithm("CatBoost")], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=10)
    cat_btn.pack(pady=10)
    add_hover_effect(cat_btn)

def run_boosting_algorithm(algorithm):
    """Run the selected boosting algorithm and display results."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if algorithm == "XGBoost":
            if problem_type == "classification":
                model = XGBClassifier(random_state=42, eval_metric='mlogloss')
            else:
                model = XGBRegressor(random_state=42)
        else:  # CatBoost
            if problem_type == "classification":
                model = CatBoostClassifier(random_state=42, silent=True)
            else:
                model = CatBoostRegressor(random_state=42, silent=True)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Create results window
        results_window = tk.Toplevel(root)
        results_window.title(f"{algorithm} Results")
        results_window.geometry("1000x800")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Main container
        main_frame = tk.Frame(results_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text=f"{algorithm} Results ({problem_type.capitalize()})", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text results
        result_text = tk.Text(results_frame, wrap=tk.WORD, height=12, bg=PRIMARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10))
        
        if problem_type == "classification":
            report = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            result_text.insert(tk.END, f"Classification Report:\n{report}\n")
            result_text.insert(tk.END, f"\nAccuracy: {acc:.2f}\n")
            
            # Confusion matrix plot
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title("Confusion Matrix", fontsize=12)
            ax1.set_xlabel("Predicted Labels")
            ax1.set_ylabel("True Labels")
            
            canvas1 = FigureCanvasTkAgg(fig1, master=results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result_text.insert(tk.END, "Regression Metrics:\n\n")
            result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.4f}\n")
            result_text.insert(tk.END, f"R² Score: {r2:.4f}\n")
            
            # True vs Predicted plot
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
            ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                    color='red', linestyle='--', linewidth=2)
            ax1.set_title("True vs Predicted Values", fontsize=12)
            ax1.set_xlabel("True Values")
            ax1.set_ylabel("Predicted Values")
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        result_text.config(state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Feature importance plot
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        if algorithm == "XGBoost":
            importances = model.feature_importances_
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
        else:  # CatBoost
            importances = model.get_feature_importance()
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
        
        ax2.set_title("Feature Importance", fontsize=12)
        ax2.set_xlabel("Importance Score")
        
        # Create a new frame for the second plot
        plot_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(20, 0))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during {algorithm} analysis: {e}")

def show_advanced_analysis_options():
    """Show advanced analysis options popup."""
    popup = tk.Toplevel(root)
    popup.title("Advanced Analysis Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Advanced Analysis", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    ensemble_btn = tk.Button(popup, text="Ensemble Methods", 
                           command=lambda: [popup.destroy(), show_ensemble_options()], 
                           bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=10)
    ensemble_btn.pack(pady=10)
    add_hover_effect(ensemble_btn)

    pca_btn = tk.Button(popup, text="PCA Analysis", 
                       command=lambda: [popup.destroy(), run_pca_analysis()], 
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=10)
    pca_btn.pack(pady=10)
    add_hover_effect(pca_btn)

    shap_btn = tk.Button(popup, text="SHAP Analysis", 
                        command=lambda: [popup.destroy(), show_shap_options()], 
                        bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                        relief=tk.FLAT, padx=30, pady=10)
    shap_btn.pack(pady=10)
    add_hover_effect(shap_btn)

def show_ensemble_options():
    """Show ensemble method options popup."""
    popup = tk.Toplevel(root)
    popup.title("Ensemble Method Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Ensemble Method", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    voting_btn = tk.Button(popup, text="Voting Ensemble", 
                          command=lambda: [popup.destroy(), run_ensemble_analysis("Voting")], 
                          bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                          relief=tk.FLAT, padx=30, pady=10)
    voting_btn.pack(pady=10)
    add_hover_effect(voting_btn)

    rf_btn = tk.Button(popup, text="Random Forest", 
                      command=lambda: [popup.destroy(), run_ensemble_analysis("RandomForest")], 
                      bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                      relief=tk.FLAT, padx=30, pady=10)
    rf_btn.pack(pady=10)
    add_hover_effect(rf_btn)

def run_ensemble_analysis(method):
    """Run the selected ensemble method and display results."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if method == "Voting":
            if problem_type == "classification":
                model = VotingClassifier(estimators=[
                    ('lr', LogisticRegression(max_iter=1000)),
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss'))
                ], voting='soft')
            else:
                model = VotingRegressor(estimators=[
                    ('lr', LinearRegression()),
                    ('rf', RandomForestRegressor(random_state=42)),
                    ('xgb', XGBRegressor(random_state=42))
                ])
        else:  # RandomForest
            if problem_type == "classification":
                model = RandomForestClassifier(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Create results window
        results_window = tk.Toplevel(root)
        results_window.title(f"{method} Ensemble Results")
        results_window.geometry("1000x800")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Main container
        main_frame = tk.Frame(results_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text=f"{method} Ensemble Results ({problem_type.capitalize()})", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text results
        result_text = tk.Text(results_frame, wrap=tk.WORD, height=12, bg=PRIMARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10))
        
        if problem_type == "classification":
            report = classification_report(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            result_text.insert(tk.END, f"Classification Report:\n{report}\n")
            result_text.insert(tk.END, f"\nAccuracy: {acc:.2f}\n")
            
            # Confusion matrix plot
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title("Confusion Matrix", fontsize=12)
            ax1.set_xlabel("Predicted Labels")
            ax1.set_ylabel("True Labels")
            
            canvas1 = FigureCanvasTkAgg(fig1, master=results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result_text.insert(tk.END, "Regression Metrics:\n\n")
            result_text.insert(tk.END, f"Mean Squared Error (MSE): {mse:.4f}\n")
            result_text.insert(tk.END, f"R² Score: {r2:.4f}\n")
            
            # True vs Predicted plot
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            ax1.scatter(y_test, y_pred, color=ACCENT_COLOR, edgecolor='k', alpha=0.7)
            ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
                    color='red', linestyle='--', linewidth=2)
            ax1.set_title("True vs Predicted Values", fontsize=12)
            ax1.set_xlabel("True Values")
            ax1.set_ylabel("Predicted Values")
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            canvas1 = FigureCanvasTkAgg(fig1, master=results_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        result_text.config(state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Feature importance plot (only for Random Forest)
        if method == "RandomForest":
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            importances = model.feature_importances_
            features = X.columns
            sorted_idx = np.argsort(importances)
            ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color=ACCENT_COLOR)
            ax2.set_yticks(range(len(sorted_idx)))
            ax2.set_yticklabels([features[i] for i in sorted_idx])
            ax2.set_title("Feature Importance", fontsize=12)
            ax2.set_xlabel("Importance Score")
            
            # Create a new frame for the second plot
            plot_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
            
            canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(20, 0))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during {method} ensemble analysis: {e}")

def run_pca_analysis():
    """Run PCA analysis and display results."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showerror("Error", "No numeric columns found for PCA!")
            return
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Run PCA
        pca = PCA()
        pca.fit(scaled_data)
        
        # Create results window
        results_window = tk.Toplevel(root)
        results_window.title("PCA Analysis Results")
        results_window.geometry("1000x800")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Main container
        main_frame = tk.Frame(results_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Principal Component Analysis (PCA)", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text results
        result_text = tk.Text(results_frame, wrap=tk.WORD, height=12, bg=PRIMARY_COLOR, 
                            fg=TEXT_COLOR, insertbackground=TEXT_COLOR, font=("Courier", 10))
        
        result_text.insert(tk.END, "PCA Explained Variance Ratio:\n")
        for i, ratio in enumerate(pca.explained_variance_ratio_, 1):
            result_text.insert(tk.END, f"PC{i}: {ratio:.4f}\n")
        
        result_text.insert(tk.END, f"\nTotal Explained Variance: {sum(pca.explained_variance_ratio_):.4f}\n")
        result_text.insert(tk.END, "\nPCA Components (First 5):\n")
        components_df = pd.DataFrame(pca.components_, columns=numeric_df.columns)
        result_text.insert(tk.END, components_df.head().to_string())
        
        result_text.config(state=tk.DISABLED)
        result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Scree plot
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.plot(range(1, len(pca.explained_variance_ratio_)+1), 
                pca.explained_variance_ratio_, 'o-', color=ACCENT_COLOR)
        ax1.set_title("Scree Plot", fontsize=12)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        canvas1 = FigureCanvasTkAgg(fig1, master=results_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create a new frame for the second plot
        plot_frame = tk.Frame(main_frame, bg=BACKGROUND_COLOR)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # PCA biplot (first two components)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        transformed_data = pca.transform(scaled_data)
        
        # Scatter plot of first two components
        ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], 
                   color=ACCENT_COLOR, alpha=0.5)
        ax2.set_title("PCA Biplot (First Two Components)", fontsize=12)
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        
        # Add feature vectors
        for i, feature in enumerate(numeric_df.columns):
            ax2.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                     color='r', alpha=0.5)
            ax2.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, 
                    feature, color='r', ha='center', va='center')
        
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=plot_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(20, 0))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during PCA analysis: {e}")

def show_shap_options():
    """Show SHAP analysis options popup."""
    popup = tk.Toplevel(root)
    popup.title("SHAP Analysis Options")
    popup.geometry("400x300")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select SHAP Visualization", 
                    font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 30))

    summary_btn = tk.Button(popup, text="Summary Plot", 
                          command=lambda: [popup.destroy(), run_shap_analysis("summary")], 
                          bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                          relief=tk.FLAT, padx=30, pady=10)
    summary_btn.pack(pady=10)
    add_hover_effect(summary_btn)

    dependence_btn = tk.Button(popup, text="Dependence Plot", 
                             command=lambda: [popup.destroy(), run_shap_analysis("dependence")], 
                             bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                             relief=tk.FLAT, padx=30, pady=10)
    dependence_btn.pack(pady=10)
    add_hover_effect(dependence_btn)

def run_shap_analysis(plot_type):
    """Run SHAP analysis and display the selected plot."""
    try:
        if df.empty:
            messagebox.showerror("Error", "No data loaded!")
            return
        
        # Determine if classification or regression
        problem_type = "classification" if df.iloc[:, -1].nunique() < 10 else "regression"
        
        # Split data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model (using XGBoost for SHAP values)
        if problem_type == "classification":
            model = XGBClassifier(random_state=42, eval_metric='mlogloss')
        else:
            model = XGBRegressor(random_state=42)
        
        model.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        
        # Create results window
        results_window = tk.Toplevel(root)
        results_window.title(f"SHAP {plot_type.capitalize()} Plot")
        results_window.geometry("1000x800")
        results_window.configure(bg=BACKGROUND_COLOR)
        
        # Main container
        main_frame = tk.Frame(results_window, bg=BACKGROUND_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text=f"SHAP {plot_type.capitalize()} Plot", 
                             font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        title_label.pack(pady=(0, 20))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(PRIMARY_COLOR)
        ax.set_facecolor(PRIMARY_COLOR)
        
        # Generate the selected plot
        if plot_type == "summary":
            shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
            ax.set_title("SHAP Feature Importance", color=TEXT_COLOR, pad=20)
        elif plot_type == "dependence":
            # Use the feature with highest importance for dependence plot
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            top_feature_idx = np.argmax(feature_importance)
            top_feature = X.columns[top_feature_idx]
            shap.dependence_plot(top_feature, shap_values.values, X_test, 
                                interaction_index=None, ax=ax, show=False)
            ax.set_title(f"SHAP Dependence Plot for {top_feature}", color=TEXT_COLOR, pad=20)
        
        # Style the plot
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Close button
        close_btn = tk.Button(main_frame, text="Close", command=results_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(0, 20))
        add_hover_effect(close_btn)
        
    except Exception as e:
        messagebox.showerror("Error", f"Error during SHAP analysis: {e}")

# ... [rest of the previous functions remain the same] ...

# Main application window
root = tk.Tk()
root.title("Advanced ML Visualization Tool")
root.geometry("1200x800")
root.configure(bg=BACKGROUND_COLOR)

# Apply modern theme
style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background=BACKGROUND_COLOR)
style.configure('TLabel', background=BACKGROUND_COLOR, foreground=TEXT_COLOR)
style.configure('TButton', background=ACCENT_COLOR, foreground=TEXT_COLOR, 
               font=('Arial', 10, 'bold'), borderwidth=0)

# Main container frame
main_container = tk.Frame(root, bg=BACKGROUND_COLOR)
main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header frame
header_frame = tk.Frame(main_container, bg=BACKGROUND_COLOR)
header_frame.pack(fill=tk.X, pady=(0, 20))

# Title label
main_label = tk.Label(header_frame, 
                     text="Machine Learning Visualization Dashboard",
                     font=("Arial", 24, "bold"), 
                     fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
main_label.pack(pady=(0, 10))

# Subtitle
sub_label = tk.Label(header_frame, 
                    text="Explore, Analyze and Visualize Your Data",
                    font=("Arial", 12), 
                    fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
sub_label.pack()

# Separator
separator = ttk.Separator(header_frame, orient='horizontal')
separator.pack(fill=tk.X, pady=10)

# Content frame
content_frame = tk.Frame(main_container, bg=BACKGROUND_COLOR)
content_frame.pack(fill=tk.BOTH, expand=True)

# Left panel (buttons)
left_panel = tk.Frame(content_frame, bg=BACKGROUND_COLOR, width=200)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

# Load button
load_btn = tk.Button(left_panel, text="Load Dataset", command=load_file,
                    bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                    relief=tk.FLAT, padx=30, pady=15)
load_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(load_btn)

def setup_missing_values_tab(tab):
    """Setup the missing values tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select how to handle missing values for each column:", 
                         font=("Arial", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    info_label.pack(pady=(10, 20))

    # Scrollable frame for columns
    scroll_frame = tk.Frame(tab, bg=BACKGROUND_COLOR)
    scroll_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(scroll_frame, bg=BACKGROUND_COLOR, highlightthickness=0)
    scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg=BACKGROUND_COLOR)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Get columns with missing values
    null_counts = df.isnull().sum()
    cols_with_missing = null_counts[null_counts > 0].index.tolist()
    
    if not cols_with_missing:
        no_missing_label = tk.Label(scrollable_frame, text="No missing values found in the dataset!", 
                                  font=("Arial", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        no_missing_label.pack(pady=20)
        return

    # Create a frame for each column with missing values
    column_frames = []
    for col in cols_with_missing:
        col_frame = tk.Frame(scrollable_frame, bg=BACKGROUND_COLOR, pady=10)
        column_frames.append(col_frame)
        col_frame.pack(fill=tk.X, padx=10, pady=5)

        # Column name and missing count
        col_label = tk.Label(col_frame, text=f"{col} ({null_counts[col]} missing values)", 
                            font=("Arial", 11, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
        col_label.pack(anchor=tk.W)

        # Radio buttons for handling options
        option_var = tk.StringVar(value="drop")
        
        drop_radio = tk.Radiobutton(col_frame, text="Drop rows with missing values", 
                                   variable=option_var, value="drop", 
                                   bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=PRIMARY_COLOR,
                                   activebackground=BACKGROUND_COLOR, activeforeground=TEXT_COLOR)
        drop_radio.pack(anchor=tk.W)

        mean_radio = tk.Radiobutton(col_frame, text="Fill with mean value", 
                                   variable=option_var, value="mean", 
                                   bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=PRIMARY_COLOR,
                                   activebackground=BACKGROUND_COLOR, activeforeground=TEXT_COLOR)
        mean_radio.pack(anchor=tk.W)

        median_radio = tk.Radiobutton(col_frame, text="Fill with median value", 
                                     variable=option_var, value="median", 
                                     bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=PRIMARY_COLOR,
                                     activebackground=BACKGROUND_COLOR, activeforeground=TEXT_COLOR)
        median_radio.pack(anchor=tk.W)

        mode_radio = tk.Radiobutton(col_frame, text="Fill with mode value", 
                                   variable=option_var, value="mode", 
                                   bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=PRIMARY_COLOR,
                                   activebackground=BACKGROUND_COLOR, activeforeground=TEXT_COLOR)
        mode_radio.pack(anchor=tk.W)

        zero_radio = tk.Radiobutton(col_frame, text="Fill with 0", 
                                   variable=option_var, value="zero", 
                                   bg=BACKGROUND_COLOR, fg=TEXT_COLOR, selectcolor=PRIMARY_COLOR,
                                   activebackground=BACKGROUND_COLOR, activeforeground=TEXT_COLOR)
        zero_radio.pack(anchor=tk.W)

        # Store the column and its option variable
        col_frame.column_name = col
        col_frame.option_var = option_var

    # Apply button
    apply_btn = tk.Button(tab, text="Apply Changes", command=lambda: apply_missing_value_changes(column_frames),
                         bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=5)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

def apply_missing_value_changes(column_frames):
    """Apply the selected missing value handling methods."""
    global df
    
    try:
        for col_frame in column_frames:
            col = col_frame.column_name
            option = col_frame.option_var.get()
            
            if option == "drop":
                df = df.dropna(subset=[col])
            elif option == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif option == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif option == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif option == "zero":
                df[col].fillna(0, inplace=True)
        
        messagebox.showinfo("Success", "Missing value handling applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to handle missing values: {e}")

def move_items(source_list, dest_list):
    """Move selected items from source list to destination list."""
    selected = source_list.curselection()
    for idx in selected[::-1]:  # Reverse to maintain order when deleting
        item = source_list.get(idx)
        dest_list.insert(tk.END, item)
        source_list.delete(idx)

def remove_items(*lists):
    """Remove selected items from lists and return them to available features."""
    for lst in lists:
        selected = lst.curselection()
        for idx in selected[::-1]:  # Reverse to maintain order when deleting
            item = lst.get(idx)
            available_list = root.nametowidget(lst.master.master.master.children['!frame'].children['!listbox'])
            available_list.insert(tk.END, item)
            lst.delete(idx)

def apply_feature_selection(features_list, target_list):
    """Apply the selected feature and target variables."""
    global df, X, y
    
    try:
        # Get selected features and target
        features = [features_list.get(i) for i in range(features_list.size())]
        target = target_list.get(0) if target_list.size() > 0 else None
        
        if not features or not target:
            messagebox.showerror("Error", "Please select at least one feature and one target variable")
            return
        
        # Update the dataframe to only include selected columns
        df = df[features + [target]]
        
        # Set X and y
        X = df[features]
        y = df[target]
        
        messagebox.showinfo("Success", "Feature selection applied successfully!")
        update_data_preview()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply feature selection: {e}")

def setup_feature_selection_tab(tab):
    """Setup the feature selection tab."""
    tab.configure(style='TFrame')
    
    # Info label
    info_label = tk.Label(tab, text="Select features and target variable:", 
                         font=("Arial", 12), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    info_label.pack(pady=(10, 20))

    # Main frame for feature selection
    selection_frame = tk.Frame(tab, bg=BACKGROUND_COLOR)
    selection_frame.pack(fill=tk.BOTH, expand=True)

    # Available features frame
    available_frame = tk.Frame(selection_frame, bg=BACKGROUND_COLOR)
    available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

    available_label = tk.Label(available_frame, text="Available Features", 
                             font=("Arial", 11, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    available_label.pack()

    # Listbox for available features
    available_list = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, 
                              bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                              selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR)
    available_list.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Populate list with columns
    for col in df.columns:
        available_list.insert(tk.END, col)

    # Buttons frame
    button_frame = tk.Frame(selection_frame, bg=BACKGROUND_COLOR)
    button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

    # Add to features button
    add_feature_btn = tk.Button(button_frame, text="→ Features →", 
                              command=lambda: move_items(available_list, features_list),
                              bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                              relief=tk.FLAT, padx=10, pady=5)
    add_feature_btn.pack(pady=10)
    add_hover_effect(add_feature_btn)

    # Add to target button
    add_target_btn = tk.Button(button_frame, text="→ Target →", 
                             command=lambda: move_items(available_list, target_list),
                             bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                             relief=tk.FLAT, padx=10, pady=5)
    add_target_btn.pack(pady=10)
    add_hover_effect(add_target_btn)

    # Remove button
    remove_btn = tk.Button(button_frame, text="← Remove ←", 
                          command=lambda: remove_items(features_list, target_list, available_list),
                          bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                          relief=tk.FLAT, padx=10, pady=5)
    remove_btn.pack(pady=10)
    add_hover_effect(remove_btn)

    # Selected features frame
    selected_frame = tk.Frame(selection_frame, bg=BACKGROUND_COLOR)
    selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Features list frame
    features_frame = tk.Frame(selected_frame, bg=BACKGROUND_COLOR)
    features_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    features_label = tk.Label(features_frame, text="Input Features", 
                            font=("Arial", 11, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    features_label.pack()

    features_list = tk.Listbox(features_frame, selectmode=tk.MULTIPLE, 
                             bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                             selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR)
    features_list.pack(fill=tk.BOTH, expand=True)

    # Target frame
    target_frame = tk.Frame(selected_frame, bg=BACKGROUND_COLOR)
    target_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    target_label = tk.Label(target_frame, text="Target Variable", 
                          font=("Arial", 11, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    target_label.pack()

    target_list = tk.Listbox(target_frame, selectmode=tk.SINGLE, 
                           bg=PRIMARY_COLOR, fg=TEXT_COLOR, font=("Arial", 10),
                           selectbackground=ACCENT_COLOR, selectforeground=TEXT_COLOR)
    target_list.pack(fill=tk.BOTH, expand=True)

    # Apply button
    apply_btn = tk.Button(tab, text="Apply Feature Selection", 
                        command=lambda: apply_feature_selection(features_list, target_list),
                        bg=SUCCESS_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                        relief=tk.FLAT, padx=20, pady=5)
    apply_btn.pack(pady=(20, 10))
    add_hover_effect(apply_btn)

def show_prepare_data_interface():
    """Show interface for data preparation."""
    prep_window = tk.Toplevel(root)
    prep_window.title("Prepare Data")
    prep_window.geometry("1000x700")
    prep_window.configure(bg=BACKGROUND_COLOR)

    # Main container
    main_frame = tk.Frame(prep_window, bg=BACKGROUND_COLOR)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Title
    title_label = tk.Label(main_frame, text="Data Preparation", 
                         font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    title_label.pack(pady=(0, 20))

    # Tab control
    tab_control = ttk.Notebook(main_frame)
    tab_control.pack(fill=tk.BOTH, expand=True)

    # Tab 1: Handle Missing Values
    tab1 = ttk.Frame(tab_control)
    tab_control.add(tab1, text="Handle Missing Values")
    setup_missing_values_tab(tab1)

    # Tab 2: Feature Selection
    tab2 = ttk.Frame(tab_control)
    tab_control.add(tab2, text="Feature Selection")
    setup_feature_selection_tab(tab2)

    # Close button
    close_btn = tk.Button(main_frame, text="Close", command=prep_window.destroy,
                         bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=5)
    close_btn.pack(pady=(20, 0))
    add_hover_effect(close_btn)
# Prepare Data button
prepare_data_btn = tk.Button(left_panel, text="Prepare Data", command=show_prepare_data_interface,
                           bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                           relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
prepare_data_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(prepare_data_btn)

def plot_graph(graph_type):
    """Function to plot selected graph."""
    if df.empty:
        messagebox.showerror("Error", "Dataset is empty. Please load a valid dataset.")
        return

    try:
        # Create a new window for the plot
        plot_window = tk.Toplevel(root)
        plot_window.title(f"{graph_type} Visualization")
        plot_window.geometry("800x600")
        plot_window.configure(bg=BACKGROUND_COLOR)

        # Create figure with larger size
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(PRIMARY_COLOR)
        ax.set_facecolor(PRIMARY_COLOR)

        if graph_type == "Pie Chart":
            counts = df.iloc[:, -1].value_counts()
            colors = [ACCENT_COLOR, HOVER_COLOR, "#B39DDB", "#D1C4E9"]
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                  colors=colors[:len(counts)], textprops={'color': TEXT_COLOR})
            ax.set_title("Class Distribution", color=TEXT_COLOR, pad=20)

        elif graph_type == "Bar Chart":
            counts = df.iloc[:, -1].value_counts()
            ax.bar(counts.index, counts.values, color=ACCENT_COLOR)
            ax.set_title("Class Frequency", color=TEXT_COLOR)
            ax.set_xlabel("Classes", color=TEXT_COLOR)
            ax.set_ylabel("Frequency", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "Scatter Plot":
            if X.shape[1] < 2:
                messagebox.showerror("Error", "Scatter plot requires at least two features.")
                plot_window.destroy()
                return
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
            ax.set_title("Feature Relationship", color=TEXT_COLOR)
            ax.set_xlabel("Feature 1", color=TEXT_COLOR)
            ax.set_ylabel("Feature 2", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            fig.colorbar(scatter, ax=ax).set_label("Classes", color=TEXT_COLOR)

        elif graph_type == "Histogram":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for histogram.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            
            for column in numeric_df.columns:
                sns.histplot(data=numeric_df, x=column, kde=False, 
                            color=ACCENT_COLOR, alpha=0.5, label=column, ax=ax)
            
            ax.set_title("Feature Distributions", color=TEXT_COLOR)
            ax.set_xlabel("Value", color=TEXT_COLOR)
            ax.set_ylabel("Frequency", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "KDE Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for KDE plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            
            for column in numeric_df.columns:
                sns.kdeplot(data=numeric_df, x=column, 
                           color=ACCENT_COLOR, alpha=0.5, label=column, ax=ax)
            
            ax.set_title("Density Estimation", color=TEXT_COLOR)
            ax.set_xlabel("Value", color=TEXT_COLOR)
            ax.set_ylabel("Density", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)

        elif graph_type == "Line Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for line plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            
            for column in numeric_df.columns:
                sns.lineplot(data=numeric_df, x=numeric_df.index, y=column, 
                           color=ACCENT_COLOR, alpha=0.7, label=column, ax=ax)
            
            ax.set_title("Line Plot", color=TEXT_COLOR)
            ax.set_xlabel("Index", color=TEXT_COLOR)
            ax.set_ylabel("Value", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

        elif graph_type == "Area Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for area plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            
            numeric_df.plot.area(ax=ax, alpha=0.7, color=[ACCENT_COLOR, HOVER_COLOR, "#B39DDB"])
            ax.set_title("Area Plot", color=TEXT_COLOR)
            ax.set_xlabel("Index", color=TEXT_COLOR)
            ax.set_ylabel("Value", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

        elif graph_type == "Box Plot":
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                messagebox.showerror("Error", "No numeric columns found for box plot.")
                plot_window.destroy()
                return
                
            fig.clf()
            ax = fig.subplots(1, 1)
            
            numeric_df.plot.box(ax=ax, patch_artist=True, 
                              boxprops=dict(facecolor=ACCENT_COLOR, color=TEXT_COLOR),
                              whiskerprops=dict(color=TEXT_COLOR),
                              capprops=dict(color=TEXT_COLOR),
                              medianprops=dict(color='red'))
            ax.set_title("Box Plot", color=TEXT_COLOR)
            ax.set_ylabel("Value", color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR)
            ax.grid(True, linestyle='--', alpha=0.3)

        # Style the plot
        for spine in ax.spines.values():
            spine.set_color(TEXT_COLOR)
        
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Close button
        close_btn = tk.Button(plot_window, text="Close", command=plot_window.destroy,
                            bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                            relief=tk.FLAT, padx=20, pady=5)
        close_btn.pack(pady=(0, 20))
        add_hover_effect(close_btn)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot graph: {e}")

def show_visualization_options():
    """Show visualization options popup."""
    popup = tk.Toplevel(root)
    popup.title("Visualization Options")
    popup.geometry("600x500")
    popup.configure(bg=BACKGROUND_COLOR)
    popup.resizable(False, False)

    label = tk.Label(popup, text="Select Visualization Type", 
                    font=("Arial", 16, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
    label.pack(pady=(20, 20))

    # Create a frame for the buttons
    btn_frame = tk.Frame(popup, bg=BACKGROUND_COLOR)
    btn_frame.pack(pady=10)

    # First row of buttons
    row1_frame = tk.Frame(btn_frame, bg=BACKGROUND_COLOR)
    row1_frame.pack(pady=5)

    pie_btn = tk.Button(row1_frame, text="Pie Chart", command=lambda: [popup.destroy(), plot_graph("Pie Chart")],
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                       relief=tk.FLAT, padx=20, pady=10, width=15)
    pie_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(pie_btn)

    bar_btn = tk.Button(row1_frame, text="Bar Chart", command=lambda: [popup.destroy(), plot_graph("Bar Chart")],
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                       relief=tk.FLAT, padx=20, pady=10, width=15)
    bar_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(bar_btn)

    # Second row of buttons
    row2_frame = tk.Frame(btn_frame, bg=BACKGROUND_COLOR)
    row2_frame.pack(pady=5)

    scatter_btn = tk.Button(row2_frame, text="Scatter Plot", command=lambda: [popup.destroy(), plot_graph("Scatter Plot")],
                          bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                          relief=tk.FLAT, padx=20, pady=10, width=15)
    scatter_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(scatter_btn)

    hist_btn = tk.Button(row2_frame, text="Histogram", command=lambda: [popup.destroy(), plot_graph("Histogram")],
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                       relief=tk.FLAT, padx=20, pady=10, width=15)
    hist_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(hist_btn)

    # Third row of buttons
    row3_frame = tk.Frame(btn_frame, bg=BACKGROUND_COLOR)
    row3_frame.pack(pady=5)

    kde_btn = tk.Button(row3_frame, text="KDE Plot", command=lambda: [popup.destroy(), plot_graph("KDE Plot")],
                      bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                      relief=tk.FLAT, padx=20, pady=10, width=15)
    kde_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(kde_btn)

    line_btn = tk.Button(row3_frame, text="Line Plot", command=lambda: [popup.destroy(), plot_graph("Line Plot")],
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                       relief=tk.FLAT, padx=20, pady=10, width=15)
    line_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(line_btn)

    # Fourth row of buttons
    row4_frame = tk.Frame(btn_frame, bg=BACKGROUND_COLOR)
    row4_frame.pack(pady=5)

    area_btn = tk.Button(row4_frame, text="Area Plot", command=lambda: [popup.destroy(), plot_graph("Area Plot")],
                       bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                       relief=tk.FLAT, padx=20, pady=10, width=15)
    area_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(area_btn)

    box_btn = tk.Button(row4_frame, text="Box Plot", command=lambda: [popup.destroy(), plot_graph("Box Plot")],
                      bg=SECONDARY_COLOR, fg=TEXT_COLOR, font=("Arial", 11),
                      relief=tk.FLAT, padx=20, pady=10, width=15)
    box_btn.pack(side=tk.LEFT, padx=5)
    add_hover_effect(box_btn)

    # Close button
    close_btn = tk.Button(popup, text="Close", command=popup.destroy,
                         bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                         relief=tk.FLAT, padx=20, pady=5)
    close_btn.pack(pady=(20, 10))
    add_hover_effect(close_btn)
# Visualize button
visualize_btn = tk.Button(left_panel, text="Visualize Data", command=show_visualization_options,
                        bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                        relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
visualize_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(visualize_btn)

# Analyze button
analyze_btn = tk.Button(left_panel, text="Analyze Data", command=show_regression_options,
                      bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                      relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
analyze_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(analyze_btn)

# Correlation Matrix button
corr_matrix_btn = tk.Button(left_panel, text="Correlation Matrix", command=show_correlation_matrix,
                          bg=ACCENT_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                          relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
corr_matrix_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(corr_matrix_btn)

# Boosting Algorithms button
boosting_btn = tk.Button(left_panel, text="Boosting Algorithms", command=show_boosting_options,
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
boosting_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(boosting_btn)

# Advanced Analysis button
advanced_btn = tk.Button(left_panel, text="Advanced Analysis", command=show_advanced_analysis_options,
                       bg=INFO_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                       relief=tk.FLAT, padx=30, pady=15, state=tk.DISABLED)
advanced_btn.pack(fill=tk.X, pady=(0, 20))
add_hover_effect(advanced_btn)

# Right panel (data preview and info)
right_panel = tk.Frame(content_frame, bg=BACKGROUND_COLOR)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Data preview label
preview_label = tk.Label(right_panel, text="Dataset Preview", 
                        font=("Arial", 14, "bold"), fg=TEXT_COLOR, bg=BACKGROUND_COLOR)
preview_label.pack(pady=(0, 10), anchor=tk.W)

# Data preview text widget with scrollbar
preview_frame = tk.Frame(right_panel, bg=BACKGROUND_COLOR)
preview_frame.pack(fill=tk.BOTH, expand=True)

data_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD, 
                                       bg=PRIMARY_COLOR, fg=TEXT_COLOR, 
                                       insertbackground=TEXT_COLOR, 
                                       font=("Courier", 10))
data_preview.pack(fill=tk.BOTH, expand=True)
data_preview.insert(tk.END, "No data loaded. Please click 'Load Dataset' to begin.")
data_preview.config(state=tk.DISABLED)

# Footer
footer_frame = tk.Frame(main_container, bg=BACKGROUND_COLOR)
footer_frame.pack(fill=tk.X, pady=(20, 0))

# Exit button
exit_btn = tk.Button(footer_frame, text="Exit", command=root.quit,
                    bg=DANGER_COLOR, fg=TEXT_COLOR, font=("Arial", 12, "bold"),
                    relief=tk.FLAT, padx=30, pady=10)
exit_btn.pack(pady=10)
add_hover_effect(exit_btn)

# Initialize empty DataFrame
df = pd.DataFrame()
original_df = pd.DataFrame()
X = pd.DataFrame()
y = pd.Series()

# Start the application
root.mainloop()