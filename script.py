import pandas as pd
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns

folder_path = 'docs'
file_gpt4o = os.path.join(folder_path, 'autoqa_output_gpt4o_240.csv')
file_gpt51 = os.path.join(folder_path, 'autoqa_output_gpt51_240.csv')
file_outcomes = os.path.join(folder_path, 'outcomes_240.csv')

def run_full_analysis():
    df_4o = pd.read_csv(file_gpt4o)
    df_51 = pd.read_csv(file_gpt51)
    df_outcomes = pd.read_csv(file_outcomes)

    reliability_df = pd.merge(df_4o, df_51, on='record_id', suffixes=('_4o', '_51'))
    predictive_df = pd.merge(df_4o, df_outcomes, on='record_id')

    reliability_results = []
    predictive_results = []

    # Bloque 1: Fiabilidad
    for i in range(1, 17):
        col_4o = f'item_{i}_answer_4o'
        col_51 = f'item_{i}_answer_51'
        agreement = (reliability_df[col_4o] == reliability_df[col_51]).mean()
        kappa = cohen_kappa_score(reliability_df[col_4o], reliability_df[col_51])
        prev_4o = reliability_df[col_4o].mean()
        prev_51 = reliability_df[col_51].mean()
        drift = abs(prev_4o - prev_51)
        
        reliability_results.append({
            'Item': f'item_{i}',
            'Agreement (%)': agreement * 100,
            'Kappa': round(kappa, 4),
            'Drift': round(drift, 4)
        })

    for i in range(1, 17):
        item_col = f'item_{i}_answer'
        corr_next, p_next = pointbiserialr(predictive_df[item_col], predictive_df['next_lesson_attended'])
        corr_ret, p_ret = pointbiserialr(predictive_df[item_col], predictive_df['m1_retained'])
        
        mean_ret = predictive_df[predictive_df['m1_retained'] == 1][item_col].mean()
        mean_churn = predictive_df[predictive_df['m1_retained'] == 0][item_col].mean()
        
        predictive_results.append({
            'Item': f'item_{i}',
            'Corr_Next_Lesson': round(corr_next, 4),
            'Corr_M1_Retained': round(corr_ret, 4),
            'P_Value_M1': round(p_ret, 4),
            'Retention_Gap': round(mean_ret - mean_churn, 4)
        })

    rel_table = pd.DataFrame(reliability_results).sort_values(by='Kappa', ascending=False)
    pred_table = pd.DataFrame(predictive_results).sort_values(by='Corr_M1_Retained', ascending=False)

    return rel_table, pred_table

def plot_reliability(rel_df):
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    sns.barplot(x='Item', y='Kappa', data=rel_df, palette='viridis')
    plt.axhline(0.6, color='green', linestyle='--', label='Sustancial (0.6)')
    plt.axhline(0.4, color='orange', linestyle='--', label='Moderado (0.4)')
    
    plt.title('Consistency Across Models (Cohen\'s Kappa)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_predictive_utility(pred_df):
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    
    pred_df_sorted = pred_df.sort_values('Corr_M1_Retained', ascending=False)
    
    sns.barplot(x='Item', y='Corr_M1_Retained', data=pred_df_sorted, palette='magma')
    plt.title('Relationship Between Items and Student Retention (M1)', fontsize=14)
    plt.ylabel('Correlation (Biserial Point)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        rel_table, pred_table = run_full_analysis()
        
        print("--- RELIABILITY TABLE ---")
        print(rel_table)
        
        plot_reliability(rel_table)
        plot_predictive_utility(pred_table)
        
    except FileNotFoundError:
        print(f"Error: The CSV files must be located in the folder '{folder_path}'")