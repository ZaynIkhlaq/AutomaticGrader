import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1) Define helper functions
def read_csv_data(filepath: str) -> pd.DataFrame:
    """
    Reads a CSV file containing at least two columns: 'StudentID' and 'Score'.
    Returns a pandas DataFrame or raises an Exception if the file is missing columns.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {filepath}")
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV: {e}")

    # Basic error handling for missing columns
    required_cols = {'StudentID', 'Score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    return df

def assign_absolute_grade(score, thresholds=None):
    """
    Assigns an absolute letter grade based on fixed numeric thresholds.
    Example thresholds:
    {
        'A': 90,
        'B': 80,
        'C': 70,
        'D': 60
    }
    """
    if thresholds is None:
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}

    if score >= thresholds['A']:
        return 'A'
    elif score >= thresholds['B']:
        return 'B'
    elif score >= thresholds['C']:
        return 'C'
    elif score >= thresholds['D']:
        return 'D'
    else:
        return 'F'

def transform_scores_normal_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs z-score scaling (normal-curve approach) on 'Score'.
    """
    df_new = df.copy()
    mu = df['Score'].mean()
    sigma = df['Score'].std()

    if sigma == 0:
        # All scores are identical; no transformation
        df_new['AdjustedScore'] = df['Score']
        return df_new

    # Standard z-score
    z_scores = (df['Score'] - mu) / sigma
    df_new['AdjustedScore'] = z_scores

    return df_new

def assign_letter_grades_from_percentiles(df: pd.DataFrame, grade_col='FinalGrade') -> pd.DataFrame:
    """
    Assign letter grades based on the percentile of 'AdjustedScore' in a normal distribution.
    By default:
      - A: top 20% (percentile >= 0.80)
      - B: next 30% (0.50 to 0.80)
      - C: next 30% (0.20 to 0.50)
      - D: next 10% (0.10 to 0.20)
      - F: bottom 10% (0.00 to 0.10)
    """
    df_new = df.copy()
    if 'AdjustedScore' not in df_new.columns:
        # If no 'AdjustedScore', just copy 'Score' to 'AdjustedScore'
        df_new['AdjustedScore'] = df_new['Score']

    if df_new['AdjustedScore'].nunique() == 1:
        # If all adjusted scores are the same, assign 'C'
        df_new[grade_col] = 'C'
        return df_new

    z_scores = df_new['AdjustedScore']
    percentiles = norm.cdf(z_scores)

    letter_bins = {'A': 0.80, 'B': 0.50, 'C': 0.20, 'D': 0.10, 'F': 0.00}
    letter_grades = []
    for p in percentiles:
        if p >= letter_bins['A']:
            letter_grades.append('A')
        elif p >= letter_bins['B']:
            letter_grades.append('B')
        elif p >= letter_bins['C']:
            letter_grades.append('C')
        elif p >= letter_bins['D']:
            letter_grades.append('D')
        else:
            letter_grades.append('F')

    df_new[grade_col] = letter_grades
    return df_new

def assign_relative_grade_distribution(df: pd.DataFrame, 
                                       distribution=None, 
                                       grade_col='FinalGrade') -> pd.DataFrame:
    """
    Forces letter grades by sorting from highest to lowest 'Score' 
    and assigning top X% to 'A', next Y% to 'B', etc., 
    based on the given distribution.
    """
    df_new = df.copy()

    if distribution is None:
        distribution = {'A': 0.20, 'B': 0.30, 'C': 0.30, 'D': 0.10, 'F': 0.10}

    # Sort scores descending
    sorted_df = df_new.sort_values(by='Score', ascending=False).reset_index(drop=True)
    n = len(sorted_df)

    # Convert percentages to counts
    grade_counts = {}
    for g, pct in distribution.items():
        grade_counts[g] = int(round(pct * n))

    # Correct any rounding errors to ensure total == n
    diff = n - sum(grade_counts.values())
    if diff != 0:
        last_grade = list(distribution.keys())[-1]
        grade_counts[last_grade] += diff

    # Assign
    assigned_grades = [''] * n
    start_idx = 0
    for g in distribution.keys():
        count = grade_counts[g]
        end_idx = start_idx + count
        for i in range(start_idx, end_idx):
            assigned_grades[i] = g
        start_idx = end_idx

    sorted_df[grade_col] = assigned_grades

    # Merge back to original order
    df_merged = pd.merge(df_new, sorted_df[['StudentID', grade_col]], on='StudentID', how='left')
    return df_merged


# 2) Main Streamlit App
def main():
    st.title("Grading System & Exploratory Analysis")

    st.markdown("""
    This Streamlit app demonstrates:
    - How to load a CSV file with student IDs and scores
    - How to assign absolute grades
    - How to transform scores to a normal curve
    - How to assign letter grades using a percentile-based approach
    - How to assign relative letter grades using a forced distribution
    - Basic exploratory data analysis plots
    """)

    # -- File uploader in Streamlit
    uploaded_file = st.file_uploader("Upload CSV file with columns ['StudentID', 'Score']:",
                                     type=["csv"])
    if not uploaded_file:
        st.warning("Please upload a CSV file to proceed.")
        return

    # --- Read and display data
    df = read_csv_data(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # --- Basic Info
    st.write("**Data Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    # --- Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # --- Exploratory Plots
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    st.subheader("Distribution of Numerical Columns")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(df[col], bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

    # --- Correlation Matrix
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots()
    ax.matshow(corr_matrix, cmap='coolwarm')
    st.pyplot(fig)

    # --- Statistics about Original Scores
    mean_score = df['Score'].mean()
    var_score = df['Score'].var()
    std_score = df['Score'].std()
    skew_score = df['Score'].skew()

    st.subheader("Original Scores Statistics")
    st.write(f"**Mean Score:** {mean_score:.2f}")
    st.write(f"**Variance:** {var_score:.2f}")
    st.write(f"**Standard Deviation:** {std_score:.2f}")
    st.write(f"**Skewness:** {skew_score:.2f}")

    # --- Plots of Original Scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df['Score'], bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    axes[0].set_title("Histogram of Original Scores", fontsize=14)
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Count")

    axes[1].plot(df['Score'], color='lightcoral')
    axes[1].set_title("Line Plot of Original Scores", fontsize=14)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Score")

    plt.tight_layout()
    st.pyplot(fig)

    # --- Choose Grading Method
    grading_method = st.selectbox("Select Grading Method", ["absolute", "relative"])
    relative_approach = st.selectbox("Select Relative Approach", ["normal_curve", "forced_distribution"])

    abs_thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
    rel_distribution = {'A': 0.20, 'B': 0.30, 'C': 0.30, 'D': 0.10, 'F': 0.10}

    df_grading = df.copy()

    if grading_method == 'absolute':
        df_grading['Grade'] = df_grading['Score'].apply(assign_absolute_grade, thresholds=abs_thresholds)
        df_grading['AdjustedScore'] = df_grading['Score']
    else:
        if relative_approach == 'normal_curve':
            df_transformed = transform_scores_normal_curve(df_grading)
            df_final = assign_letter_grades_from_percentiles(df_transformed, grade_col='Grade')
            df_grading = df_final.copy()
        else:
            df_final = assign_relative_grade_distribution(
                df_grading, distribution=rel_distribution, grade_col='Grade'
            )
            df_final['AdjustedScore'] = df_final['Score']
            df_grading = df_final.copy()

    # --- Final Grade Distribution
    st.subheader(f"Final Grade Distribution: {grading_method.capitalize()} Method")
    final_counts = df_grading['Grade'].value_counts().sort_index()

    # Show distribution in text
    for g, cnt in final_counts.items():
        st.write(f"Grade {g}: {cnt} students")

    # Bar plot of final grade distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(final_counts.index, final_counts.values, color='salmon')
    ax.set_title(f"Final Grade Distribution ({grading_method.capitalize()})", fontsize=14)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # If we have an AdjustedScore column, show distribution
    if 'AdjustedScore' in df_grading.columns:
        st.subheader("Adjusted Scores Analysis")
        adj_mean = df_grading['AdjustedScore'].mean()
        adj_std = df_grading['AdjustedScore'].std()

        st.write(f"**Adjusted Mean:** {adj_mean:.2f}")
        st.write(f"**Adjusted Std:** {adj_std:.2f}")

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        axes2[0].hist(df_grading['AdjustedScore'], bins='auto', color='blueviolet', alpha=0.7, rwidth=0.85)
        axes2[0].set_title("Histogram of Adjusted Scores", fontsize=14)
        axes2[0].set_xlabel("AdjustedScore")
        axes2[0].set_ylabel("Count")

        axes2[1].plot(df_grading['AdjustedScore'], color='goldenrod')
        axes2[1].set_title("Line Plot of Adjusted Scores", fontsize=14)
        axes2[1].set_xlabel("Index")
        axes2[1].set_ylabel("AdjustedScore")

        plt.tight_layout()
        st.pyplot(fig2)

    # Compare to absolute grading
    df_abs_compare = df.copy()
    df_abs_compare['AbsGrade'] = df_abs_compare['Score'].apply(assign_absolute_grade, thresholds=abs_thresholds)

    merged = pd.merge(
        df_abs_compare[['StudentID', 'AbsGrade']],
        df_grading[['StudentID', 'Grade']],
        on='StudentID',
        how='left'
    )
    merged['Changed'] = merged['AbsGrade'] != merged['Grade']
    changed_count = merged['Changed'].sum()

    st.write(f"Number of students whose grade changed compared to absolute grading: {changed_count}")
    st.success("Grading and analysis completed successfully!")


if __name__ == '__main__':
    main()