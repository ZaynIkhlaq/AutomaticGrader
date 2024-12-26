import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configure Streamlit page
st.set_page_config(
    page_title="Grading System App",
    page_icon="📊",
    layout="centered"
)

sns.set_theme(style='whitegrid', palette='pastel')

# -----------------------------
# Helper Functions
# -----------------------------
def read_csv_data(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV file containing 'StudentID' and 'Score'.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise Exception(f"An error occurred while reading the CSV: {e}")

    required_cols = {'StudentID', 'Score'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    return df


def assign_absolute_grade(score, thresholds=None):
    """
    Assign an absolute letter grade based on numeric thresholds.
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
    Performs z-score scaling on 'Score'.
    """
    df_new = df.copy()
    mu = df['Score'].mean()
    sigma = df['Score'].std()

    if sigma == 0:
        df_new['AdjustedScore'] = df['Score']
    else:
        df_new['AdjustedScore'] = (df['Score'] - mu) / sigma

    return df_new


def assign_letter_grades_from_percentiles(df: pd.DataFrame,
                                          grade_col='FinalGrade') -> pd.DataFrame:
    """
    Assign letter grades based on percentile (normal CDF) of 'AdjustedScore'.
    """
    df_new = df.copy()
    # If there's no AdjustedScore, fallback to raw Score
    if 'AdjustedScore' not in df_new.columns:
        df_new['AdjustedScore'] = df_new['Score']

    z_scores = df_new['AdjustedScore']
    percentiles = norm.cdf(z_scores)

    # Typical percentile cutoffs (adjust as needed)
    cutoffs = {'A': 0.80, 'B': 0.50, 'C': 0.20, 'D': 0.10, 'F': 0.00}
    letter_grades = []
    for p in percentiles:
        if p >= cutoffs['A']:
            letter_grades.append('A')
        elif p >= cutoffs['B']:
            letter_grades.append('B')
        elif p >= cutoffs['C']:
            letter_grades.append('C')
        elif p >= cutoffs['D']:
            letter_grades.append('D')
        else:
            letter_grades.append('F')

    df_new[grade_col] = letter_grades
    return df_new


def plot_distribution(df, col='Score', title='Score Distribution'):
    """
    Plot histogram + KDE + normal PDF for a given numeric column.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # Histogram
    sns.histplot(df[col], bins=15, stat='density', color='skyblue',
                 alpha=0.6, edgecolor='black', label='Histogram', ax=ax)
    # KDE
    sns.kdeplot(df[col], color='blue', linewidth=2, label='KDE', ax=ax)

    # Theoretical normal PDF
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val != 0:
        x_vals = np.linspace(df[col].min(), df[col].max(), 200)
        pdf_vals = norm.pdf(x_vals, loc=mean_val, scale=std_val)
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Normal PDF')

    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


def plot_grade_distribution(df, grade_col='Grade', title='Grade Distribution'):
    """
    Bar chart showing how many students got each grade.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    order = sorted(df[grade_col].unique())
    sns.countplot(x=grade_col, data=df, order=order, color='salmon', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Grade")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def plot_grade_vs_score(df, grade_col='Grade', score_col='Score',
                        all_grades=None, title='Average Score by Grade'):
    """
    Plots a simple line chart: x-axis = Grade, y-axis = average Score.
    """
    if all_grades is None:
        # If user hasn't passed an explicit grade list, just use sorted unique:
        all_grades = sorted(df[grade_col].unique())

    # Calculate the average for each grade, preserving the specified order
    grouped = df.groupby(grade_col)[score_col].mean().reindex(all_grades)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(
        x=grouped.index,
        y=grouped.values,
        marker='o',
        color='purple',
        linewidth=2,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Grade")
    ax.set_ylabel(f"Average {score_col}")
    ax.set_ylim(0, 100)  # Adjust if your scores can exceed 100
    st.pyplot(fig)

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.title("📊 Grading System: Absolute vs. Relative Grading")
    
    st.markdown("""
    **Welcome to the Grading System App!**

    This application:
    1. Lets you upload a CSV with **StudentID** and **Score**.
    2. Lets you choose **Absolute** or **Relative** grading.
    3. Shows distribution plots (histogram + KDE + normal PDF).
    4. **Plots a line chart of grade vs. average score.**
    """)

    uploaded_file = st.file_uploader(
        "Upload CSV with ['StudentID','Score']",
        type=["csv"]
    )

    if not uploaded_file:
        st.warning("Please upload a valid CSV to proceed.")
        return

    # Read data
    try:
        df = read_csv_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    st.subheader("Data Preview")
    st.dataframe(df.head(), height=200)

    grading_method = st.selectbox(
        "Choose a Grading Method",
        ["Absolute Grading", "Relative Grading"]
    )

    if grading_method == "Absolute Grading":
        st.subheader("Absolute Grading")
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
        df['Grade'] = df['Score'].apply(assign_absolute_grade, thresholds=thresholds)
        
        st.write("Grades assigned based on these **absolute thresholds**:")
        st.json(thresholds)
        st.dataframe(df[['StudentID','Score','Grade']].head(), height=200)
        
        # Plot distribution of raw scores
        plot_distribution(df, col='Score', title='Score Distribution (Absolute)')

        # Grade distribution
        plot_grade_distribution(df, grade_col='Grade', 
                                title='Final Grade Distribution (Absolute)')

        # Plot average Score by Grade
        plot_grade_vs_score(
            df, 
            grade_col='Grade', 
            score_col='Score',
            all_grades=["A","B","C","D","F"],  # your chosen categories
            title='Average Score by Grade (Absolute)'
        )

    else:
        st.subheader("Relative Grading")
        df_z = transform_scores_normal_curve(df)
        df_grades = assign_letter_grades_from_percentiles(df_z, 'FinalGrade')

        st.dataframe(df_grades[['StudentID','Score','AdjustedScore','FinalGrade']].head(), height=200)

        # Plot raw score distribution
        plot_distribution(df_grades, col='Score', title='Raw Score Distribution (Relative)')

        # Plot adjusted score distribution
        st.write("**Adjusted Score (Z-Score) Distribution**")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df_grades['AdjustedScore'], bins=15, stat='density',
                     color='skyblue', alpha=0.7, edgecolor='black', label='Histogram', ax=ax)
        sns.kdeplot(df_grades['AdjustedScore'], color='blue', lw=2, label='KDE', ax=ax)

        # Theoretical Standard Normal
        x_vals = np.linspace(-4, 4, 200)
        pdf_vals = norm.pdf(x_vals, 0, 1)
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Std Normal PDF')
        ax.set_title("Distribution of Adjusted Scores (Z-Scores)")
        ax.set_xlabel("Adjusted Score")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        # Grade distribution
        plot_grade_distribution(df_grades, grade_col='FinalGrade', 
                                title='Final Grade Distribution (Relative)')

        # Finally, the EXACT Grade vs. Score PLOT:
        plot_grade_vs_score(
            df_grades,
            grade_col='FinalGrade',
            score_col='Score',
            all_grades=["A","B","C","D","F"],  # or tweak to your liking
            title='Average Score by Grade (Relative)'
        )

    st.success("Grading completed successfully!")

if __name__ == '__main__':
    main()
