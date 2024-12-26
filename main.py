import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set Streamlit page configurations
st.set_page_config(
    page_title="Grading System App",
    page_icon="📊",
    layout="centered"
)

# Set a Seaborn style/theme
sns.set_theme(style='whitegrid', palette='pastel')

# -----------------------------
# Helper Functions
# -----------------------------
def read_csv_data(uploaded_file) -> pd.DataFrame:
    """
    Reads a CSV file containing at least two columns: 'StudentID' and 'Score'.
    Returns a pandas DataFrame or raises an Exception if the file is missing columns.
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
    Assigns an absolute letter grade based on fixed numeric thresholds.
    """
    if thresholds is None:
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}
    # You can add or change thresholds for E, F, etc.

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
    If all scores are identical (std=0), no transformation is applied.
    """
    df_new = df.copy()
    mu = df['Score'].mean()
    sigma = df['Score'].std()

    # If all scores are the same, can't standardize
    if sigma == 0:
        df_new['AdjustedScore'] = df['Score']
    else:
        # Standard z-score transformation
        z_scores = (df['Score'] - mu) / sigma
        df_new['AdjustedScore'] = z_scores

    return df_new


def assign_letter_grades_from_percentiles(df: pd.DataFrame, grade_col='FinalGrade') -> pd.DataFrame:
    """
    Assign letter grades based on the percentile of 'AdjustedScore'
    in a (theoretical) normal distribution.
    """
    df_new = df.copy()
    if 'AdjustedScore' not in df_new.columns:
        df_new['AdjustedScore'] = df_new['Score']

    # Convert each z-score into its percentile via the Normal CDF
    z_scores = df_new['AdjustedScore']
    percentiles = norm.cdf(z_scores)

    # Define percentile cutoffs for letter grades
    letter_bins = {
        'A': 0.80,
        'B': 0.50,
        'C': 0.20,
        'D': 0.10,
        'F': 0.00
    }
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


def plot_score_distribution(df, score_col='Score', label='Score Distribution'):
    """
    Plots a histogram + KDE + theoretical Normal PDF based on the data's mean & std.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram (density = True)
    sns.histplot(
        df[score_col], 
        bins=15, 
        stat='density', 
        color='skyblue', 
        alpha=0.6,
        edgecolor='black',
        label='Histogram',
        ax=ax
    )
    # KDE plot
    sns.kdeplot(
        df[score_col],
        color='blue',
        linewidth=2,
        label='KDE',
        ax=ax
    )

    # Theoretical normal curve with sample mean/std
    mean_val = df[score_col].mean()
    std_val = df[score_col].std()

    # Guard against zero std (all scores identical)
    if std_val > 0:
        x_vals = np.linspace(df[score_col].min(), df[score_col].max(), 200)
        pdf_vals = norm.pdf(x_vals, mean_val, std_val)
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Normal PDF')

    ax.set_title(label)
    ax.set_xlabel(score_col)
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)


def plot_grade_vs_avg_score(df, grade_col='Grade', score_col='Score', possible_grades=None):
    """
    Plots a simple line plot of average score by letter grade, in the order given.
    """
    # If the user doesn't supply an order, we try a default
    # (If you have A, B, C, D, E, F or something else, you can adjust.)
    if possible_grades is None:
        # For the "Absolute Grading" example (A,B,C,D,F).
        # For "Relative Grading," it's also A,B,C,D,F.
        possible_grades = ["A", "B", "C", "D", "F"]

    # Group by grade and compute average
    grade_means = (
        df.groupby(grade_col)[score_col]
        .mean()
        .reindex(possible_grades)
    )

    # Sometimes certain grades might not appear in the dataset at all.
    # We'll drop NaNs if the grade doesn't appear
    grade_means = grade_means.dropna()

    fig, ax = plt.subplots(figsize=(6,4))
    sns.lineplot(
        x=grade_means.index,
        y=grade_means.values,
        marker='o',
        color="purple",
        linewidth=2,
        ax=ax
    )
    ax.set_title(f"Average {score_col} by Letter Grade")
    ax.set_xlabel("Letter Grade")
    ax.set_ylabel(f"Average {score_col}")
    ax.set_ylim(0, 100)  # Assuming scores from 0 to 100
    st.pyplot(fig)


# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.title("📊 Grading System: Absolute vs. Relative Grading")
    
    st.markdown("""
    **Welcome to the Grading System App!**

    This application allows you to:
    - **Upload a CSV file** containing student scores.
    - **Choose a grading method**: Absolute or Relative.
    - **Review the assigned grades** and distribution, including:
        - A histogram + normal curve of your scores
        - A line plot of *average score* by letter grade
    
    **Note**: Your CSV file must have **at least** these two columns:
    - `StudentID`
    - `Score`
    """)

    # Upload CSV file
    uploaded_file = st.file_uploader(
        "Upload CSV file with columns ['StudentID', 'Score']",
        type=["csv"]
    )

    if not uploaded_file:
        st.warning("Please upload a valid CSV file to proceed.")
        return

    # Read CSV data
    try:
        df = read_csv_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read the file: {e}")
        return

    # Display a preview of the uploaded data
    st.subheader("Data Preview")
    st.dataframe(df.head(), height=200)

    # Let the user pick a grading method
    grading_method = st.selectbox(
        "Choose a Grading Method",
        ["Absolute Grading", "Relative Grading"]
    )
    
    if grading_method == "Absolute Grading":
        st.subheader("Absolute Grading")
        thresholds = {'A': 90, 'B': 80, 'C': 70, 'D': 60}

        # Assign grades
        df['Grade'] = df['Score'].apply(assign_absolute_grade, thresholds=thresholds)

        st.write("Grades assigned based on these **absolute thresholds**:")
        st.json(thresholds)

        # Show some rows
        st.dataframe(df[['StudentID', 'Score', 'Grade']].head(), height=200)

        # Plot distribution (raw scores)
        st.write("### Score Distribution (Histogram + Normal Curve)")
        plot_score_distribution(df, score_col='Score', label="Distribution of Raw Scores")

        # Grade distribution (counts)
        st.write("### Grade Distribution (Counts)")
        grade_counts = df['Grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

        # Plot average score by letter grade
        st.write("### Average Score by Letter Grade")
        plot_grade_vs_avg_score(df, grade_col='Grade', score_col='Score', 
                                possible_grades=["A","B","C","D","F"])

    else:
        st.subheader("Relative Grading")

        # Transform scores to z-scores
        df_transformed = transform_scores_normal_curve(df)
        # Assign letter grades from percentile cutoffs
        df_grades = assign_letter_grades_from_percentiles(df_transformed)

        st.write("Grades assigned based on **percentile ranks** in a normal distribution.")

        st.dataframe(
            df_grades[['StudentID', 'Score', 'AdjustedScore', 'FinalGrade']].head(),
            height=200
        )

        # Plot distribution of *raw* scores or of *AdjustedScore*? Let's do both.
        # First, raw scores:
        st.write("### Raw Score Distribution (Histogram + Normal Curve)")
        plot_score_distribution(df_grades, score_col='Score', label="Distribution of Raw Scores")

        # Then, adjusted z-scores:
        st.write("### Adjusted Score (Z-Score) Distribution + Std Normal")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Histogram of z-scores
        sns.histplot(
            df_grades['AdjustedScore'], 
            bins=15,
            stat='density',
            color='skyblue',
            alpha=0.7,
            edgecolor='black',
            label='Histogram',
            ax=ax
        )
        # KDE
        sns.kdeplot(
            df_grades['AdjustedScore'],
            color='blue',
            linewidth=2,
            label='KDE',
            ax=ax
        )
        # Theoretical standard normal curve
        x_vals = np.linspace(-4, 4, 200)
        pdf_vals = norm.pdf(x_vals, 0, 1)
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Std Normal PDF')

        ax.set_title("Distribution of Adjusted Scores (Z-Scores)")
        ax.set_xlabel("Adjusted Score")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        # Grade distribution (counts)
        st.write("### Grade Distribution (Counts)")
        grade_counts = df_grades['FinalGrade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

        # Plot average *raw* score by letter grade
        st.write("### Average Raw Score by Letter Grade (Relative Grading)")
        plot_grade_vs_avg_score(
            df_grades, 
            grade_col='FinalGrade', 
            score_col='Score', 
            possible_grades=["A","B","C","D","F"]
        )

    st.success("Grading completed successfully!")

# Run the app
if __name__ == '__main__':
    main()
