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

# Set a more attractive Seaborn style/theme
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
    # If 'AdjustedScore' does not exist, just use the raw Score
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
    - **Review the assigned grades** and distribution.

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

        # Display results
        st.dataframe(df[['StudentID', 'Score', 'Grade']].head(), height=200)
        
        # Grade distribution
        st.write("### Grade Distribution")
        grade_counts = df['Grade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

    else:
        st.subheader("Relative Grading")
        
        # Transform scores to z-scores
        df_transformed = transform_scores_normal_curve(df)
        df_grades = assign_letter_grades_from_percentiles(df_transformed)

        st.write("Grades assigned based on **percentile ranks** in a normal distribution.")
        
        # Display results
        st.dataframe(
            df_grades[['StudentID', 'Score', 'AdjustedScore', 'FinalGrade']].head(),
            height=200
        )
        
        # -----------------------------------------------------
        # Enhanced Plot: Adjusted Score Distribution + Normal Curve
        # -----------------------------------------------------
        st.write("### Adjusted Score Distribution with Normal Curve")
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot histogram of Adjusted Scores (z-scores), with density on the y-axis
        # Increase bins, add some transparency, and remove the black edgecolor if you like
        sns.histplot(
            df_grades['AdjustedScore'], 
            bins=20,                    # You can tweak this number
            stat='density', 
            color='skyblue', 
            alpha=0.7, 
            edgecolor='black',
            ax=ax,
            label='Histogram'
        )

        # Overlay a KDE for a smoother visualization
        sns.kdeplot(
            df_grades['AdjustedScore'], 
            color='blue', 
            linewidth=2, 
            ax=ax,
            label='KDE'
        )

        # Because we used z-scores, the theoretical normal distribution 
        # is Standard Normal: mean=0, std=1
        x_vals = np.linspace(-4, 4, 200)
        pdf_vals = norm.pdf(x_vals, 0, 1)

        # Plot the PDF (standard normal) as a dashed red line
        ax.plot(x_vals, pdf_vals, 'r--', lw=2, label='Std Normal PDF')
        
        ax.set_title("Distribution of Adjusted Scores (Z-Scores)")
        ax.set_xlabel("Adjusted Score (Z-Score)")
        ax.set_ylabel("Density")
        ax.legend()

        st.pyplot(fig)
        # -----------------------------------------------------

        # Grade distribution
        st.write("### Grade Distribution")
        grade_counts = df_grades['FinalGrade'].value_counts().sort_index()
        st.bar_chart(grade_counts)

    st.success("Grading completed successfully!")

if __name__ == '__main__':
    main()
