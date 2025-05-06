import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import ast
import pickle
import os # Added for file checking

# Download NLTK resources if not already downloaded
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

warnings.filterwarnings('ignore')

class JobRecommenderSystem:
    def __init__(self, simulate_bias=False, bias_factor=0.3, bias_attribute='gender', bias_target_keywords=None):
        """
        Initialize the recommender system.

        Args:
            simulate_bias (bool): If True, artificially introduce bias for demonstration.
            bias_factor (float): Strength of the simulated bias (0 to 1).
                                 Positive factor penalizes the target group.
            bias_attribute (str): Protected attribute to simulate bias on (e.g., 'gender', 'age_group').
            bias_target_keywords (list): Keywords in job title/desc to target for bias simulation.
                                         Defaults to ['manager', 'senior', 'lead', 'director'].
        """
        self.jobs_df = None
        self.applicants_df = None
        self.job_vectors = None
        self.tfidf_vectorizer = None
        # self.preprocessor = None # This was defined but not used, removing for clarity
        self.fairness_metrics = {}
        self.debiased_model = False
        self.bias_mitigation_method = None
        self.protected_attributes = []

        # Bias Simulation Parameters
        self.simulate_bias = simulate_bias
        self.bias_factor = bias_factor if 0 <= bias_factor <= 1 else 0.3
        self.bias_attribute = bias_attribute
        self.bias_target_keywords = bias_target_keywords if bias_target_keywords else ['manager', 'senior', 'lead', 'director']
        print(f"Bias Simulation Initialized: simulate_bias={self.simulate_bias}, factor={self.bias_factor}, attribute='{self.bias_attribute}'")


    def load_data(self, jobs_path, applicants_path):
        """Load and preprocess the job and applicant data"""
        print("Loading and preprocessing data...")

        if not os.path.exists(jobs_path):
            print(f"ERROR: Jobs file not found at {jobs_path}")
            return None, None
        if not os.path.exists(applicants_path):
            print(f"ERROR: Applicants file not found at {applicants_path}")
            return None, None

        try:
            # Load the job listings
            self.jobs_df = pd.read_csv(jobs_path, encoding="unicode_escape")
            print(f"Initial Job Columns: {self.jobs_df.columns.tolist()}")


            # Load the applicant data
            self.applicants_df = pd.read_csv(applicants_path)
            print(f"Initial Applicant Columns: {self.applicants_df.columns.tolist()}")

            # Display info about the loaded data
            print(f"Loaded {len(self.jobs_df)} job listings and {len(self.applicants_df)} applicant profiles")

            return self.jobs_df, self.applicants_df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def preprocess_job_data(self):
        """Clean and preprocess the job data"""
        if self.jobs_df is None:
            print("Job data not loaded. Cannot preprocess.")
            return None

        print("Preprocessing job data...")

        # --- Robust Column Mapping ---
        # Define potential names for each required field
        col_map = {
            'job_id': ['Job Id', 'job_id', 'ID'],
            'job_title': ['Job Id.14', 'Title', 'Job Title'],
            'job_category': ['Job Id.15', 'Category', 'Job Category'],
            'skills': ['Job Id.19', 'Skills', 'Required Skills'],
            'responsibilities': ['Job Id.20', 'Responsibilities', 'Job Description'],
            'experience': ['Job Id.1', 'Experience', 'Years Experience'],
            'education': ['Job Id.2', 'Education', 'Required Education'],
            'salary_range': ['Job Id.3', 'Salary', 'Salary Range'],
            'city': ['Job Id.4', 'City', 'Location City'],
            'country': ['Job Id.5', 'Country', 'Location Country'],
            'employment_type': ['Job Id.8', 'Employment Type', 'Type'],
            'company': ['Job Id.21', 'Company Name', 'Company'],
            'company_info': ['Job Id.22', 'Company Info', 'About Company']
        }

        # Assign columns based on the first match found
        processed_cols = set()
        for target_col, potential_names in col_map.items():
            found = False
            for name in potential_names:
                if name in self.jobs_df.columns:
                    self.jobs_df[target_col] = self.jobs_df[name]
                    processed_cols.add(name) # Keep track of original cols used
                    found = True
                    print(f"Mapped '{name}' to '{target_col}'")
                    break
            if not found:
                print(f"Warning: Could not find suitable column for '{target_col}'. Creating empty column.")
                self.jobs_df[target_col] = '' # Create empty column if not found

        # Ensure job_id is unique and handle potential missing values if it wasn't found properly
        if 'job_id' not in self.jobs_df.columns or self.jobs_df['job_id'].isnull().any() or self.jobs_df['job_id'].duplicated().any():
             print("Warning: 'job_id' column is missing, has NaNs, or duplicates. Generating unique IDs.")
             self.jobs_df['job_id'] = range(len(self.jobs_df))


        # Fill NaNs for text combination
        text_cols_for_combine = ['job_title', 'job_category', 'skills', 'responsibilities', 'education', 'experience']
        for col in text_cols_for_combine:
             if col in self.jobs_df.columns:
                 self.jobs_df[col] = self.jobs_df[col].fillna('')
             else:
                 # This case should be covered by the robust mapping, but as a safeguard:
                 self.jobs_df[col] = ''


        # Create a combined text field for TF-IDF
        self.jobs_df['combined_text'] = self.jobs_df.apply(
            lambda x: ' '.join([
                str(x.get(col, '')) for col in text_cols_for_combine
            ]), axis=1
        )

        # If combined_text is empty for many rows, fall back to using all object columns
        if (self.jobs_df['combined_text'].str.strip() == '').sum() > len(self.jobs_df) * 0.5:
             print("Warning: Many combined_text entries are empty. Falling back to using all string columns.")
             object_columns = self.jobs_df.select_dtypes(include=['object']).columns.tolist()
             # Exclude already processed/created columns to avoid duplication
             object_columns = [c for c in object_columns if c not in processed_cols and c not in ['job_id', 'combined_text']]
             self.jobs_df['combined_text_fallback'] = self.jobs_df[object_columns].fillna('').astype(str).apply(
                 lambda x: ' '.join(x), axis=1
             )
             self.jobs_df['combined_text'] = self.jobs_df['combined_text'] + ' ' + self.jobs_df['combined_text_fallback']
             self.jobs_df.drop(columns=['combined_text_fallback'], inplace=True)


        # Clean the combined text
        self.jobs_df['combined_text'] = self.jobs_df['combined_text'].apply(self._clean_text)

        # Ensure essential columns exist and handle potential missing ones created by mapping
        essential_columns = ['job_id', 'combined_text', 'job_title', 'company', 'salary_range']
        final_columns = []
        for col in essential_columns:
            if col in self.jobs_df.columns:
                final_columns.append(col)
            else:
                print(f"Warning: Essential column '{col}' is missing after processing.")
        # Add other potentially useful columns that were successfully mapped
        final_columns.extend([c for c in col_map.keys() if c in self.jobs_df.columns and c not in final_columns])

        # Keep only existing columns and drop duplicates based on the reliable job_id
        self.jobs_df = self.jobs_df[list(dict.fromkeys(final_columns))].drop_duplicates(subset=['job_id']) # dict.fromkeys preserves order while unique
        self.jobs_df.reset_index(drop=True, inplace=True) # Reset index after drop_duplicates

        print(f"Job data preprocessed. Shape: {self.jobs_df.shape}. Columns: {self.jobs_df.columns.tolist()}")
        return self.jobs_df


    def preprocess_applicant_data(self):
        """Clean and preprocess the applicant data"""
        if self.applicants_df is None:
            print("Applicant data not loaded. Cannot preprocess.")
            return None

        print("Preprocessing applicant data...")

        # Create a unique identifier for each applicant
        if 'applicant_id' not in self.applicants_df.columns:
             self.applicants_df['applicant_id'] = self.applicants_df.index
        else:
             # Ensure it's unique if it exists
             if self.applicants_df['applicant_id'].duplicated().any():
                 print("Warning: Existing 'applicant_id' column has duplicates. Generating unique IDs.")
                 self.applicants_df['applicant_id'] = range(len(self.applicants_df))


        # Process education information if 'Education' column exists
        if 'Education' in self.applicants_df.columns:
            education_mapping = {
                'No high school diploma': 0, 'High school': 1, 'Some college': 2,
                'Associate degree': 3, 'Bachelor degree': 4, 'Master degree': 5,
                'Doctoral degree': 6, 'Professional degree': 7
            }
            # Normalize values before mapping (lowercase, strip whitespace)
            self.applicants_df['Education_Clean'] = self.applicants_df['Education'].str.lower().str.strip()
            self.applicants_df['education_level'] = self.applicants_df['Education_Clean'].map(
                education_mapping).fillna(0).astype(int)
            self.applicants_df.drop(columns=['Education_Clean'], inplace=True)
            print("Processed 'Education' column into 'education_level'.")
        else:
            print("Warning: 'Education' column not found in applicant data. Skipping education level processing.")
            self.applicants_df['education_level'] = 0 # Default value

        # Create combined text field for applicants
        # Identify potential text columns, exclude known non-textual or sensitive ones
        potential_text_cols = self.applicants_df.select_dtypes(include=['object']).columns.tolist()
        excluded_cols = ['applicant_id', 'Sex', 'Race', 'Hispanic', 'Education'] # Exclude original Education too
        text_columns = [col for col in potential_text_cols if col not in excluded_cols]

        if not text_columns:
             print("Warning: No suitable text columns found for applicant 'combined_text'. Using a placeholder.")
             # Attempt to use Experience if available, otherwise placeholder
             if 'Experience' in self.applicants_df.columns:
                 self.applicants_df['combined_text'] = self.applicants_df['Experience'].fillna('').astype(str)
             else:
                 self.applicants_df['combined_text'] = 'applicant profile' # Placeholder
        else:
             print(f"Using columns for applicant combined_text: {text_columns}")
             self.applicants_df['combined_text'] = self.applicants_df[text_columns].fillna('').astype(str).apply(
                 lambda x: ' '.join(x), axis=1
             )

        self.applicants_df['combined_text'] = self.applicants_df['combined_text'].apply(self._clean_text)

        # Handle protected attributes
        self._prepare_protected_attributes()

        print(f"Applicant data preprocessed. Shape: {self.applicants_df.shape}. Columns: {self.applicants_df.columns.tolist()}")
        return self.applicants_df


    def _prepare_protected_attributes(self):
        """Prepare protected attributes for bias detection and mitigation"""
        self.protected_attributes = [] # Reset list

        # Gender
        if 'Sex' in self.applicants_df.columns:
            self.applicants_df['gender'] = self.applicants_df['Sex'].fillna('Unknown') # Handle potential NaNs
            # Normalize common variations if necessary
            self.applicants_df['gender'] = self.applicants_df['gender'].replace({'M': 'Male', 'F': 'Female'})
            if 'gender' not in self.protected_attributes: self.protected_attributes.append('gender')
            print("Processed 'Sex' into 'gender'.")
        else:
            print("Warning: 'Sex' column not found for gender attribute.")

        # Race
        if 'Race' in self.applicants_df.columns:
            self.applicants_df['race'] = self.applicants_df['Race'].fillna('Unknown') # Handle potential NaNs
            if 'race' not in self.protected_attributes: self.protected_attributes.append('race')
            print("Processed 'Race' into 'race'.")
        else:
             print("Warning: 'Race' column not found.")

        # Age - Create age groups if 'Age' column exists
        if 'Age' in self.applicants_df.columns:
             # Ensure Age is numeric, coerce errors to NaN
             self.applicants_df['Age_Num'] = pd.to_numeric(self.applicants_df['Age'], errors='coerce')
             # Define bins and labels
             bins = [0, 25, 35, 45, 55, 65, 100]
             labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
             # Apply pd.cut
             self.applicants_df['age_group'] = pd.cut(
                 self.applicants_df['Age_Num'],
                 bins=bins,
                 labels=labels,
                 right=False # [0, 25), [25, 35), ...
             )
             # Handle potential NaNs from coercion or ages outside bins
             self.applicants_df['age_group'] = self.applicants_df['age_group'].cat.add_categories('Unknown').fillna('Unknown')
             if 'age_group' not in self.protected_attributes: self.protected_attributes.append('age_group')
             self.applicants_df.drop(columns=['Age_Num'], inplace=True) # Remove temporary numeric column
             print("Processed 'Age' into 'age_group'.")

        else:
             print("Warning: 'Age' column not found for age_group attribute.")


        print(f"Protected attributes identified: {self.protected_attributes}")
        # Validate the chosen bias simulation attribute exists
        if self.simulate_bias and self.bias_attribute not in self.protected_attributes:
             print(f"WARNING: Bias simulation attribute '{self.bias_attribute}' not found in processed applicant data. Disabling simulation.")
             self.simulate_bias = False


    def _clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove emails
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove numbers and punctuation, keep spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenization (simple split)
        tokens = text.split()

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1] # Keep words > 1 char

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def build_recommendation_model(self):
        """Build the job recommendation model using TF-IDF vectorization"""
        if self.jobs_df is None or 'combined_text' not in self.jobs_df.columns:
             print("ERROR: Job data or 'combined_text' column not available. Cannot build model.")
             return None
        if self.jobs_df['combined_text'].isnull().all():
             print("ERROR: 'combined_text' column contains only null values. Cannot build model.")
             return None

        print("Building recommendation model...")

        # Create and fit the TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,       # Limit features to prevent excessive memory usage
            stop_words='english',
            ngram_range=(1, 2),      # Use unigrams and bigrams
            min_df=2,                # Ignore terms that appear in less than 2 documents
            max_df=0.95              # Ignore terms that appear in more than 95% of documents
        )

        try:
            # Transform job descriptions
            self.job_vectors = self.tfidf_vectorizer.fit_transform(self.jobs_df['combined_text'])
            print(f"Created job vectors with shape: {self.job_vectors.shape}")
            if self.job_vectors.shape[0] != len(self.jobs_df):
                 print("WARNING: Mismatch between job vector rows and jobs DataFrame length!")
            return self.job_vectors
        except ValueError as ve:
             print(f"ERROR building TF-IDF model: {ve}")
             print("This might happen if the vocabulary is empty after applying filters (min_df, max_df). Check text cleaning and data.")
             return None
        except Exception as e:
             print(f"An unexpected error occurred during TF-IDF model building: {e}")
             return None


    def get_job_recommendations(self, applicant_id, top_n=5, debias=False):
        """Get job recommendations for a specific applicant, potentially simulating bias or applying mitigation."""
        if self.tfidf_vectorizer is None or self.job_vectors is None:
            print("Model not built. Cannot get recommendations.")
            return pd.DataFrame()
        if self.applicants_df is None:
             print("Applicant data not loaded. Cannot get recommendations.")
             return pd.DataFrame()


        # Verify applicant exists
        applicant_row = self.applicants_df[self.applicants_df['applicant_id'] == applicant_id]
        if applicant_row.empty:
            print(f"Applicant ID {applicant_id} not found")
            return pd.DataFrame()
        applicant = applicant_row.iloc[0]

        # Transform applicant text
        try:
            applicant_vector = self.tfidf_vectorizer.transform([applicant['combined_text']])
        except Exception as e:
             print(f"Error transforming applicant text for ID {applicant_id}: {e}")
             return pd.DataFrame()


        # Calculate similarity scores
        similarity_scores = cosine_similarity(applicant_vector, self.job_vectors).flatten()

        # --- BIAS SIMULATION STEP ---
        # Apply simulated bias only if flag is set AND debiasing is NOT active
        apply_simulation = self.simulate_bias and self.bias_factor > 0 and not debias
        if apply_simulation and self.bias_attribute in applicant and self.bias_attribute in self.protected_attributes:
            applicant_group = applicant[self.bias_attribute]
            print(f"--- Simulating bias for applicant {applicant_id} (Group: {self.bias_attribute}={applicant_group}) ---")

            # Example: Penalize females for 'senior/manager' roles
            if self.bias_attribute == 'gender':
                target_group = 'Female' # Group to penalize
                favored_group = 'Male' # Group to potentially favor (optional)

                # Iterate through scores and adjust based on job title and applicant group
                adjusted_scores = []
                job_titles = self.jobs_df['job_title'].fillna('').str.lower().tolist() # Get titles once

                for i, score in enumerate(similarity_scores):
                     job_title = job_titles[i] # Use cached title
                     is_target_job = any(keyword in job_title for keyword in self.bias_target_keywords)

                     if is_target_job:
                         if applicant_group == target_group:
                             adjusted_score = score * (1 - self.bias_factor)
                             #print(f"  - Penalizing job {i} ('{job_title}') for {target_group}: {score:.3f} -> {adjusted_score:.3f}")
                             adjusted_scores.append(adjusted_score)
                         # Optional: Boost score for the favored group for the same jobs
                         # elif applicant_group == favored_group:
                         #     adjusted_score = score * (1 + self.bias_factor / 2) # Smaller boost
                         #     #print(f"  + Boosting job {i} ('{job_title}') for {favored_group}: {score:.3f} -> {adjusted_score:.3f}")
                         #     adjusted_scores.append(adjusted_score)
                         else:
                             adjusted_scores.append(score) # No change for other groups on target jobs
                     else:
                         adjusted_scores.append(score) # No change for non-target jobs

                similarity_scores = np.array(adjusted_scores) # Update scores with biased ones

            # Add similar logic here if biasing based on 'age_group' or 'race'
            elif self.bias_attribute == 'age_group':
                 target_group = '56-65' # Example: Penalize older workers for tech jobs
                 tech_keywords = ['software', 'engineer', 'developer', 'data scientist', 'programmer']
                 job_titles = self.jobs_df['job_title'].fillna('').str.lower().tolist()
                 adjusted_scores = []
                 for i, score in enumerate(similarity_scores):
                     job_title = job_titles[i]
                     is_tech_job = any(keyword in job_title for keyword in tech_keywords)
                     if is_tech_job and applicant_group == target_group:
                         adjusted_score = score * (1 - self.bias_factor)
                         adjusted_scores.append(adjusted_score)
                     else:
                         adjusted_scores.append(score)
                 similarity_scores = np.array(adjusted_scores)


        # Get top N*3 indices (get more for potential re-ranking)
        # Ensure we don't request more indices than available scores
        num_scores = len(similarity_scores)
        n_candidates = min(top_n * 3, num_scores)
        if n_candidates <= 0:
            return pd.DataFrame() # No scores or no jobs

        # Use argpartition for efficiency if num_scores is large, otherwise argsort is fine
        if num_scores > 1000:
             top_indices = np.argpartition(similarity_scores, -n_candidates)[-n_candidates:]
             top_indices = top_indices[np.argsort(similarity_scores[top_indices])][::-1] # Sort the top candidates
        else:
             top_indices = np.argsort(similarity_scores)[-n_candidates:][::-1]


        # --- Create initial recommendation dataframe ---
        # Ensure indices are valid
        valid_indices = [idx for idx in top_indices if idx < len(self.jobs_df)]
        if not valid_indices:
             print(f"Warning: No valid job indices found for applicant {applicant_id} after sorting.")
             return pd.DataFrame()

        recommendations = pd.DataFrame({
            'job_id': self.jobs_df.iloc[valid_indices]['job_id'].values,
            'similarity_score': similarity_scores[valid_indices],
            # Use .get() with default for potentially missing columns after preprocessing changes
            'job_title': self.jobs_df.iloc[valid_indices].get('job_title', pd.Series(['N/A'] * len(valid_indices))).values,
            'company': self.jobs_df.iloc[valid_indices].get('company', pd.Series(['N/A'] * len(valid_indices))).values,
            'salary_range': self.jobs_df.iloc[valid_indices].get('salary_range', pd.Series(['N/A'] * len(valid_indices))).values
        })

        # --- Apply DEBIASING (Mitigation) Step ---
        if debias and self.debiased_model and self.bias_mitigation_method:
            print(f"--- Applying debiasing ({self.bias_mitigation_method}) for applicant {applicant_id} ---")
            recommendations = self._apply_bias_mitigation(recommendations, applicant, top_n)
        else:
            # If not debiasing, just take the top N based on (potentially biased) scores
             recommendations = recommendations.head(top_n)


        return recommendations


    def analyze_bias(self, test_size=0.2, random_state=42, simulate_bias_in_analysis=False):
        """Analyze bias in recommendations across protected groups."""
        print(f"\nAnalyzing bias in recommendations... (Simulation during analysis: {simulate_bias_in_analysis})")

        if self.applicants_df is None or self.applicants_df.empty:
             print("ERROR: Applicant data not available for bias analysis.")
             return {}
        if not self.protected_attributes:
            print("No protected attributes identified. Skipping bias analysis.")
            return {}

        # Split data for analysis to avoid overfitting metrics to the whole dataset
        if len(self.applicants_df) * test_size < 1:
             print("Warning: Dataset too small for train/test split during bias analysis. Using all data.")
             test_applicants = self.applicants_df.copy()
        else:
             try:
                 _, test_applicants = train_test_split(
                     self.applicants_df, test_size=test_size, random_state=random_state,
                     stratify=self.applicants_df[self.bias_attribute] if self.bias_attribute in self.applicants_df.columns else None # Stratify if possible
                 )
             except ValueError as e:
                 print(f"Warning: Could not stratify split for bias analysis ({e}). Using random split.")
                 _, test_applicants = train_test_split(
                     self.applicants_df, test_size=test_size, random_state=random_state
                 )


        bias_metrics = {}
        all_recommendations_list = [] # For overall metrics

        # Analyze each protected attribute
        for attribute in self.protected_attributes:
            print(f"\nAnalyzing bias for attribute: {attribute}")
            if attribute not in test_applicants.columns:
                 print(f"  Attribute '{attribute}' not found in test set. Skipping.")
                 continue

            group_metrics = {}
            # Get unique groups present in the test set, handle NaNs gracefully
            groups = test_applicants[attribute].unique()
            groups = [g for g in groups if pd.notna(g)] # Filter out NaN groups

            # For each group, calculate recommendation statistics
            for group in groups:
                group_applicants = test_applicants[test_applicants[attribute] == group]

                # Skip if no applicants in this group
                if len(group_applicants) == 0:
                    print(f"  Skipping group '{group}': No applicants in test set.")
                    continue

                print(f"  Processing group: {group} (Size: {len(group_applicants)})")
                group_recommendations = []
                applicant_ids_in_group = group_applicants['applicant_id'].tolist()

                # Get recommendations for the group
                # Pass simulate_bias flag - use the raw (potentially biased) scores for initial analysis
                # If analyzing *after* mitigation, debias=True should be passed by the calling context
                is_post_mitigation_analysis = self.debiased_model # Check if mitigation has been applied
                debias_flag_for_get_recs = is_post_mitigation_analysis # Apply debiasing only if model is marked debiased

                for app_id in applicant_ids_in_group:
                     # IMPORTANT: When simulating, the 'get_recommendations' call needs to know
                     # whether to apply the raw simulation (simulate_bias=True, debias=False)
                     # or to apply the mitigation ON TOP of simulation (simulate_bias=True, debias=True)
                     # We pass the 'simulate_bias_in_analysis' flag to control the simulation part,
                     # and 'debias_flag_for_get_recs' to control the mitigation part.
                     # For the *initial* analysis: simulate=True (if desired), debias=False
                     # For the *post-mitigation* analysis: simulate=True (if desired), debias=True
                     recs = self.get_job_recommendations(
                         app_id,
                         top_n=5,
                         debias=debias_flag_for_get_recs # Apply mitigation only if model is debiased
                     )
                     # Note: get_job_recommendations internally checks self.simulate_bias,
                     # so we don't need to pass simulate_bias_in_analysis directly here,
                     # but the 'debias' flag controls whether the raw simulation or the mitigated output is used.

                     if recs is not None and not recs.empty:
                        group_recommendations.append(recs)
                        all_recommendations_list.append(recs) # Add to overall list

                # Skip if no recommendations generated for this group
                if not group_recommendations:
                    print(f"  Skipping group '{group}': No recommendations generated.")
                    group_metrics[group] = {
                        'avg_similarity': np.nan, 'unique_jobs': 0, 'sample_size': len(group_applicants), 'top_jobs': []
                    }
                    continue

                # Combine recommendations for the group
                combined_recs = pd.concat(group_recommendations, ignore_index=True)

                # Calculate metrics
                avg_score = combined_recs['similarity_score'].mean() if not combined_recs.empty else np.nan
                unique_jobs = combined_recs['job_id'].nunique() if not combined_recs.empty else 0
                top_jobs_counts = combined_recs['job_id'].value_counts()
                top_jobs = top_jobs_counts.head(5).index.tolist() if not combined_recs.empty else []

                # Store metrics
                group_metrics[group] = {
                    'avg_similarity': avg_score,
                    'unique_jobs': unique_jobs,
                    'sample_size': len(group_applicants),
                    'top_jobs': top_jobs
                }

            # Calculate fairness metrics only if there are multiple groups with data
            valid_groups = [g for g, m in group_metrics.items() if pd.notna(m['avg_similarity']) and m['sample_size'] > 0]

            if len(valid_groups) > 1:
                 # Statistical parity (using average similarity): Ratio of min avg score to max avg score
                 similarity_scores = [group_metrics[g]['avg_similarity'] for g in valid_groups]
                 max_score = np.max(similarity_scores)
                 min_score = np.min(similarity_scores)
                 statistical_parity = min_score / max_score if max_score > 0 else 1.0

                 # Diversity parity: Ratio of min unique jobs per user to max unique jobs per user
                 # We need unique jobs *per user* first
                 avg_unique_jobs_per_group = []
                 for group in valid_groups:
                     group_recs_list = [r for r, app in zip(group_recommendations, group_applicants.iterrows()) if app[1][attribute] == group] # Get recs only for this group
                     if not group_recs_list:
                         avg_unique_jobs_per_group.append(0)
                         continue
                     unique_counts = [df['job_id'].nunique() for df in group_recs_list]
                     avg_unique_jobs_per_group.append(np.mean(unique_counts) if unique_counts else 0)

                 if avg_unique_jobs_per_group:
                     max_unique = np.max(avg_unique_jobs_per_group)
                     min_unique = np.min(avg_unique_jobs_per_group)
                     diversity_ratio = min_unique / max_unique if max_unique > 0 else 1.0
                 else:
                     diversity_ratio = 1.0


                 # Store fairness metrics
                 bias_metrics[attribute] = {
                     'group_metrics': group_metrics,
                     'statistical_parity': statistical_parity,
                     'diversity_ratio': diversity_ratio,
                     'overall_fairness': (statistical_parity + diversity_ratio) / 2
                 }
            elif len(valid_groups) == 1 :
                 print(f"  Only one valid group ('{valid_groups[0]}') found for attribute '{attribute}'. Cannot calculate comparative fairness metrics.")
                 bias_metrics[attribute] = {
                     'group_metrics': group_metrics,
                     'statistical_parity': 1.0, # Default to fair if only one group
                     'diversity_ratio': 1.0,
                     'overall_fairness': 1.0
                 }
            else:
                 print(f"  No valid groups with recommendations found for attribute '{attribute}'. Cannot calculate fairness metrics.")
                 # Provide default structure
                 bias_metrics[attribute] = {
                     'group_metrics': group_metrics,
                     'statistical_parity': np.nan,
                     'diversity_ratio': np.nan,
                     'overall_fairness': np.nan
                 }

        # Store overall fairness metrics
        self.fairness_metrics = bias_metrics

        # Print summary
        print("\n--- Bias Analysis Summary ---")
        for attribute, metrics in bias_metrics.items():
            print(f"\nMetrics for {attribute}:")
            if pd.notna(metrics['statistical_parity']):
                print(f"  Statistical Parity (Similarity): {metrics['statistical_parity']:.4f}")
                print(f"  Diversity Ratio (Unique Jobs per User): {metrics['diversity_ratio']:.4f}")
                print(f"  Overall Fairness: {metrics['overall_fairness']:.4f}")
            else:
                print("  Fairness metrics could not be calculated.")

            print("\n  Group-specific metrics:")
            if metrics['group_metrics']:
                 for group, group_mets in metrics['group_metrics'].items():
                     avg_sim_str = f"{group_mets['avg_similarity']:.4f}" if pd.notna(group_mets['avg_similarity']) else "N/A"
                     print(f"    {group} (n={group_mets['sample_size']}): "
                           f"Avg Similarity={avg_sim_str}, "
                           f"Total Unique Jobs Rec'd={group_mets['unique_jobs']}")
            else:
                 print("    No group metrics available.")
        print("--- End Bias Analysis Summary ---")

        return bias_metrics

    def implement_bias_mitigation(self, method='re_ranking'):
        """Implement bias mitigation techniques"""
        print(f"\nImplementing bias mitigation using: {method}...")

        if method == 're_ranking':
            # Re-ranking is applied at inference time via _apply_bias_mitigation
            self.bias_mitigation_method = 're_ranking'
            print("Re-ranking strategy selected. Will be applied during recommendation generation when debias=True.")
        # elif method == 'fair_representation': # Placeholder for future methods
        #     # This would require retraining or modifying the vectors/model itself
        #     print("Fair representation learning is more complex and not fully implemented here.")
        #     # Example: Could involve adversarial training or projecting vectors
        #     # For now, just set the method name
        #     self.bias_mitigation_method = 'fair_representation'
        #     # self._implement_fair_representation() # Call hypothetical implementation
        else:
            print(f"ERROR: Unknown bias mitigation method: {method}")
            return None # Indicate failure or no change

        # Mark model as debiased - this flag controls application in get_recommendations
        self.debiased_model = True
        print(f"Model set to apply '{self.bias_mitigation_method}' mitigation when debias=True.")

        # Re-analyze bias *after* marking the model for debiasing
        # The analyze_bias function will now call get_job_recommendations with debias=True
        print("\nAnalyzing bias AFTER enabling mitigation strategy:")
        # Pass the same simulation flag used initially to ensure a fair comparison baseline
        post_mitigation_metrics = self.analyze_bias(simulate_bias_in_analysis=self.simulate_bias)

        # Compare before and after (if initial metrics exist)
        if hasattr(self, 'initial_fairness_metrics'):
             self._compare_fairness_metrics(self.initial_fairness_metrics, post_mitigation_metrics)
        else:
             print("Initial fairness metrics not found for comparison.")


        return post_mitigation_metrics


    def _apply_bias_mitigation(self, recommendations, applicant, top_n):
        """Apply the selected bias mitigation technique at inference time."""
        if self.bias_mitigation_method == 're_ranking':
            # Apply re-ranking for diversity/fairness
            return self._apply_re_ranking(recommendations, applicant, top_n)
        # elif self.bias_mitigation_method == 'fair_representation':
        #     # If fair representation was applied during training/vector modification,
        #     # no extra step might be needed here, just return top N.
        #     return recommendations.head(top_n)
        else:
            # If no known method or not debiased, return original top N
             print("Warning: Debiasing requested, but no valid mitigation method set or active. Returning top N.")
             return recommendations.head(top_n)

    def _apply_re_ranking(self, recommendations, applicant, top_n):
        """
        Apply re-ranking for diversity and potentially fairness.
        This is a simplified Maximum Marginal Relevance (MMR)-like approach focusing on diversity.
        """
        if recommendations.empty:
            return recommendations

        # Lambda for MMR trade-off (0 = pure diversity, 1 = pure relevance)
        lambda_param = 0.7 # Balance between relevance and diversity

        # Use a slightly larger pool of candidates if available
        candidate_recs = recommendations.copy() # Already has more than top_n items
        final_recs_list = []
        candidate_indices = list(candidate_recs.index)

        # Select the first item (highest similarity)
        if not candidate_indices: return pd.DataFrame() # Should not happen if recommendations wasn't empty

        first_idx = candidate_indices.pop(0)
        final_recs_list.append(candidate_recs.loc[first_idx])

        # Iteratively select remaining items
        while len(final_recs_list) < top_n and candidate_indices:
            best_score = -np.inf
            best_idx_to_add = -1

            # Calculate score for each remaining candidate
            for idx in candidate_indices:
                candidate_job = candidate_recs.loc[idx]
                relevance_score = candidate_job['similarity_score']

                # Calculate diversity score (max similarity to already selected items)
                diversity_penalty = 0
                max_sim_to_selected = 0
                candidate_vector = self.job_vectors[self.jobs_df[self.jobs_df['job_id'] == candidate_job['job_id']].index[0]] # Get vector for this job

                for selected_job_df in final_recs_list:
                     selected_vector = self.job_vectors[self.jobs_df[self.jobs_df['job_id'] == selected_job_df['job_id']].index[0]]
                     sim = cosine_similarity(candidate_vector, selected_vector)[0, 0]
                     max_sim_to_selected = max(max_sim_to_selected, sim)

                diversity_penalty = max_sim_to_selected

                # MMR Score = lambda * Relevance - (1 - lambda) * DiversityPenalty
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx_to_add = idx

            # Add the best candidate found in this iteration
            if best_idx_to_add != -1:
                final_recs_list.append(candidate_recs.loc[best_idx_to_add])
                candidate_indices.remove(best_idx_to_add)
            else:
                # Should not happen unless all remaining candidates are identical
                break

        if not final_recs_list:
            return pd.DataFrame()

        final_recommendations = pd.DataFrame(final_recs_list)
        # Optional: Sort final list by original similarity or keep MMR order
        # final_recommendations = final_recommendations.sort_values('similarity_score', ascending=False)

        return final_recommendations.head(top_n) # Ensure exactly top_n are returned


    def _compare_fairness_metrics(self, before_metrics, after_metrics):
        """Compare fairness metrics before and after bias mitigation"""
        print("\n--- Bias Mitigation Results Comparison ---")
        print("=" * 50)

        if not before_metrics:
             print("No 'before' metrics to compare.")
             return
        if not after_metrics:
             print("No 'after' metrics to compare.")
             return


        for attribute in before_metrics.keys():
            if attribute not in after_metrics:
                print(f"\nAttribute: {attribute} (Only present in 'before' metrics)")
                continue

            print(f"\nAttribute: {attribute}")
            print("-" * 30)

            before = before_metrics[attribute]
            after = after_metrics[attribute]

            metrics_to_compare = ['statistical_parity', 'diversity_ratio', 'overall_fairness']
            labels = {'statistical_parity': 'Statistical Parity', 'diversity_ratio': 'Diversity Ratio', 'overall_fairness': 'Overall Fairness'}

            for metric in metrics_to_compare:
                before_val = before.get(metric, np.nan)
                after_val = after.get(metric, np.nan)

                if pd.isna(before_val) or pd.isna(after_val):
                    print(f"  {labels[metric]}: Before={before_val:.4f}, After={after_val:.4f} (Cannot compare NAs)")
                    continue

                change = after_val - before_val
                change_percent = (change / before_val) * 100 if before_val != 0 else 0
                indicator = '+' if change > 0 else '' # Show + for improvement

                print(f"  {labels[metric]:<20}: {before_val:.4f} -> {after_val:.4f} "
                      f"({indicator}{change:+.4f} / {indicator}{change_percent:+.1f}%)")

        print("=" * 50)


    def visualize_bias(self, metrics_to_plot=None, title_suffix=""):
        """Visualize bias metrics."""
        if metrics_to_plot is None:
            metrics_to_plot = self.fairness_metrics

        if not metrics_to_plot:
            print("No fairness metrics available to visualize.")
            return

        num_attributes = len(metrics_to_plot)
        if num_attributes == 0:
            print("No attributes found in fairness metrics.")
            return

        plt.figure(figsize=(15, 5 * num_attributes))
        plot_index = 1

        print(f"\nVisualizing Bias Metrics{title_suffix}")

        for attribute, metrics in metrics_to_plot.items():
            if not metrics or 'group_metrics' not in metrics or not metrics['group_metrics']:
                print(f"  Skipping visualization for {attribute}: No group metrics found.")
                continue

            group_data = metrics['group_metrics']
            groups = list(group_data.keys())
            
            # Filter out groups with NaN data for plotting
            valid_groups_sim = [g for g in groups if pd.notna(group_data[g].get('avg_similarity'))]
            valid_groups_div = [g for g in groups if pd.notna(group_data[g].get('unique_jobs'))]

            # Plot Average Similarity
            if valid_groups_sim:
                plt.subplot(num_attributes, 2, plot_index)
                avg_similarities = [group_data[g]['avg_similarity'] for g in valid_groups_sim]
                try:
                    sns.barplot(x=valid_groups_sim, y=avg_similarities)
                    plt.title(f"Avg Similarity by {attribute}{title_suffix}")
                    plt.ylabel("Avg Similarity Score")
                    plt.xticks(rotation=30, ha='right')
                    plt.ylim(bottom=0) # Ensure y-axis starts at 0
                except Exception as e:
                    print(f"Error plotting similarity for {attribute}: {e}")
                plot_index += 1
            else:
                 plot_index += 1 # Increment even if subplot is skipped

            # Plot Unique Jobs Recommended (Total)
            if valid_groups_div:
                plt.subplot(num_attributes, 2, plot_index)
                unique_jobs = [group_data[g]['unique_jobs'] for g in valid_groups_div]
                try:
                    sns.barplot(x=valid_groups_div, y=unique_jobs)
                    plt.title(f"Total Unique Jobs Rec'd by {attribute}{title_suffix}")
                    plt.ylabel("Num Unique Jobs")
                    plt.xticks(rotation=30, ha='right')
                    plt.ylim(bottom=0) # Ensure y-axis starts at 0
                except Exception as e:
                    print(f"Error plotting unique jobs for {attribute}: {e}")
                plot_index += 1
            else:
                 plot_index += 1 # Increment even if subplot is skipped


        plt.suptitle(f"Bias Analysis{title_suffix}", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
        plt.show()


    def visualize_bias_comparison(self, before_metrics, after_metrics):
        """Plot comparison of fairness metrics before and after bias mitigation"""
        if not before_metrics or not after_metrics:
             print("Cannot visualize comparison: Missing 'before' or 'after' metrics.")
             return

        attributes = list(before_metrics.keys())
        if not attributes or not any(attr in after_metrics for attr in attributes):
             print("Cannot visualize comparison: No common attributes found in metrics.")
             return

        metrics_to_plot = ['statistical_parity', 'diversity_ratio', 'overall_fairness']
        metric_labels = ['Statistical Parity', 'Diversity Ratio', 'Overall Fairness']

        # Filter attributes present in both and metrics that exist
        common_attributes = [attr for attr in attributes if attr in after_metrics]
        plot_data = {metric: {'labels': [], 'before': [], 'after': []} for metric in metrics_to_plot}

        for attr in common_attributes:
             valid_metric_found_for_attr = False
             for metric in metrics_to_plot:
                 before_val = before_metrics[attr].get(metric)
                 after_val = after_metrics[attr].get(metric)
                 if pd.notna(before_val) and pd.notna(after_val):
                     plot_data[metric]['before'].append(before_val)
                     plot_data[metric]['after'].append(after_val)
                     valid_metric_found_for_attr = True # Mark that this metric is valid for this attribute

             if valid_metric_found_for_attr:
                 # Add the attribute label only if at least one metric was valid for it
                 for metric in metrics_to_plot:
                    # Check again if this specific metric had data for this attribute before adding label
                    if pd.notna(before_metrics[attr].get(metric)) and pd.notna(after_metrics[attr].get(metric)):
                        if attr not in plot_data[metric]['labels']: # Ensure label added only once per metric
                           plot_data[metric]['labels'].append(attr)


        num_metrics_to_plot = len([m for m in metrics_to_plot if plot_data[m]['labels']]) # Count metrics that actually have data
        if num_metrics_to_plot == 0:
            print("No comparable valid fairness metrics found across attributes to plot.")
            return

        plt.figure(figsize=(6 * num_metrics_to_plot, 5))
        plot_idx = 1

        for i, metric in enumerate(metrics_to_plot):
            if not plot_data[metric]['labels']: # Skip if no data for this metric
                continue

            ax = plt.subplot(1, num_metrics_to_plot, plot_idx)
            plot_idx += 1

            labels = plot_data[metric]['labels']
            before_values = plot_data[metric]['before']
            after_values = plot_data[metric]['after']

            x = np.arange(len(labels))
            width = 0.35

            rects1 = ax.bar(x - width/2, before_values, width, label='Before Mitigation', color='skyblue')
            rects2 = ax.bar(x + width/2, after_values, width, label='After Mitigation', color='lightcoral')

            ax.set_ylabel('Score (Higher is Better)')
            ax.set_title(metric_labels[i])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.legend()
            ax.set_ylim(0, max(1.1, np.nanmax(before_values + after_values) * 1.1)) # Adjust ylim dynamically
            ax.bar_label(rects1, padding=3, fmt='%.2f')
            ax.bar_label(rects2, padding=3, fmt='%.2f')

        plt.suptitle("Fairness Metrics Before vs. After Mitigation", fontsize=16, y=1.03)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()


    def save_model(self, filepath="job_recommender_model.pkl"):
        """Save model components for future use"""
        print(f"Saving model to {filepath}...")
        try:
            model_components = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'job_vectors': self.job_vectors,
                'jobs_df_columns': self.jobs_df.columns.tolist() if self.jobs_df is not None else None, # Save columns for consistency check on load
                'jobs_df_ids': self.jobs_df['job_id'].tolist() if self.jobs_df is not None else None, # Save only IDs to reduce size? Or full df? Let's stick to columns/ids for now.
                'debiased_model': self.debiased_model,
                'bias_mitigation_method': getattr(self, 'bias_mitigation_method', None),
                'protected_attributes': self.protected_attributes,
                'fairness_metrics': self.fairness_metrics, # Save last calculated metrics
                # Save simulation parameters used when saving
                'simulate_bias_config': {
                     'simulate_bias': self.simulate_bias,
                     'bias_factor': self.bias_factor,
                     'bias_attribute': self.bias_attribute,
                     'bias_target_keywords': self.bias_target_keywords
                }
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_components, f)

            print(f"Model saved successfully to {filepath}")

        except Exception as e:
            print(f"Error saving model: {e}")


    def load_model(self, filepath="job_recommender_model.pkl"):
        """Load model components from file. Requires jobs_df to be loaded separately or passed."""
        print(f"Loading model from {filepath}...")
        if not os.path.exists(filepath):
             print(f"ERROR: Model file not found at {filepath}")
             return False

        try:
            with open(filepath, 'rb') as f:
                model_components = pickle.load(f)

            # Load core components
            self.tfidf_vectorizer = model_components.get('tfidf_vectorizer')
            self.job_vectors = model_components.get('job_vectors')
            saved_jobs_columns = model_components.get('jobs_df_columns')
            # self.jobs_df = model_components['jobs_df'] # Avoid loading full potentially large df

            # Load state
            self.debiased_model = model_components.get('debiased_model', False)
            self.bias_mitigation_method = model_components.get('bias_mitigation_method')
            self.protected_attributes = model_components.get('protected_attributes', [])
            self.fairness_metrics = model_components.get('fairness_metrics', {})

            # Load simulation config used at save time (optional, could reset on load)
            sim_config = model_components.get('simulate_bias_config', {})
            self.simulate_bias = sim_config.get('simulate_bias', False)
            self.bias_factor = sim_config.get('bias_factor', 0.3)
            self.bias_attribute = sim_config.get('bias_attribute', 'gender')
            self.bias_target_keywords = sim_config.get('bias_target_keywords', ['manager', 'senior', 'lead', 'director'])


            # --- Crucial Check ---
            # The loaded model relies on the current self.jobs_df having the same structure
            # (especially the same jobs in the same order) as when the model was saved.
            if self.jobs_df is None:
                 print("WARNING: jobs_df is not loaded. Recommendations might fail or be incorrect.")
                 print("         Load and preprocess job data before using the loaded model.")
                 # We can't fully validate without the jobs_df loaded first.
            elif saved_jobs_columns is not None:
                 if list(self.jobs_df.columns) != saved_jobs_columns:
                     print("WARNING: Columns of currently loaded jobs_df do not match saved model.")
                 if self.job_vectors is not None and self.job_vectors.shape[0] != len(self.jobs_df):
                     print(f"WARNING: Loaded job vectors length ({self.job_vectors.shape[0]}) does not match current jobs_df length ({len(self.jobs_df)}).")
                     print("         Ensure the correct jobs CSV was loaded and preprocessed identically.")


            if self.tfidf_vectorizer is None or self.job_vectors is None:
                 print("WARNING: TF-IDF vectorizer or job vectors missing in loaded model.")
                 return False


            print(f"Model loaded successfully from {filepath}")
            print(f"  Loaded state: debiased={self.debiased_model}, method={self.bias_mitigation_method}")
            print(f"  Loaded simulation config: simulate={self.simulate_bias}, factor={self.bias_factor}, attr={self.bias_attribute}")
            return True

        except ModuleNotFoundError as e:
             print(f"ERROR loading model: Missing module - {e}. Ensure all required libraries (sklearn, pandas, etc.) are installed in the environment.")
             return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


    def evaluate_recommendations(self, test_size=0.2, k=5, random_state=42, eval_debiased=False):
        """Evaluate recommendation quality using offline metrics."""
        print(f"\nEvaluating recommendation system quality... (Evaluating {'Debiased' if eval_debiased else 'Original/Biased'})")

        if self.applicants_df is None or self.jobs_df is None or self.job_vectors is None:
             print("ERROR: Required dataframes or model not available for evaluation.")
             return {}

        # Split data for evaluation
        if len(self.applicants_df) * test_size < 1:
             print("Warning: Dataset too small for train/test split during evaluation. Using all data.")
             test_applicants = self.applicants_df.copy()
        else:
             try:
                 _, test_applicants = train_test_split(
                     self.applicants_df, test_size=test_size, random_state=random_state,
                     stratify=self.applicants_df[self.bias_attribute] if self.bias_attribute in self.applicants_df.columns else None
                 )
             except ValueError as e:
                 print(f"Warning: Could not stratify split for evaluation ({e}). Using random split.")
                 _, test_applicants = train_test_split(
                     self.applicants_df, test_size=test_size, random_state=random_state
                 )

        # Evaluation metrics
        all_recommendations = []
        recommended_job_ids = set()
        total_similarity = 0
        num_recs_generated = 0

        # Get recommendations for test applicants
        for _, applicant in test_applicants.iterrows():
            # IMPORTANT: Pass the eval_debiased flag to get_job_recommendations
            recs = self.get_job_recommendations(
                applicant['applicant_id'],
                top_n=k,
                debias=eval_debiased # Control whether to get mitigated recommendations
            )
            if recs is not None and not recs.empty:
                all_recommendations.append(recs)
                recommended_job_ids.update(recs['job_id'].tolist())
                total_similarity += recs['similarity_score'].sum()
                num_recs_generated += len(recs)

        if not all_recommendations:
            print("No recommendations generated for evaluation set.")
            return {'coverage': 0, 'avg_list_diversity': 0, 'avg_item_similarity': 0}

        # 1. Coverage: Percentage of unique jobs recommended out of all jobs
        coverage = len(recommended_job_ids) / len(self.jobs_df) if len(self.jobs_df) > 0 else 0

        # 2. Diversity (Intra-list diversity): Average number of unique items per recommendation list
        #    This is different from the 'unique_jobs' in bias analysis which was total unique across a group.
        list_diversities = [recs['job_id'].nunique() for recs in all_recommendations]
        avg_list_diversity = np.mean(list_diversities) if list_diversities else 0

        # 3. Average Item Similarity: Average similarity score of all recommended items across all users
        avg_item_similarity = total_similarity / num_recs_generated if num_recs_generated > 0 else 0

        # Combine all metrics
        evaluation_metrics = {
            'coverage': coverage,
            'avg_list_diversity': avg_list_diversity, # Avg unique items per user list
            'avg_item_similarity': avg_item_similarity, # Avg score of items in lists
        }

        print("\n--- Recommendation System Evaluation Summary ---")
        print(f"  Coverage: {coverage:.4f} ({len(recommended_job_ids)} unique jobs recommended / {len(self.jobs_df)} total jobs)")
        print(f"  Avg. List Diversity: {avg_list_diversity:.4f} (Avg. unique jobs per applicant's top-{k} list)")
        print(f"  Avg. Item Similarity: {avg_item_similarity:.4f} (Avg. similarity score of recommended jobs)")
        print("--- End Evaluation Summary ---")

        return evaluation_metrics

# --- End of Class Definition ---


# Example of running the full pipeline
def run_job_recommender_pipeline(
    jobs_path="Book1.csv",
    applicants_path="ds.csv",
    simulate_bias=True, # <<< Set to True to demonstrate bias
    bias_factor=0.4,    # <<< Adjust strength of simulated bias (e.g., 0.4 = 40% penalty)
    bias_attribute='gender', # <<< Choose 'gender' or 'age_group' (if available)
    mitigation_method='re_ranking'
    ):
    """Run the complete job recommender pipeline with bias simulation and mitigation."""

    print("--- Starting Job Recommender Pipeline ---")
    print(f"Settings: Simulate Bias={simulate_bias}, Factor={bias_factor}, Attribute={bias_attribute}, Mitigation={mitigation_method}")

    # Initialize the recommender system with simulation settings
    recommender = JobRecommenderSystem(
        simulate_bias=simulate_bias,
        bias_factor=bias_factor,
        bias_attribute=bias_attribute
    )

    # Load data
    jobs_df, applicants_df = recommender.load_data(jobs_path, applicants_path)
    if jobs_df is None or applicants_df is None:
         print("Failed to load data. Exiting pipeline.")
         return None

    # Preprocess data
    recommender.preprocess_job_data()
    recommender.preprocess_applicant_data()
    # Re-check if bias attribute is valid after preprocessing
    if simulate_bias and recommender.bias_attribute not in recommender.protected_attributes:
         print(f"Pipeline Warning: Chosen bias attribute '{recommender.bias_attribute}' not usable. Disabling simulation.")
         recommender.simulate_bias = False # Turn off simulation if attribute isn't valid


    # Build recommendation model
    if recommender.build_recommendation_model() is None:
        print("Failed to build recommendation model. Exiting pipeline.")
        return None


    # --- Initial Analysis (Potentially Biased) ---
    print("\n--- Phase 1: Initial Analysis (Before Mitigation) ---")
    # Evaluate initial recommendations (reflecting simulated bias if active)
    initial_eval = recommender.evaluate_recommendations(eval_debiased=False) # Evaluate the raw output

    # Analyze bias in initial recommendations (with simulation if active)
    # The simulate_bias flag inside the recommender controls the get_recommendations behavior here
    initial_bias = recommender.analyze_bias(simulate_bias_in_analysis=recommender.simulate_bias)
    recommender.initial_fairness_metrics = initial_bias # Store for comparison

    # Visualize initial bias
    recommender.visualize_bias(metrics_to_plot=initial_bias, title_suffix=" (Before Mitigation)")


    # --- Bias Mitigation ---
    print("\n--- Phase 2: Bias Mitigation ---")
    post_mitigation_bias = recommender.implement_bias_mitigation(method=mitigation_method)
    if post_mitigation_bias is None:
         print("Bias mitigation step failed or was skipped.")
         # Optionally decide whether to proceed or exit
    else:
        # Visualize comparison if both metrics are available
        recommender.visualize_bias_comparison(initial_bias, post_mitigation_bias)

        # --- Post-Mitigation Analysis ---
        print("\n--- Phase 3: Post-Mitigation Analysis ---")
        # Evaluate recommendations after bias mitigation is enabled
        post_mitigation_eval = recommender.evaluate_recommendations(eval_debiased=True) # Evaluate the mitigated output

        # The bias analysis was already run inside implement_bias_mitigation,
        # but we can visualize the 'after' state standalone if needed.
        recommender.visualize_bias(metrics_to_plot=post_mitigation_bias, title_suffix=" (After Mitigation)")


    # --- Save Model ---
    print("\n--- Phase 4: Saving Model ---")
    recommender.save_model("job_recommender_model.pkl")

    # --- Example Inference ---
    print("\n--- Phase 5: Example Inference (Post Mitigation) ---")
    if not recommender.applicants_df.empty:
        try:
            example_applicant_id = recommender.applicants_df['applicant_id'].iloc[0]
            print(f"\nGetting recommendations for Applicant ID: {example_applicant_id}")
            # Get debiased recommendations
            recommendations = recommender.get_job_recommendations(example_applicant_id, top_n=5, debias=True)
            if not recommendations.empty:
                 print("Top 5 Debiased Recommendations:")
                 print(recommendations.to_string())
            else:
                 print("No recommendations generated for the example applicant.")

            # Optional: Show non-debiased recommendations for comparison
            print(f"\nGetting non-debiased recommendations for Applicant ID: {example_applicant_id} (for comparison)")
            raw_recommendations = recommender.get_job_recommendations(example_applicant_id, top_n=5, debias=False)
            if not raw_recommendations.empty:
                 print("Top 5 Original/Biased Recommendations:")
                 print(raw_recommendations.to_string())
            else:
                 print("No non-debiased recommendations generated.")

        except IndexError:
            print("Could not get an example applicant ID.")
        except Exception as e:
            print(f"Error during example inference: {e}")
    else:
        print("No applicants loaded, skipping example inference.")

    print("\n--- Job Recommender Pipeline Finished ---")
    return recommender


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the CSV files are in the same directory as the script, or provide full paths
    # jobs_file = "Book1.csv"
    # applicants_file = "ds.csv"

    jobs_file = "Book1-small.csv"
    applicants_file = "ds-small.csv"

    if not os.path.exists(jobs_file):
         print(f"FATAL ERROR: Jobs CSV file '{jobs_file}' not found.")
    elif not os.path.exists(applicants_file):
         print(f"FATAL ERROR: Applicants CSV file '{applicants_file}' not found.")
    else:
        # Run the pipeline with bias simulation enabled for demonstration
        trained_recommender = run_job_recommender_pipeline(
            jobs_path=jobs_file,
            applicants_path=applicants_file,
            simulate_bias=True,   # <<< DEMO: Enable bias simulation
            bias_factor=0.5,     # <<< DEMO: Set bias strength (e.g., 50% penalty)
            bias_attribute='gender', # <<< DEMO: Bias based on gender (ensure 'Sex' column exists in ds.csv)
                                     #      Change to 'age_group' to test age bias (ensure 'Age' column exists)
            mitigation_method='re_ranking' # Use re-ranking mitigation
        )

        # Example of loading the saved model later (requires data to be loaded again)
        # print("\n--- Example: Loading and Using Saved Model ---")
        # new_recommender = JobRecommenderSystem()
        # # Load data again (essential for context)
        # new_recommender.load_data(jobs_file, applicants_file)
        # if new_recommender.jobs_df is not None and new_recommender.applicants_df is not None:
        #     new_recommender.preprocess_job_data() # Must preprocess jobs the same way
        #     new_recommender.preprocess_applicant_data() # Need applicant data for inference
        #     # Now load the model state
        #     if new_recommender.load_model("job_recommender_model.pkl"):
        #         if not new_recommender.applicants_df.empty:
        #             example_id = new_recommender.applicants_df['applicant_id'].iloc[5] # Different applicant
        #             print(f"\nGetting recommendations using loaded model for Applicant ID: {example_id}")
        #             loaded_recs = new_recommender.get_job_recommendations(example_id, top_n=5, debias=True) # Use debias=True as saved model state dictates
        #             print(loaded_recs)
        #         else:
        #              print("No applicants loaded in the new instance.")

