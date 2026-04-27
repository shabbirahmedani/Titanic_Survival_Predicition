# ============================================================================
# TITANIC SURVIVAL PREDICTION - DECISION TREE MODEL
# Copy each block into separate Jupyter notebook cells
# ============================================================================

# ============================================================================
# BLOCK 1: Import Libraries
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# ============================================================================
# BLOCK 2: Load Training Dataset
# ============================================================================
train_df = pd.read_csv('train.csv')
print("Training dataset loaded!")
print(f"Shape: {train_df.shape}")
print("\nFirst few rows:")
train_df.head()

# ============================================================================
# BLOCK 3: Load Test Dataset
# ============================================================================
test_df = pd.read_csv('test.csv')
print("Test dataset loaded!")
print(f"Shape: {test_df.shape}")
print("\nFirst few rows:")
test_df.head()

# ============================================================================
# BLOCK 4: Check Missing Values
# ============================================================================
print("Missing values in training data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())

# ============================================================================
# BLOCK 5: Data Preprocessing - Handle Missing Values
# ============================================================================
# Create a copy of training data for preprocessing
train_processed = train_df.copy()
test_processed = test_df.copy()

# Fill missing Age with median
train_processed['Age'].fillna(train_processed['Age'].median(), inplace=True)
test_processed['Age'].fillna(test_processed['Age'].median(), inplace=True)

# Fill missing Embarked with mode (most common)
train_processed['Embarked'].fillna(train_processed['Embarked'].mode()[0], inplace=True)
test_processed['Embarked'].fillna(test_processed['Embarked'].mode()[0], inplace=True)

# Fill missing Fare with median
train_processed['Fare'].fillna(train_processed['Fare'].median(), inplace=True)
test_processed['Fare'].fillna(test_processed['Fare'].median(), inplace=True)

# Fill missing Cabin with 'Unknown'
train_processed['Cabin'].fillna('Unknown', inplace=True)
test_processed['Cabin'].fillna('Unknown', inplace=True)

print("Missing values handled!")

# ============================================================================
# BLOCK 6: Feature Engineering
# ============================================================================
# Convert Sex to numerical (male=0, female=1)
train_processed['Sex'] = train_processed['Sex'].map({'male': 0, 'female': 1})
test_processed['Sex'] = test_processed['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numerical
train_processed['Embarked'] = train_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_processed['Embarked'] = test_processed['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Create family size feature
train_processed['FamilySize'] = train_processed['SibSp'] + train_processed['Parch'] + 1
test_processed['FamilySize'] = test_processed['SibSp'] + test_processed['Parch'] + 1

# Create is alone feature
train_processed['IsAlone'] = (train_processed['FamilySize'] == 1).astype(int)
test_processed['IsAlone'] = (test_processed['FamilySize'] == 1).astype(int)

print("Feature engineering completed!")

# ============================================================================
# BLOCK 7: Prepare Features for Training
# ============================================================================
# Select features for training
# We'll use: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, FamilySize, IsAlone
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

# Prepare training data
X_train = train_processed[features]
y_train = train_processed['Survived']

# Prepare test data
X_test = test_processed[features]

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"\nFeatures used: {features}")

# ============================================================================
# BLOCK 8: Split Data for Validation
# ============================================================================
# Split training data to evaluate model performance
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training split: {X_train_split.shape}")
print(f"Validation split: {X_val_split.shape}")

# ============================================================================
# BLOCK 9: Train Decision Tree Model
# ============================================================================
# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# Train on full training dataset
dt_model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")
print(f"Model depth: {dt_model.tree_.max_depth}")
print(f"Number of features: {dt_model.n_features_in_}")

# ============================================================================
# BLOCK 10: Evaluate Model Performance
# ============================================================================
# Evaluate model on validation set
y_val_pred = dt_model.predict(X_val_split)
accuracy = accuracy_score(y_val_split, y_val_pred)


print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val_split, y_val_pred))
cm = confusion_matrix(y_val_split, y_val_pred)  
print("Confusion Matrix:")
print(cm)

# ============================================================================
# BLOCK 11: Make Predictions on Test Dataset
# ============================================================================
# Make predictions on test dataset
test_predictions = dt_model.predict(X_test)

print(f"Predictions made for {len(test_predictions)} passengers")
print(f"Survived (1): {np.sum(test_predictions == 1)}")
print(f"Did not survive (0): {np.sum(test_predictions == 0)}")

# ============================================================================
# BLOCK 12: Create Submission CSV File
# ============================================================================
# Create submission CSV file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
print("\nFirst 10 predictions:")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")

# ============================================================================
# BLOCK 13: Verify Submission File
# ============================================================================
# Verify the submission file
verify_df = pd.read_csv('submission.csv')
print("Submission file verification:")
print(f"Shape: {verify_df.shape}")
print(f"Columns: {verify_df.columns.tolist()}")
print(f"\nSurvived value counts:")
print(verify_df['Survived'].value_counts())
print("\nFirst 20 rows:")
print(verify_df.head(20))

