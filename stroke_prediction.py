import pandas as pd

def user_input_prediction(model, feature_selector, label_encoder, categorical_cols):
    print("\n--- Stroke Prediction Tool ---")
    
    # Define expected input features and their descriptions for user guidance
    input_features = {
        'gender': "Gender (0: Female, 1: Male, 2: Other)",
        'age': "Age (in years, e.g., 45.5)",
        'hypertension': "Hypertension (0: No, 1: Yes)",
        'heart_disease': "Heart Disease (0: No, 1: Yes)",
        'ever_married': "Ever Married (0: No, 1: Yes)",
        'work_type': "Work Type (0: Children, 1: Govt_job, 2: Never_worked, 3: Private, 4: Self-employed)",
        'Residence_type': "Residence Type (0: Rural, 1: Urban)",
        'avg_glucose_level': "Average Glucose Level (e.g., 95.0)",
        'bmi': "BMI (Body Mass Index, e.g., 24.5)",
        'smoking_status': "Smoking Status (0: Unknown, 1: Formerly Smoked, 2: Never Smoked, 3: Smokes)"
    }
    
    user_input = {}
    
    # Collect and validate user input
    for feature, description in input_features.items():
        while True:
            value = input(f"Enter value for {feature} ({description}): ")
            if value.replace('.', '', 1).isdigit():  # Check if value is numeric
                user_input[feature] = float(value) if '.' in value else int(value)
                break
            elif value.isdigit() == False and feature in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
                user_input[feature] = value  # Account for strings in categorical data
                break
            else:
                print(f"Invalid input for {feature}. Please follow the guidelines: {description}")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical variables to match model training
    for col in categorical_cols:
        if col in input_df.columns:
            try:
                # Check if label_encoder is able to handle unseen labels
                input_df[col] = label_encoder.transform(input_df[col].astype(str))
            except ValueError as e:
                print(f"Encoding issue for column {col}: {e}. Trying to handle unseen labels.")
                # Handle unseen labels by adding them to the label encoder
                label_encoder.fit(input_df[col].astype(str))  # Fit the encoder again with new labels
                input_df[col] = label_encoder.transform(input_df[col].astype(str))
    
    # Select the same features used for training
    try:
        input_selected = feature_selector.transform(input_df)
    except ValueError as e:
        print("Error in feature selection:", e)
        return
    
    
    # Predict using the provided model
    prediction = model.predict(input_selected)
    
    # Output result
    print("\n--- Prediction Result ---")
    print("Stroke Risk:", "Yes" if prediction[0] == 1 else "No")

# Example call to the function (make sure to define `linear_svm`, `feature_selector`, `label_encoder`, and `categorical_cols`):
# user_input_prediction(linear_svm, feature_selector, label_encoder, categorical_cols)
