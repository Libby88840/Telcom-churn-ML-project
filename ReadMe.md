![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

**Telecom churn prediction project CRISP- DM Approach using machine learning.**

**Problem statement.**

Using machine learning to predict which customers are likely to leave a service (churn).

**Project Overview.**

A predictive modeling pipeline built to identify telecom customers who are likely to churn. Using demographic, service usage, and billing data (~21 features), compared multiple models and identified the most effective one for real-world deployment.

**Data Acquisition and Preparation**

Used the data set for SyriaTel customer churn from Kaggle https://bit.ly/46kTkFG

The following was done to clean the data.

*   Dropped irrelevant or excessively identifiers, such as removed phone number to avoid unnecessary model complexity. Kept area code (low cardinality) and dropped state or optionally transformed it via grouping or one-hot encoding.
*   Converted target to binary by transforming churn from True/False into 1/0 for modeling purposes.
*   Encoded categorical features by transforming international plan and voice mail plan (yes/no) into binary 1/0 values. Applied label or one-hot encoding to remaining categorical (e.g., state, area code as needed).
*   Handled missing/infinite values by replacing any missing or infinite values across numeric columns like usage metrics (total day/eve/night/intl minutes, etc.), charges, and number vmail messages to prevent model errors.

**Why is all this nessesary?**

1\. Dropping identifiers like phone number removes noise; grouping state reduces cardinality Medium guide.

2\. Binary mapping of yes/no flags and encoding of area code/region ensures modeling-ready numeric data .

3\. Replacing NaNs/infinite values with medians avoids model errors and maintains robust distributions.

**Exploratory data Analysis.**

Checking for correlations on numeric features showed strong positive correlations.

**Usage vs. Charges:**

*   Very high correlation between total day minutes & total day charge; similarly between evening/night usage and their respective charges

**Cross-period usage coupling:**

*   Moderate positive correlation between day and evening minutes suggests that heavy daytime users are often heavy evening users.

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

**Checking how overall usage distributes differently for churners vs. non-churners.**

This comparison reveals behavioral patterns that indicate dissatisfaction or disengagement. By examining usage distribution, we can identify thresholds where decreased (or sometimes increased) activity signals a higher risk of churn.

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)

The density-histogram shows how overall usage distributes differently for churners vs. non-churners.

*   Churners tend to have higher total minutes, this may suggest that high usage leads to dissatisfaction or cost concerns.
*   Churners skew toward lower usage, it may indicate disengagement or low perceived value.

**Modeling**

Four models were trained and compared with the target variable as churn.

1\. Logistic Regression: A solid baseline for binary classification—interpretable, fast, and effective.

2\. Decision Tree & Random Forest: Captures nonlinear patterns, requires minimal preprocessing.

3\. Support Vector Machine (SVM): Great for complex decision boundaries with scaling.

4\. Gradient Boosting (XGBoost / LightGBM): Ensembles that often win in churn prediction benchmarks.

**Evaluation was done using:**

1\. Accuracy - Percentage of total correct predictions (TP + TN) ÷ (TP + TN + FP + FN).

2\. Precision - Precision - TP ÷ (TP + FP): how many predicted churns were correct.

3\. Recall (Sensitivity) - Recall = TP ÷ (TP + FN): how many actual churners were captured.

4\. F1-Score - Harmonic mean of Precision and Recall: F1 = 2 × (Precision × Recall) ÷ (Precision + Recall).

5\. ROC‑AUC - Plots true positive rate vs false positive rate across thresholds.

6\. Confusion Matrix - A table of True Positives, False Positives, False Negatives, and True Negatives.

7\. Cross validation…This helps us to evaluate how well a machine learning model will perform on unseen data.

**Model Evaluation and Performance**

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

**ROC-AUC Results Summary**

*   XGBoost emerged as the top performer, achieving the highest ROC-AUC scores, typically in the range of 0.87–0.88. This aligns with telecom churn research where XGBoost’s gradient boosting consistently captures subtle patterns in customer behavior.
*   Random Forest followed closely behind, with ROC-AUC values between 0.85–0.87, showcasing strong generalization through ensemble learning and feature randomness.
*   Decision Tree performed respectably, usually scoring around 0.74–0.80. While interpretable, its performance is more volatile due to a higher tendency to overfit, especially without pruning or regularization.
*   Logistic Regression served as a dependable baseline model, consistently scoring between 0.80–0.82 in ROC-AUC. Though less powerful than ensemble methods, it provides transparency, speed, and a solid starting point for churn classification.

**Interpretation.**

Ensemble methods (Random Forest and XGBoost) outperform simpler models due to their ability to capture complex patterns and reduce overfitting. XGBoost, in particular, stands out as the top performer in this setting. Logistic regression remains a reliable baseline, valued for its interpretability and speed, but typically doesn’t match the predictive power of tree-based ensembles.

**Confusion Matrices**

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image015.png)

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image017.jpg)

**What was observed across models:**

~ High True Negatives (TN) were observed across all models, which is expected due to the natural class imbalance—with more non-churners than churners in the dataset.

**Model-Specific Observations:**

**Random Forest & Decision Tree**

~  Both models showed a strong balance between identifying churners (True Positives) and avoiding false alarms (False Positives).

~ This aligns with their structure—Decision Trees are sensitive to patterns, and Random Forests reduce variance, improving generalization.

**XGBoost**

~ Delivered the best true positive rate, meaning it correctly identified more actual churners than other models.

~ It’s especially effective at capturing subtle churn patterns often missed by simpler models.

**Logistic Regression**

~ Tended to have lower recall—missing some churners—but when it did predict churn, it was usually accurate (i.e., higher precision).

This makes it a reliable model when false positives are more costly than false negatives.

**Interpretation:**

~ The confusion matrices highlight each model’s trade-offs:

~ Tree-based models (Random Forest, XGBoost) provide strong recall—better at flagging churn risk early.

~ Logistic Regression is conservative, making fewer churn predictions but with greater confidence when it does.

**Choosing the right model depends on your business need:**

~ Want to reduce churn at all costs? → XGBoost or Random Forest.

~ Want fewer false alarms and simpler decisions? → Logistic Regression.

**Feature Importance**

![](file:///C:/Users/User/AppData/Local/Temp/msohtmlclip1/01/clip_image019.jpg)

 **Top predictors often included:**

*   Features related to billing, customer support, and service usage patterns are most influential.
*   High service usage (especially during the day), frequent support calls, and international plan activation are consistent churn signals.
*   These features help businesses prioritize interventions for at-risk customers.

These mirror industry findings where usage and service interaction metrics are key churn indicators.

 **Key Insights**

1\. XGBoost provided the best discrimination between churn and non-churn cases.

2\. Strongest predictors:

   - Total usage minutes

   - Number of customer service calls

   - Account tenure

   - Billing/charge metrics & international plan usage

3\. Confusion matrices show XGBoost and Random Forest balance catching churners while avoiding false alarms.

**Business Recommendations.**

1\. Deploy XGBoost for real-time churn scoring to enable early intervention.

2\. Prioritize high-risk customers high usage + frequent support calls.

3\.  Use retained models for targeted retention campaigns (offers, loyalty programs, etc.).

4\. Apply threshold optimization to match cost tolerance and business strategy.

5\. Supplement with interpretable models (Logistic Regression) for reports and stakeholder communication.

**Next Steps**

1.  Perform hyper parameter tuning (e.g., grid search) for final model optimization.

2\. Integrate SHAP explainability for transparent model insights.

3\. Develop a real-time scoring system and monitor ROI from retention campaigns.

 **Conclusion**

*   The churn models are highly accurate and reliable.
*   The business can now flag high-risk customers early and take action (e.g., personalized offers, loyalty programs).
*   Feature analysis helps prioritize factors to improve customer experience and retention.
*   Telecom churn studies often rely on a single dataset (like this), which may not reflect broader customer behaviors. Models trained on such data risk limited generalization and branch-specific bias.
*   Our dataset didn't include data like customer social network interactions or real-time behavior—features shown to boost model performance significantly (AUC increase from ~0.84 to 0.93 in some studies). This implies our models may miss crucial churn signals outside usage and billing metrics.