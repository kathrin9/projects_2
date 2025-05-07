This project involved the analysis and modeling of a real estate dataset obtained from TechPro Academy. The objective was to predict house prices using multiple linear regression and compare it with a decision tree regression model.

The data, originally stored in an SQLite database, was extracted and merged from three different tables: housing, city demographics, and energy classification. After assembling the final dataset, an extensive preprocessing phase was conducted. This included removing irrelevant columns, handling missing values (e.g., using median or mode imputation), eliminating duplicate rows, and correcting data types. Categorical features such as city and energy class were encoded numerically, and outliers were examined through visualizations.

A correlation analysis was performed to identify which features most significantly influenced house prices. Multiple Linear Regression was then applied using the most relevant features. The model achieved an R² score of approximately 0.90 on the testing set and 0.88 on the training set, indicating strong predictive performance with minimal overfitting.

Additionally, a Decision Tree Regressor was trained and optimized using hyperparameter tuning (e.g., max_depth and min_samples_split). Although the tree model reached a reasonable level of accuracy, the linear regression consistently outperformed it in terms of both RMSE and R² metrics.

To validate model stability and generalizability, 10-fold cross-validation was also employed, yielding an average RMSE that further confirmed the robustness of the linear regression model.

In conclusion, the linear regression model proved to be an effective and interpretable approach for predicting house prices in this dataset, outperforming the more complex decision tree model in both accuracy and generalizability.
