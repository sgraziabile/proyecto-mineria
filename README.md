# Gym Routine Difficulty Predictor

## 1. Project Overview
This project applies Machine Learning to predict the difficulty level (Beginner, Intermediate, Advanced) of fitness routines. By transforming raw, sequential workout logs into structured mathematical profiles, the project uncovers the underlying behavioral and physiological patterns that define different levels of gym-goers.

## 2. Data Engineering & Aggregation
### Original vs. Processed Dataset
- **Original Data:** Over 400,000 highly granular rows where each record represented a single exercise on a specific day of a multi-week program. 
- **Processed Data:** Through extensive data cleaning and groupby operations, the noise was distilled down into **1,850 unique, highly stable routine profiles**.

### Feature Engineering
To capture the true essence of a workout program, we engineered advanced periodization metrics, including:
- `avg_weekly_volume`: The total workload per week.
- `avg_intensity` & `max_intensity`: The effort relative to the lifter's max.
- `avg_exercises_per_day`: Identifying workout density.
- `intensity_variance_total`: Measuring the structural discipline and periodization of the program.

## 3. Modeling: Random Forest vs. XGBoost
- **The Challenge:** The dataset presented a severe class imbalance (Advanced routines comprised only ~4% of the data) and a high degree of feature overlap (a hard intermediate routine is mathematically nearly identical to an easy advanced routine).
- **Random Forest:** Struggled significantly with the minority class, achieving a precision of just 0.04 for Advanced routines. It resorted to casting a wide, messy net to capture the minority class, heavily damaging overall accuracy.
- **XGBoost:** Gradient boosting proved vastly superior for this imbalanced tabular data. It successfully isolated Advanced routines without needing synthetic data injections (like SMOTE), skyrocketing the minority precision to 0.50.

## 4. The Depth Trade-off: Baseline vs. Fine-Tuned XGBoost
- **Baseline XGBoost (`max_depth=6`):** Allowed to "think deeper," this model managed to carve out highly specific, complex rules to accurately identify the rare Advanced routines (0.50 precision).
- **Fine-Tuned XGBoost (`max_depth=3`):** When constrained by hyperparameter tuning to maximize global metrics, the model became highly conservative. It achieved the highest overall accuracy (56.49%) by becoming an absolute expert at identifying Beginners (78% recall) but completely ignored the Advanced routines (0%). This beautifully illustrated the "Accuracy Illusion"—where a model sacrifices minority classes to pad its overall stats.

## 5. SHAP Analysis (Explainability)
Using SHAP (`TreeExplainer`) on the Baseline XGBoost model, we reverse-engineered the algorithm's decision-making process to reveal three massive fitness secrets:
1. **The Ultimate Gatekeeper (`avg_intensity`):** SHAP revealed a perfect linear relationship. High intensity is a strict mathematical requirement for an Advanced classification, while low intensity irrevocably defines a Beginner.
2. **Quality vs. Quantity (`avg_exercises_per_day`):** The model learned that Beginners perform a high number of different exercises per day ("gym tourists"), whereas Advanced lifters execute fewer, heavier compound movements.
3. **The Volume Paradox (`avg_weekly_volume`):** SHAP plots showed highly mixed signals for volume, proving that volume alone is ambiguous. Both novices (doing endless light reps) and elites (doing heavy volume) can have high weekly volume, perfectly illustrating why the algorithm faces a mathematical ceiling when trying to separate the classes.

## 6. Conclusions
This project successfully transformed chaotic, real-world fitness data into a structured Machine Learning pipeline. The modeling phase proved that human-assigned difficulty labels are highly subjective. Despite the statistical "glass ceiling" of ~56% accuracy on aggregate tabular features, the combination of XGBoost and SHAP allowed us to uncover the underlying human psychology of fitness, mathematically proving that Beginners prioritize quantity, while Advanced lifters prioritize quality and structured intensity.
