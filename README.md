# 🏨 Hotel Booking Cancellation Predictor & Revenue Protection

Predicting hotel reservation cancellations to optimize overbooking policies and protect hotel revenue.

## 🚀 Live Demo
Try the deployed application here: [Live Web App Link]

---

## 📌 Problem Framing
Last-minute booking cancellations lead to severe revenue loss and operational inefficiency for hotels. This project provides a machine learning solution to forecast cancellations at the time of booking, enabling revenue managers to apply targeted deposit requirements and smarter overbooking thresholds.

## 📊 Dataset & Pipeline Overview
- **Dataset**: Hotel Booking Demand (~119K records from real hotel management systems).
- **Leakage Prevention**: Strictly removed columns known only after check-in/out (`reservation_status`, `reservation_status_date`).
- **Pipeline Architecture**: Automated handling of missing values, standard scaling for numeric features, and one-hot encoding for categorical variables inside a scikit-learn `Pipeline`.

## 📈 Model Performance
| Model | ROC-AUC | Notes |
| :--- | :--- | :--- |
| **Baseline (Dummy)** | 0.5000 | Majority class strategy |
| **Logistic Regression** | ~0.79 | Fast linear baseline |
| **Random Forest** | ~0.88 | Strong non-linear baseline |
| **HistGradientBoosting (Tuned)** | **~0.89+** | Best balance between accuracy and latency |

## 🛠️ How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/hotel-cancellation-project.git
   cd hotel-cancellation-project

   ## 6. Business Insights & Deployment Recommendations




   

### Core Business Takeaways:
1. **Lead Time is the Strongest Predictor**: Guests booking more than 90 days in advance account for over 60% of cancellations. Early re-confirmation triggers should be automated for this cohort.
2. **Deposit Policy Effectiveness**: Bookings with 'Non Refund' deposits currently show anomalous cancellation behavior, pointing to bulk agent bookings. Shifting to dynamic risk-based deposits will protect inventory more effectively.
3. **Revenue Impact**: Deploying this screening tool allows the hotel to prevent revenue loss from empty rooms while maintaining an 85% Precision rate (avoiding unnecessary penalties on loyal guests).

### Actionable Policy Matrix:
* **High Risk (> 60% probability)**: Mandate full non-refundable upfront payment or credit card pre-authorization.
* **Medium Risk (30% - 60% probability)**: Require 1-night deposit and send automated confirmation requests 48 hours before arrival.
* **Low Risk (< 30% probability)**: Flexible standard booking policy with free cancellation up to 24 hours prior.

### Model Limitations:
* **External Disruptions**: The model does not capture unexpected macro-events (e.g., flight groundings, extreme weather, geopolitical changes).
* **High Season Shifts**: Booking patterns during local festivals or unexpected holidays may deviate from standard historical data.