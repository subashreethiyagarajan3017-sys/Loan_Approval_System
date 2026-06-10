# Loan Approval Website

This repository converts the notebook into a full Flask web app with a frontend built using HTML, CSS, and JavaScript.

## Setup

1. Place `loan_approval_dataset.csv` in the same folder as `app.py`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the website:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Behavior

- The app reads your dataset, trains a scikit-learn pipeline, and saves it with `pickle` as `loan_model.pkl`.
- The frontend uses AJAX to send prediction requests without reloading the page.
- If the dataset is missing or invalid, the page displays a helpful error message.
- Use the form to input loan applicant values and receive an approval prediction.
