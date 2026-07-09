import json
import joblib
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE_DIR, 'xgboost_credit_risk_model.pkl'))
SCALER = joblib.load(os.path.join(BASE_DIR, 'credit_risk_scaler.pkl'))

EXPECTED_COLUMNS = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 
    'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_B', 'loan_grade_C', 
    'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G', 'cb_person_default_on_file_Y'
]

@csrf_exempt
def predict_risk(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            df = pd.DataFrame([data])
            df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
            
            scaled_data = SCALER.transform(df)
            prediction = MODEL.predict(scaled_data)[0]
            probability = MODEL.predict_proba(scaled_data)[0][1]
            
            return JsonResponse({
                'status': 'success',
                'risk_score': round(float(probability) * 100, 2),
                'decision': 'DENIED' if prediction == 1 else 'APPROVED'
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            
    # THIS is the safety net that prevents the "returned None" error!
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed'}, status=405)