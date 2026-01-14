"""Risk band classification for customer churn prediction."""

def assign_risk_band(churn_probability, low_threshold=0.30, high_threshold=0.60):
    """Classify customer into risk band based on churn probability."""
    if churn_probability < low_threshold:
        return 'Low Risk'
    elif churn_probability < high_threshold:
        return 'Medium Risk'
    else:
        return 'High Risk'

def get_risk_color(risk_band):
    """Get color representation for risk band."""
    colors = {
        'Low Risk': 'green',
        'Medium Risk': 'yellow',
        'High Risk': 'red'
    }
    return colors.get(risk_band, 'gray')
