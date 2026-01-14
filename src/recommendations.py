"""Recommendations module for churn prediction.

Provides personalized recommendations to retain at-risk customers
based on churn predictions and risk classifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RecommendationEngine:
    """Generate retention recommendations for high-risk customers."""
    
    RETENTION_STRATEGIES = {
        'High Risk': [
            'Priority support: Assign dedicated account manager',
            'Special discount: Offer 20-30% discount on next billing cycle',
            'Service upgrade: Free upgrade to premium tier for 3 months',
            'Loyalty program: Enroll in exclusive customer retention program',
            'Win-back offer: Custom bundle based on customer preferences'
        ],
        'Medium Risk': [
            'Proactive contact: Schedule satisfaction check-in call',
            'Moderate discount: Offer 10-15% discount on renewal',
            'Product demo: Showcase new features relevant to customer',
            'Extended trial: Add 1-2 months free service trial'
        ],
        'Low Risk': [
            'Engagement: Share product tips and best practices',
            'Referral program: Encourage customer to refer friends',
            'Community: Invite to user group or webinar'
        ]
    }
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.recommendations_cache = {}
    
    def get_recommendations(self, customer_id: str, risk_band: str, 
                          churn_probability: float) -> Dict[str, any]:
        """Get recommendations for a specific customer.
        
        Args:
            customer_id: Unique customer identifier
            risk_band: Risk classification (Low/Medium/High)
            churn_probability: Predicted churn probability (0-1)
            
        Returns:
            Dictionary with recommendations and urgency level
        """
        strategies = self.RETENTION_STRATEGIES.get(risk_band, [])
        
        return {
            'customer_id': customer_id,
            'risk_band': risk_band,
            'churn_probability': churn_probability,
            'urgency': self._calculate_urgency(churn_probability),
            'strategies': strategies[:3],  # Top 3 strategies
            'recommended_actions': self._prioritize_actions(risk_band, churn_probability)
        }
    
    def _calculate_urgency(self, churn_prob: float) -> str:
        """Calculate urgency level based on churn probability."""
        if churn_prob > 0.8:
            return 'CRITICAL'
        elif churn_prob > 0.6:
            return 'HIGH'
        elif churn_prob > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _prioritize_actions(self, risk_band: str, churn_prob: float) -> List[str]:
        """Prioritize actions based on risk and churn probability."""
        actions = []
        
        if churn_prob > 0.7 and risk_band == 'High Risk':
            actions.append('IMMEDIATE: Contact customer within 24 hours')
            actions.append('Prepare special retention offer')
        elif churn_prob > 0.5 and risk_band in ['High Risk', 'Medium Risk']:
            actions.append('Schedule support contact within 48 hours')
            actions.append('Review customer account details')
        else:
            actions.append('Add to proactive engagement queue')
        
        return actions
    
    def batch_recommendations(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate recommendations for multiple customers.
        
        Args:
            customers_df: DataFrame with customer_id, risk_band, churn_probability
            
        Returns:
            DataFrame with recommendations for each customer
        """
        recommendations = []
        
        for _, row in customers_df.iterrows():
            rec = self.get_recommendations(
                row['customer_id'],
                row['risk_band'],
                row['churn_probability']
            )
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)


def generate_customer_retention_plan(risk_classifier_output: Dict) -> Dict:
    """Generate comprehensive retention plan from classifier output.
    
    Args:
        risk_classifier_output: Output from risk classifier module
        
    Returns:
        Dictionary with retention strategies and implementation plan
    """
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(
        risk_classifier_output['customer_id'],
        risk_classifier_output['risk_band'],
        risk_classifier_output['churn_probability']
    )
    
    return {
        'customer_id': recommendations['customer_id'],
        'retention_plan': recommendations['strategies'],
        'implementation_timeline': 'Within 7 days for high-risk customers',
        'success_metrics': [
            'Customer satisfaction score > 8/10',
            'No churn within 90 days',
            'Increased service usage'
        ]
    }
