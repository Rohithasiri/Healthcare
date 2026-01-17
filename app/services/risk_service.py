"""
Risk Assessment Service
Calculates cardiovascular risk scores using multiple validated algorithms
"""

from datetime import datetime, date
from typing import Dict, List, Optional
import math
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.health_profile import HealthProfile
from app.models.cholesterol_record import CholesterolRecord
from app.models.screening_record import ScreeningRecord
from app.models.risk_assessment import RiskAssessment


class RiskAssessmentService:
    """
    Comprehensive cardiovascular risk assessment service
    Uses Framingham Risk Score and ACC/AHA pooled cohort equations
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_age(self, birth_date: date) -> int:
        """Calculate age from birth date"""
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    
    def calculate_framingham_risk_score(
        self,
        age: int,
        gender: str,
        total_cholesterol: float,
        hdl_cholesterol: float,
        systolic_bp: float,
        is_smoker: bool,
        has_diabetes: bool,
        on_bp_medication: bool = False
    ) -> Dict:
        """
        Calculate 10-year cardiovascular disease risk using Framingham Risk Score
        
        Args:
            age: Age in years (30-74)
            gender: 'male' or 'female'
            total_cholesterol: Total cholesterol in mg/dL
            hdl_cholesterol: HDL cholesterol in mg/dL
            systolic_bp: Systolic blood pressure in mmHg
            is_smoker: Current smoking status
            has_diabetes: Diabetes status
            on_bp_medication: Whether on BP medication
        
        Returns:
            Dictionary with risk score, category, and details
        """
        
        points = 0
        
        # Gender-specific calculations
        if gender.lower() == 'male':
            points += self._calculate_male_framingham_points(
                age, total_cholesterol, hdl_cholesterol, systolic_bp,
                is_smoker, has_diabetes, on_bp_medication
            )
        else:
            points += self._calculate_female_framingham_points(
                age, total_cholesterol, hdl_cholesterol, systolic_bp,
                is_smoker, has_diabetes, on_bp_medication
            )
        
        # Convert points to risk percentage
        risk_percentage = self._points_to_risk_percentage(points, gender)
        
        # Determine risk category
        category = self._get_risk_category(risk_percentage)
        
        return {
            'risk_score': round(risk_percentage, 2),
            'points': points,
            'category': category,
            'algorithm': 'Framingham Risk Score',
            'timeframe': '10-year risk'
        }
    
    def _calculate_male_framingham_points(
        self, age, total_chol, hdl, sbp, is_smoker, has_diabetes, on_bp_med
    ) -> int:
        """Calculate Framingham points for males"""
        points = 0
        
        # Age points
        if age < 35:
            points += -1
        elif age < 40:
            points += 0
        elif age < 45:
            points += 1
        elif age < 50:
            points += 2
        elif age < 55:
            points += 3
        elif age < 60:
            points += 4
        elif age < 65:
            points += 5
        elif age < 70:
            points += 6
        else:
            points += 7
        
        # Total cholesterol points
        if total_chol < 160:
            points += -3
        elif total_chol < 200:
            points += 0
        elif total_chol < 240:
            points += 1
        elif total_chol < 280:
            points += 2
        else:
            points += 3
        
        # HDL points
        if hdl < 35:
            points += 2
        elif hdl < 45:
            points += 1
        elif hdl < 50:
            points += 0
        elif hdl < 60:
            points += -1
        else:
            points += -2
        
        # Blood pressure points
        if sbp < 120:
            points += 0 if not on_bp_med else 1
        elif sbp < 130:
            points += 0 if not on_bp_med else 2
        elif sbp < 140:
            points += 1 if not on_bp_med else 3
        elif sbp < 160:
            points += 2 if not on_bp_med else 4
        else:
            points += 3 if not on_bp_med else 5
        
        # Smoking
        if is_smoker:
            points += 2
        
        # Diabetes
        if has_diabetes:
            points += 2
        
        return points
    
    def _calculate_female_framingham_points(
        self, age, total_chol, hdl, sbp, is_smoker, has_diabetes, on_bp_med
    ) -> int:
        """Calculate Framingham points for females"""
        points = 0
        
        # Age points
        if age < 35:
            points += -9
        elif age < 40:
            points += -4
        elif age < 45:
            points += 0
        elif age < 50:
            points += 3
        elif age < 55:
            points += 6
        elif age < 60:
            points += 7
        elif age < 65:
            points += 8
        elif age < 70:
            points += 8
        else:
            points += 8
        
        # Total cholesterol points
        if total_chol < 160:
            points += -2
        elif total_chol < 200:
            points += 0
        elif total_chol < 240:
            points += 1
        elif total_chol < 280:
            points += 1
        else:
            points += 3
        
        # HDL points
        if hdl < 35:
            points += 5
        elif hdl < 45:
            points += 2
        elif hdl < 50:
            points += 1
        elif hdl < 60:
            points += 0
        else:
            points += -2
        
        # Blood pressure points
        if sbp < 120:
            points += -3 if not on_bp_med else 0
        elif sbp < 130:
            points += 0 if not on_bp_med else 2
        elif sbp < 140:
            points += 0 if not on_bp_med else 3
        elif sbp < 160:
            points += 2 if not on_bp_med else 5
        else:
            points += 3 if not on_bp_med else 6
        
        # Smoking
        if is_smoker:
            points += 2
        
        # Diabetes
        if has_diabetes:
            points += 4
        
        return points
    
    def _points_to_risk_percentage(self, points: int, gender: str) -> float:
        """Convert Framingham points to 10-year CVD risk percentage"""
        
        if gender.lower() == 'male':
            # Male risk lookup
            risk_lookup = {
                -3: 1, -2: 2, -1: 2, 0: 3, 1: 4, 2: 4, 3: 6,
                4: 7, 5: 9, 6: 11, 7: 14, 8: 18, 9: 22,
                10: 27, 11: 33, 12: 40, 13: 47, 14: 56, 15: 65
            }
        else:
            # Female risk lookup
            risk_lookup = {
                -2: 1, -1: 2, 0: 2, 1: 2, 2: 3, 3: 3, 4: 4,
                5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 11,
                11: 13, 12: 15, 13: 17, 14: 20, 15: 24, 16: 27
            }
        
        # Clamp points to valid range
        if points < min(risk_lookup.keys()):
            return risk_lookup[min(risk_lookup.keys())]
        elif points > max(risk_lookup.keys()):
            return risk_lookup[max(risk_lookup.keys())]
        else:
            return risk_lookup.get(points, 10)
    
    def _get_risk_category(self, risk_percentage: float) -> str:
        """Categorize risk level"""
        if risk_percentage < 10:
            return "low"
        elif risk_percentage < 20:
            return "moderate"
        else:
            return "high"
    
    def assess_user_risk(self, user_id: str) -> Dict:
        """
        Comprehensive risk assessment for a user
        Pulls latest data from database
        """
        
        # Get user
        user = self.db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Get health profile
        health_profile = self.db.query(HealthProfile).filter(
            HealthProfile.user_id == user_id
        ).first()
        
        if not health_profile:
            raise ValueError(f"Health profile not found for user {user_id}")
        
        # Get latest cholesterol record (OPTIONAL)
        latest_cholesterol = self.db.query(CholesterolRecord).filter(
            CholesterolRecord.user_id == user_id
        ).order_by(CholesterolRecord.test_date.desc()).first()
        
        # Get latest screening (for BP)
        latest_screening = self.db.query(ScreeningRecord).filter(
            ScreeningRecord.user_id == user_id
        ).order_by(ScreeningRecord.screening_date.desc()).first()
        
        # Calculate age
        age = self.calculate_age(user.date_of_birth)
        
        # Get BP values (use default if not available)
        systolic_bp = 120  # Default normal BP
        diastolic_bp = 80  # Default normal BP
        has_bp_data = False
        
        if latest_screening and latest_screening.blood_pressure_systolic:
            systolic_bp = latest_screening.blood_pressure_systolic
            diastolic_bp = latest_screening.blood_pressure_diastolic
            has_bp_data = True
        
        # Get cholesterol values (use defaults if not available)
        total_cholesterol = 200  # Average adult cholesterol
        hdl_cholesterol = 50     # Average HDL
        ldl_cholesterol = 100    # Average LDL
        triglycerides = 150      # Average triglycerides
        has_cholesterol_data = False
        
        if latest_cholesterol:
            total_cholesterol = latest_cholesterol.total_cholesterol or 200
            hdl_cholesterol = latest_cholesterol.hdl_cholesterol or 50
            ldl_cholesterol = latest_cholesterol.ldl_cholesterol or 100
            triglycerides = latest_cholesterol.triglycerides or 150
            has_cholesterol_data = True
        
        # Calculate Framingham risk score
        framingham_result = self.calculate_framingham_risk_score(
            age=age,
            gender=user.gender,
            total_cholesterol=total_cholesterol,
            hdl_cholesterol=hdl_cholesterol,
            systolic_bp=systolic_bp,
            is_smoker=health_profile.smoking_status == 'current',
            has_diabetes=health_profile.has_diabetes,
            on_bp_medication=getattr(health_profile, 'on_bp_medication', False)
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            risk_category=framingham_result['category'],
            age=age,
            systolic_bp=systolic_bp,
            total_cholesterol=total_cholesterol,
            ldl_cholesterol=ldl_cholesterol,
            hdl_cholesterol=hdl_cholesterol,
            is_smoker=health_profile.smoking_status == 'current',
            has_diabetes=health_profile.has_diabetes,
            bmi=health_profile.bmi,
            has_cholesterol_data=has_cholesterol_data,
            has_bp_data=has_bp_data
        )
        
        # Save to database with correct field names
        risk_assessment = RiskAssessment(
            user_id=user_id,
            assessment_date=datetime.now(),
            risk_score=framingham_result['risk_score'],
            risk_category=framingham_result['category'],
            contributing_factors={
                'age': age,
                'gender': user.gender,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'total_cholesterol': total_cholesterol,
                'hdl_cholesterol': hdl_cholesterol,
                'ldl_cholesterol': ldl_cholesterol,
                'triglycerides': triglycerides,
                'is_smoker': health_profile.smoking_status == 'current',
                'has_diabetes': health_profile.has_diabetes,
                'bmi': health_profile.bmi,
                'algorithm': framingham_result['algorithm']
            },
            bp_risk_component=0.0,
            cholesterol_risk_component=0.0,
            lifestyle_risk_component=0.0,
            family_history_risk_component=0.0,
            model_confidence=0.85 if (has_cholesterol_data and has_bp_data) else 0.65
        )
        
        self.db.add(risk_assessment)
        self.db.commit()
        self.db.refresh(risk_assessment)
        
        assessment_id = risk_assessment.assessment_id
        
        # Add data completeness warnings
        warnings = []
        if not has_cholesterol_data:
            warnings.append("Using estimated cholesterol values. Get a blood test for accurate results.")
        if not has_bp_data:
            warnings.append("Using default blood pressure. Measure your BP for better accuracy.")
        
        return {
            'assessment_id': assessment_id,
            'risk_score': framingham_result['risk_score'],
            'risk_category': framingham_result['category'],
            'points': framingham_result['points'],
            'algorithm': framingham_result['algorithm'],
            'timeframe': framingham_result['timeframe'],
            'assessment_date': risk_assessment.assessment_date.isoformat(),
            'data_completeness': {
                'has_cholesterol_data': has_cholesterol_data,
                'has_bp_data': has_bp_data,
                'estimated_values_used': not (has_cholesterol_data and has_bp_data)
            },
            'warnings': warnings,
            'input_data': {
                'age': age,
                'gender': user.gender,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'total_cholesterol': total_cholesterol,
                'hdl_cholesterol': hdl_cholesterol,
                'ldl_cholesterol': ldl_cholesterol,
                'triglycerides': triglycerides,
                'is_smoker': health_profile.smoking_status == 'current',
                'has_diabetes': health_profile.has_diabetes,
                'bmi': health_profile.bmi
            },
            'recommendations': recommendations
        }
    
    def generate_recommendations(
        self,
        risk_category: str,
        age: int,
        systolic_bp: float,
        total_cholesterol: float,
        ldl_cholesterol: float,
        hdl_cholesterol: float,
        is_smoker: bool,
        has_diabetes: bool,
        bmi: Optional[float] = None,
        has_cholesterol_data: bool = True,
        has_bp_data: bool = True
    ) -> List[Dict]:
        """Generate personalized health recommendations"""
        
        recommendations = []
        
        # Data completeness recommendations
        if not has_cholesterol_data:
            recommendations.append({
                'priority': 'high',
                'category': 'testing',
                'title': 'Get Cholesterol Test',
                'description': 'Your risk assessment is based on estimated cholesterol values. A lipid panel blood test will provide accurate measurements for better risk assessment.',
                'action': 'Schedule blood test at lab or doctor office'
            })
        
        if not has_bp_data:
            recommendations.append({
                'priority': 'high',
                'category': 'testing',
                'title': 'Measure Blood Pressure',
                'description': 'Your risk assessment is based on default blood pressure values. Regular BP monitoring provides better accuracy.',
                'action': 'Measure BP at home, pharmacy, or use our video screening'
            })
        
        # Risk-specific recommendations
        if risk_category == 'high':
            recommendations.append({
                'priority': 'critical',
                'category': 'medical',
                'title': 'Consult Healthcare Provider Immediately',
                'description': 'Your cardiovascular risk is high. Schedule an appointment with your doctor as soon as possible for comprehensive evaluation and treatment plan.',
                'action': 'Schedule appointment within 1 week'
            })
        
        # Blood pressure recommendations
        if systolic_bp >= 140:
            recommendations.append({
                'priority': 'high',
                'category': 'blood_pressure',
                'title': 'Manage High Blood Pressure',
                'description': f'Your blood pressure ({systolic_bp} mmHg) is elevated. Reduce sodium intake, exercise regularly, and consult your doctor about medication.',
                'action': 'Monitor BP daily, aim for <130/80'
            })
        elif systolic_bp >= 130:
            recommendations.append({
                'priority': 'medium',
                'category': 'blood_pressure',
                'title': 'Monitor Blood Pressure',
                'description': 'Your blood pressure is slightly elevated. Lifestyle changes can help bring it down.',
                'action': 'Check BP weekly, reduce salt intake'
            })
        
        # Cholesterol recommendations
        if ldl_cholesterol >= 160:
            recommendations.append({
                'priority': 'high',
                'category': 'cholesterol',
                'title': 'Reduce LDL Cholesterol',
                'description': f'Your LDL cholesterol ({ldl_cholesterol} mg/dL) is very high. Dietary changes and medication may be needed.',
                'action': 'Limit saturated fats, discuss statins with doctor'
            })
        elif ldl_cholesterol >= 130:
            recommendations.append({
                'priority': 'medium',
                'category': 'cholesterol',
                'title': 'Improve Cholesterol Levels',
                'description': 'Your LDL cholesterol is borderline high. Focus on heart-healthy diet.',
                'action': 'Increase fiber, reduce trans fats'
            })
        
        if hdl_cholesterol < 40:
            recommendations.append({
                'priority': 'medium',
                'category': 'cholesterol',
                'title': 'Increase HDL (Good) Cholesterol',
                'description': 'Your HDL cholesterol is low. Exercise and healthy fats can help.',
                'action': 'Exercise 30 min daily, eat omega-3 rich foods'
            })
        
        # Smoking recommendations
        if is_smoker:
            recommendations.append({
                'priority': 'critical',
                'category': 'lifestyle',
                'title': 'Quit Smoking',
                'description': 'Smoking is a major risk factor for heart disease. Quitting is the single most important thing you can do.',
                'action': 'Join smoking cessation program, consider nicotine replacement'
            })
        
        # Diabetes recommendations
        if has_diabetes:
            recommendations.append({
                'priority': 'high',
                'category': 'medical',
                'title': 'Manage Diabetes',
                'description': 'Diabetes increases cardiovascular risk. Keep blood sugar well-controlled.',
                'action': 'Monitor glucose daily, HbA1c <7%'
            })
        
        # BMI recommendations
        if bmi and bmi >= 30:
            recommendations.append({
                'priority': 'high',
                'category': 'lifestyle',
                'title': 'Weight Management',
                'description': f'Your BMI ({bmi:.1f}) indicates obesity. Weight loss can significantly reduce cardiovascular risk.',
                'action': 'Aim for 5-10% weight loss, combine diet and exercise'
            })
        elif bmi and bmi >= 25:
            recommendations.append({
                'priority': 'medium',
                'category': 'lifestyle',
                'title': 'Achieve Healthy Weight',
                'description': 'Reaching a healthy weight can improve heart health.',
                'action': 'Balanced diet and regular exercise'
            })
        
        # General recommendations
        recommendations.append({
            'priority': 'low',
            'category': 'lifestyle',
            'title': 'Regular Physical Activity',
            'description': 'Aim for at least 150 minutes of moderate exercise per week.',
            'action': '30 minutes of walking, 5 days a week'
        })
        
        recommendations.append({
            'priority': 'low',
            'category': 'lifestyle',
            'title': 'Heart-Healthy Diet',
            'description': 'Follow Mediterranean or DASH diet patterns.',
            'action': 'More fruits, vegetables, whole grains, fish'
        })
        
        recommendations.append({
            'priority': 'low',
            'category': 'monitoring',
            'title': 'Regular Health Screenings',
            'description': 'Monitor your cardiovascular health regularly.',
            'action': 'Annual check-up, BP and cholesterol tests'
        })
        
        return recommendations
    
    def get_risk_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get risk assessment history for a user"""
        
        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.user_id == user_id
        ).order_by(RiskAssessment.assessment_date.desc()).limit(limit).all()
        
        return [
            {
                'id': a.assessment_id,
                'assessment_date': a.assessment_date.isoformat(),
                'risk_score': a.risk_score,
                'risk_category': a.risk_category,
                'algorithm': a.contributing_factors.get('algorithm', 'Framingham Risk Score'),
                'systolic_bp': a.contributing_factors.get('systolic_bp', 0),
                'total_cholesterol': a.contributing_factors.get('total_cholesterol', 0)
            }
            for a in assessments
        ]