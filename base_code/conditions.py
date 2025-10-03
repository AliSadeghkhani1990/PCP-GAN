"""
Modular condition management system.

Provides flexible framework for managing controllable properties (depth, porosity).
Easily extensible to add new conditions. Handles categorical (one-hot) and
continuous conditions with appropriate scaling and dimension management.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np

@dataclass
class Condition:
    name: str
    enabled: bool
    scaling_function: Callable
    extraction_function: Callable
    dimension: int = 1  # Default is 1 for most conditions

class ConditionManager:
    def __init__(self):
        self._conditions = {}
        
    def add_condition(self, condition: Condition):
        self._conditions[condition.name] = condition
        
    def enable_condition(self, condition_name: str):
        if condition_name in self._conditions:
            self._conditions[condition_name].enabled = True
            
    def disable_condition(self, condition_name: str):
        if condition_name in self._conditions:
            self._conditions[condition_name].enabled = False
    
    def update_condition_dimension(self, condition_name: str, dimension: int):
        """Update the dimension of a condition based on dataset analysis"""
        if condition_name in self._conditions:
            self._conditions[condition_name].dimension = dimension
            print(f"Updated {condition_name} condition dimension to {dimension}")
    
    @property
    def active_conditions(self) -> List[Condition]:
        return [cond for cond in self._conditions.values() if cond.enabled]
    
    def get_condition_values(self, patch_info: Dict) -> List[float]:
        values = []
        for condition in self.active_conditions:
            raw_value = condition.extraction_function(patch_info)
            scaled_value = condition.scaling_function(raw_value)
            values.append(scaled_value)
        return values

def create_default_condition_manager():
    manager = ConditionManager()
    
    # Add category condition - dimension will be set later based on dataset
    category_condition = Condition(
        name='category',
        enabled=True,
        scaling_function=lambda x: x,  # Using your existing scaled_category
        extraction_function=lambda x: x['category_scaled'],
        dimension=1  # This will be updated dynamically based on actual dataset
    )
    manager.add_condition(category_condition)
    
    # Add porosity condition
    porosity_condition = Condition(
        name='porosity',
        enabled=True,
        scaling_function=lambda x: x,  # Using your existing porosity_value
        extraction_function=lambda x: x['porosity_value']
    )
    manager.add_condition(porosity_condition)
    
    return manager