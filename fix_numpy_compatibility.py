#!/usr/bin/env python3
"""
Fix NumPy 2.0 compatibility issues in the codebase
"""

import os
import re

def fix_numpy_inf():
    """Replace np.Inf with np.inf for NumPy 2.0 compatibility"""
    
    files_to_fix = [
        'advanced_model_training.py',
        'run_real_experiments.py'
    ]
    
    for filename in files_to_fix:
        if os.path.exists(filename):
            print(f"Fixing {filename}...")
            
            with open(filename, 'r') as f:
                content = f.read()
            
            # Replace np.Inf with np.inf
            original_count = content.count('np.Inf')
            content = content.replace('np.Inf', 'np.inf')
            
            with open(filename, 'w') as f:
                f.write(content)
            
            print(f"  Fixed {original_count} occurrences of np.Inf → np.inf")
        else:
            print(f"  {filename} not found")
    
    print("✅ NumPy compatibility fixed!")

if __name__ == "__main__":
    fix_numpy_inf()