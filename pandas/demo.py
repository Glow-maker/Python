import pandas as pd
import numpy as np

teacher = pd.DataFrame({
    'teacher_id': [1, 1, 1, 2, 2, 2, 2],
    'subject_id': [2, 2, 3, 1, 2, 3, 4],
    'dept_id': [3, 4, 3, 1, 1, 1, 1]
})
dt = list(teacher.groupby('teacher_id'))
df = teacher.groupby('teacher_id')['subject_id'].nunique().reset_index()

print(df)