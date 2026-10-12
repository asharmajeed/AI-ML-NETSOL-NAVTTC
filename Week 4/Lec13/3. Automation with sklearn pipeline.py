from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([40, 45, 52, 60, 68, 75, 82, 90])

pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

pipe.fit(X, y)
print(pipe.predict([[9]]))
