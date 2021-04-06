from sklearn.linear_model import LinearRegression

class LinearRegressionModel:

    def __init__(self):
        self.model = LinearRegression()

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
