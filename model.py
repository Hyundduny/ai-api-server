import numpy as np

class Model:
    def __init__(self, model_type):
        self.model_type = model_type

        # 입력과 출력 데이터를 딕셔너리로 정의하여 처리 간소화
        logic_gates = {
            'NOT': {"inputs": np.array([[0], [1]]), "outputs": np.array([1, 0])},
            'AND': {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([0, 0, 0, 1])},
            'OR':  {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([0, 1, 1, 1])},
            'XOR': {"inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), "outputs": np.array([0, 1, 1, 0])}
        }

        if model_type not in logic_gates:
            raise ValueError("Invalid model type. Choose from 'NOT', 'AND', 'OR', 'XOR'.")

        self.inputs = logic_gates[model_type]["inputs"]
        self.outputs = logic_gates[model_type]["outputs"]

        self.weights = np.random.rand(self.inputs.shape[1])  # 입력 개수에 맞는 가중치
        self.bias = np.random.rand(1)  # 편향 초기화

    def train(self):
        learning_rate = 0.1
        epochs = 20   
        for epoch in range(epochs):
            for i in range(len(self.inputs)):
                # 총 입력 계산
                total_input = np.dot(self.inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = self.outputs[i] - prediction
                print(f'inputs[i] : {self.inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * self.inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)  # 벡터 연산 적용

    def predict(self):
        total_input = np.dot(self.inputs, self.weights) + self.bias
        return self.step_function(total_input)  # 벡터 연산 적용