import numpy as np
import pickle
import os

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
    
    def train(self):
        self.weights = np.random.rand(self.inputs.shape[1])  # 입력 개수에 맞는 가중치
        self.bias = np.random.rand(1)  # 편향 초기화
    
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

    def load_model(self):
        file_name = f"model/{self.model_type}.pkl"
        
        if not os.path.exists(file_name):
            print(f"Error: Model file {file_name} not found.")
            return False

        with open(file_name, "rb") as f:
            model_data = pickle.load(f)
        print(model_data)
        self.weights = model_data["weight"]
        self.bias = model_data["bias"]
        print(f"Model {self.model_type} loaded successfulSy from {file_name}")
        return True
    
    def step_function(self, x):
        return np.where(x >= 0, 1, 0)  # 벡터 연산 적용

    def predict(self):
        total_input = np.dot(self.inputs, self.weights) + self.bias
        return self.step_function(total_input)  # 벡터 연산 적용
    
    def save_model(self):
        model_data = {
            "weight": self.weights,
            "bias": self.bias
        }
        file_name = f"model/{self.model_type}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {file_name}")




def main():
    for model_type in ["AND", "OR", "NOT", "XOR"]:
        model = Model(model_type)
        model.train()
        model.save_model()


if __name__ == "__main__":
    main()
