from locust import HttpUser, task, between

class HVACApiUser(HttpUser):
    # 模拟每个操作员每次操作之间停顿 1 到 3 秒
    wait_time = between(1, 3) 
    token = None

    def on_start(self):
        """每个虚拟用户开始测试前的生命周期钩子：负责登录并获取身份令牌"""
        response = self.client.post("/token", data={
            "username": "admin",
            "password": "admin123"
        })
        if response.status_code == 200:
            self.token = response.json().get("access_token")
        else:
            print("登录失败，请检查账号密码！")

    @task(3) # 权重 3：普通监控接口请求最频繁
    def test_monitoring(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.get("/api/v1/monitoring", headers=headers, name="GET /monitoring")

    @task(1) # 权重 1：AI 预测接口，重点观察含 LSTM 模型的并发耗时
    def test_predict(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/v1/predict", headers=headers, name="POST /predict")

    @task(1) # 权重 1：多目标粒子群优化算法接口
    def test_optimize(self):
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/v1/optimize", headers=headers, name="POST /optimize")
