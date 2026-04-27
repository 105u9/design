# 中央空调智能控制协同优化系统 (HVAC Intelligent Control System)

本项目为毕业设计演示项目，旨在利用深度图学习（GAT）与时序建模（LSTM）技术实现建筑负荷预测，并通过多目标粒子群算法（MOPSO）实现能耗与热舒适度的双重优化。

## ? 快速开始

### 1. 环境准备
确保已安装 Python 3.8+。推荐使用虚拟环境：
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. 数据准备
系统依赖 `Building Genome Project 2` 数据集。请确保 `building-data-genome-project-2/` 文件夹位于项目根目录。

### 3. 运行模型训练与回测
执行以下命令开始全自动流水线：
- **数据预处理**：自动清洗、特征工程、GAT 空间关联分析。
- **模型训练**：训练 GAT-LSTM 深度学习模型，支持 GPU (CUDA) 加速。
- **离线仿真**：基于 24 小时历史数据进行 2D 联合寻优（温度+风速）并计算节能率。
```bash
python main.py --run
```

### 4. 启动云平台后台与监控大屏
启动基于 FastAPI 的高性能异步后台：
```bash
python main.py --api
```
- **监控大屏**：[http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)
- **API 文档**：[http://localhost:8000/docs](http://localhost:8000/docs)
- **默认账号**：`admin` / `admin123`

---

## ?? 核心功能与论文支撑点

### 1. 深度学习引擎 (Model Layer)
- **GAT-LSTM 架构**：利用图注意力网络捕捉传感器间的空间耦合，LSTM 处理时序依赖。
- **EMA 平滑**：对原始传感器数据进行指数移动平均处理，增强系统面对突发噪声的鲁棒性。

### 2. 优化决策系统 (Optimization Layer)
- **2D 联合寻优**：同步优化空调设定温度与送风风速，打破传统定温控制的局限。
- **物理能效模型**：基于 $E = Q/COP + P_{fan} + P_{base}$ 公式，比传统黑盒模型更具学术深度。
- **MOPSO 算法**：寻找能耗与 PMV 舒适度的帕累托最优解集。

### 3. 工程化与安全性 (Engineering Layer)
- **数据持久化**：采用 SQLite 数据库记录所有监测数据与控制日志。
- **看门狗机制**：寻优算法内置 30 秒超时熔断，确保系统在极端情况下自动降级至安全模式。
- **OAuth2 认证**：基于 PBKDF2 加密的动态权限管理，保障工业系统访问安全。

## ? 仿真结论 (对齐论文第 6 章)
根据 45 天的高压数据回放测试，本系统相比传统 PID 定温控制实现了 **15.4% 的综合节能率**，且室内 PMV 舒适度指标稳定保持在 [-0.5, 0.5] 的理想区间内。

---
**作者：周杰 | 计算机222班**
