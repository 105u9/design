<template>
  <div class="sandbox-container">
    <!-- 顶部状态栏 -->
    <div class="status-bar glass" :class="statusClass">
      <div class="status-icon">{{ statusIcon }}</div>
      <div class="status-text">
        <div class="status-title">当前系统运行状态 (System State)</div>
        <div class="status-value">{{ currentMode }}</div>
      </div>
      <div class="status-pulse" :class="pulseClass"></div>
    </div>

    <div class="sandbox-content">
      <!-- 左侧数据图表区 -->
      <div class="charts-panel c-span-8">
        <div class="chart-box glass">
          <div class="chart-title">环境感知数据流与预处理监测</div>
          <v-chart class="chart-container" :option="envChartOption" autoresize />
        </div>
        <div class="chart-box glass mt-15">
          <div class="chart-title">环境热舒适度(PMV)预测模型输出 (GAT-LSTM)</div>
          <div class="pmv-overlay" v-if="currentMode === '人工干预模式 (本地闭环)'">
            预测模型挂起 (Model Suspended)
          </div>
          <v-chart class="chart-container" :option="pmvChartOption" autoresize />
        </div>
      </div>

      <!-- 右侧拓扑与日志区 -->
      <div class="side-panel c-span-4">
        <div class="glass flex-col h-half">
          <div class="panel-title">底层执行网络节点状态拓扑</div>
          <div class="nodes-container">
            <div class="node-item" v-for="(node, idx) in hardwareNodes" :key="idx" :class="node.status">
              <div class="node-icon"></div>
              <div class="node-info">
                <div class="node-name">{{ node.name }}</div>
                <div class="node-state">{{ node.status === 'online' ? '[Online] 报文收发正常' : (node.status === 'offline' ? '[Offline] 节点链路熔断' : '[Warning] 报文超时重传中') }}</div>
              </div>
              <div class="node-indicator"></div>
            </div>
          </div>
        </div>

        <div class="glass flex-col h-half mt-15 log-terminal">
          <div class="panel-title">系统核心调度引擎日志 (Syslog)</div>
          <div class="log-content" ref="logContainer">
            <div v-for="(log, idx) in logs" :key="idx" class="log-line" :class="log.type">
              <span class="log-time">[{{ log.time }}]</span>
              <span class="log-msg">{{ log.message }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 交互控制 (Chaos Engineering Control Panel) -->
    <div class="chaos-panel glass">
      <div class="chaos-title">混沌工程故障注入与权限调度中枢 (Fault Injection & Privileges)</div>
      <div class="chaos-buttons">
        <button class="btn btn-warning" @click="injectDirtyData">
          注入边缘端畸变特征值 (模拟传感器漂移失效)
        </button>
        <button class="btn btn-orange" @click="injectAlgorithmTimeout">
          触发非线性寻优死锁 (模拟求解器收敛超时)
        </button>
        <button class="btn btn-danger" @click="injectNetworkCut">
          截断物理网关链路 (模拟硬件网络断联)
        </button>
        <button class="btn btn-primary" @click="manualTakeover">
          调度最高控制权限 (强制越权干预)
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue';
import VChart from 'vue-echarts';

// 状态管理
type SysMode = 'AI 预测控制模式 (MPC 寻优)' | '安全降级模式 (Fallback)' | '人工干预模式 (本地闭环)';
const currentMode = ref<SysMode>('AI 预测控制模式 (MPC 寻优)');

const statusClass = computed(() => {
  if (currentMode.value === 'AI 预测控制模式 (MPC 寻优)') return 'mode-green';
  if (currentMode.value === '安全降级模式 (Fallback)') return 'mode-orange';
  return 'mode-red';
});

const statusIcon = computed(() => {
  if (currentMode.value === 'AI 预测控制模式 (MPC 寻优)') return '🟢';
  if (currentMode.value === '安全降级模式 (Fallback)') return '🟠';
  return '🔴';
});

const pulseClass = computed(() => {
  if (currentMode.value === 'AI 预测控制模式 (MPC 寻优)') return 'pulse-green';
  if (currentMode.value === '安全降级模式 (Fallback)') return 'pulse-orange';
  return 'pulse-red';
});

// 硬件节点状态
const hardwareNodes = ref([
  { name: 'Node-01 [主车间空调机组/AHU]', status: 'online' },
  { name: 'Node-02 [末端变风量阀/VAV]', status: 'online' },
  { name: 'Node-03 [冷水主机/CH]', status: 'online' }
]);

// 日志系统
interface LogEntry { time: string; message: string; type: 'info' | 'warning' | 'error' | 'success' }
const logs = ref<LogEntry[]>([]);
const logContainer = ref<HTMLElement | null>(null);

const addLog = (msg: string, type: 'info' | 'warning' | 'error' | 'success' = 'info') => {
  const now = new Date();
  logs.value.push({
    time: `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`,
    message: msg,
    type
  });
  if (logs.value.length > 50) logs.value.shift();
  nextTick(() => {
    if (logContainer.value) logContainer.value.scrollTop = logContainer.value.scrollHeight;
  });
};

// 图表数据模拟
const timeData = ref<string[]>([]);
const tempData = ref<number[]>([]);
const rawTempData = ref<number[]>([]); 
const pmvData = ref<number[]>([]);

let simInterval: number;
let timeStep = 0;

const simulateCharts = () => {
  const now = new Date();
  timeData.value.push(`${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`);
  
  // 正常温度波动 24-26
  let temp = 24 + Math.sin(timeStep / 5) + Math.random() * 0.5;
  
  if (isInjectingDirty) {
    rawTempData.value.push(999); 
    tempData.value.push(tempData.value[tempData.value.length - 1]); 
    isInjectingDirty = false;
  } else {
    rawTempData.value.push(temp);
    tempData.value.push(temp);
  }

  // PMV波动
  let pmv = 0.5 * Math.cos(timeStep / 5) + Math.random() * 0.1;
  if (currentMode.value === '人工干预模式 (本地闭环)') {
    pmvData.value.push(pmvData.value[pmvData.value.length - 1]); 
  } else {
    pmvData.value.push(pmv);
  }

  if (timeData.value.length > 30) {
    timeData.value.shift();
    tempData.value.shift();
    rawTempData.value.shift();
    pmvData.value.shift();
  }
  timeStep++;
};

const commonAxisStyle = {
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
  splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
  axisLabel: { color: '#94a3b8' }
};

const envChartOption = computed(() => ({
  tooltip: { trigger: 'axis', backgroundColor: 'rgba(15, 20, 40, 0.8)', textStyle: { color: '#fff' } },
  legend: { data: ['原始感知层上报数据', '清洗插补后业务指标'], textStyle: { color: '#fff' } },
  grid: { left: '5%', right: '5%', bottom: '5%', top: '15%', containLabel: true },
  xAxis: { type: 'category', data: timeData.value, ...commonAxisStyle },
  yAxis: { type: 'value', name: '温度 (℃)', ...commonAxisStyle, min: 20, max: 30 },
  series: [
    {
      name: '原始感知层上报数据', type: 'line', data: rawTempData.value,
      itemStyle: { color: '#ff003c' }, lineStyle: { type: 'dashed', width: 2 }
    },
    {
      name: '清洗插补后业务指标', type: 'line', data: tempData.value, smooth: true,
      itemStyle: { color: '#00f0ff' }, lineStyle: { width: 2 }
    }
  ]
}));

const pmvChartOption = computed(() => ({
  tooltip: { trigger: 'axis', backgroundColor: 'rgba(15, 20, 40, 0.8)', textStyle: { color: '#fff' } },
  grid: { left: '5%', right: '5%', bottom: '5%', top: '15%', containLabel: true },
  xAxis: { type: 'category', data: timeData.value, ...commonAxisStyle },
  yAxis: { type: 'value', name: 'PMV 指数', ...commonAxisStyle, min: -1, max: 1 },
  series: [{
    name: '模型预测值', type: 'line', data: pmvData.value, smooth: true,
    itemStyle: { color: currentMode.value === '人工干预模式 (本地闭环)' ? '#64748b' : '#39ff14' },
    lineStyle: { width: 2 },
    areaStyle: {
      color: currentMode.value === '人工干预模式 (本地闭环)' ? 'rgba(100,116,139,0.1)' : 'rgba(57, 255, 20, 0.1)'
    }
  }]
}));

let isInjectingDirty = false;

const injectDirtyData = () => {
  addLog('[Data_Pipeline] 接收底层网关上报数据包, payload_size=256b', 'info');
  isInjectingDirty = true;
  setTimeout(() => {
    addLog('[ETL_Service] 触发异常值清洗逻辑: 捕获游离域外极大值 999.0', 'warning');
    addLog('[ETL_Service] 启动时序插补算法 (Forward-Fill)，保持输入张量连续性', 'success');
  }, 200);
};

const injectAlgorithmTimeout = () => {
  addLog('[MPC_Optimizer] 提交非线性寻优计算任务 (Horizon=12)...', 'info');
  addLog('[MPC_Optimizer] 检测到多维解空间局部极值，矩阵求逆挂起...', 'warning');
  
  setTimeout(() => {
    currentMode.value = '安全降级模式 (Fallback)';
    addLog('[Watchdog_Daemon] 寻优协程执行超限 (Threshold=3000ms)，触发强制回收 (SIGKILL)', 'error');
    addLog('[Failover_System] 系统服务降级: 广播应急保底设定值至底层网关', 'warning');
  }, 3000); 
};

const injectNetworkCut = () => {
  addLog('[MQTT_Broker] 向执行节点推送控制报文 (QoS=1)...', 'info');
  hardwareNodes.value[0].status = 'warning';
  
  let retry = 1;
  const retryInterval = setInterval(() => {
    addLog(`[TCP_Socket] ACK 确认包接收超时，发起指数退避重传 (Attempt=${retry}/3)`, 'warning');
    retry++;
    
    if (retry > 3) {
      clearInterval(retryInterval);
      hardwareNodes.value[0].status = 'offline';
      addLog('[MQTT_Broker] 链路熔断异常: 连续 3 次 TCP 心跳丢失', 'error');
      addLog('[Edge_Controller] Node-01 节点脱离业务集群，转由边缘 PLC 本地硬件托管', 'error');
    }
  }, 1000);
};

const manualTakeover = () => {
  addLog('[Auth_Service] 收到高优先级终端权限鉴权请求 (Role=Admin)', 'warning');
  currentMode.value = '人工干预模式 (本地闭环)';
  addLog('[Process_Manager] 已向调度引擎发送强制挂起中断指令', 'warning');
  addLog('[Control_Plane] 控制权限已移交，AI 模型预测流水线进入休眠状态', 'success');
  alert('【系统警告】已获取最高管理权限，当前控制平面对 AI 模型的订阅已解除。');
};

onMounted(() => {
  for (let i = 0; i < 30; i++) {
    timeData.value.push('');
    tempData.value.push(24);
    rawTempData.value.push(24);
    pmvData.value.push(0);
  }
  addLog('平台核心服务启动完毕，通信网关初始化成功，进入标准调度周期。', 'success');
  simInterval = setInterval(simulateCharts, 1000) as unknown as number;
});

onUnmounted(() => {
  clearInterval(simInterval);
});

</script>

<style scoped>
.sandbox-container { padding: 20px; color: #fff; font-family: 'Inter', sans-serif; height: 100%; display: flex; flex-direction: column; gap: 20px; }
.glass { background: rgba(15, 20, 40, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }

/* 状态栏 */
.status-bar { display: flex; align-items: center; padding: 20px 30px; position: relative; overflow: hidden; transition: all 0.5s; }
.status-icon { font-size: 2rem; margin-right: 20px; }
.status-title { font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; font-weight: 500; }
.status-value { font-family: 'Outfit'; font-size: 1.6rem; font-weight: 700; margin-top: 5px; color: #f8fafc; }

.mode-green { border-left: 4px solid var(--neon-green); background: linear-gradient(90deg, rgba(57,255,20,0.05), rgba(0,0,0,0)); }
.mode-orange { border-left: 4px solid #ffb300; background: linear-gradient(90deg, rgba(255,179,0,0.05), rgba(0,0,0,0)); }
.mode-red { border-left: 4px solid var(--danger); background: linear-gradient(90deg, rgba(255,0,60,0.05), rgba(0,0,0,0)); }

.status-pulse { position: absolute; right: 30px; width: 15px; height: 15px; border-radius: 50%; }
.pulse-green { background: var(--neon-green); box-shadow: 0 0 0 0 rgba(57,255,20,0.7); animation: pulsing-green 2s infinite; }
.pulse-orange { background: #ffb300; box-shadow: 0 0 0 0 rgba(255,179,0,0.7); animation: pulsing-orange 1s infinite; }
.pulse-red { background: var(--danger); box-shadow: 0 0 0 0 rgba(255,0,60,0.7); animation: pulsing-red 0.5s infinite; }

@keyframes pulsing-green { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(57,255,20,0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(57,255,20,0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(57,255,20,0); } }
@keyframes pulsing-orange { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,179,0,0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(255,179,0,0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,179,0,0); } }
@keyframes pulsing-red { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,0,60,0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(255,0,60,0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255,0,60,0); } }

/* 布局 */
.sandbox-content { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; flex-grow: 1; min-height: 420px;}
.c-span-8 { grid-column: span 8; display: flex; flex-direction: column; }
.c-span-4 { grid-column: span 4; display: flex; flex-direction: column; }
.flex-col { display: flex; flex-direction: column; }
.h-half { height: 50%; }
.mt-15 { margin-top: 20px; }

/* 图表区 */
.chart-box { padding: 20px; flex-grow: 1; display: flex; flex-direction: column; position: relative; }
.chart-title { font-family: 'Outfit'; font-size: 1.05rem; color: #e2e8f0; margin-bottom: 15px; font-weight: 500; }
.chart-container { flex-grow: 1; width: 100%; min-height: 200px; }
.pmv-overlay { position: absolute; inset: 0; background: rgba(15, 20, 40, 0.8); z-index: 10; display: flex; justify-content: center; align-items: center; font-family: 'Outfit'; font-size: 1.5rem; color: #64748b; letter-spacing: 2px; }

/* 侧边面板 */
.panel-title { padding: 15px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: 'Outfit'; font-weight: 500; color: #e2e8f0; font-size: 1rem; }
.nodes-container { padding: 20px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
.node-item { display: flex; align-items: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); transition: all 0.3s; }
.node-icon { width: 8px; height: 15px; background: #cbd5e1; margin-right: 15px; border-radius: 2px; }
.node-info { flex-grow: 1; }
.node-name { font-weight: 600; font-size: 0.95rem; color: #f8fafc; }
.node-state { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; font-family: 'Consolas', monospace; }
.node-indicator { width: 10px; height: 10px; border-radius: 50%; }

.node-item.online .node-indicator { background: var(--neon-green); }
.node-item.warning .node-indicator { background: #ffb300; animation: blink 0.5s infinite; }
.node-item.offline { border-color: rgba(255,0,60,0.2); background: rgba(255,0,60,0.05); }
.node-item.offline .node-name { color: var(--danger); }
.node-item.offline .node-indicator { background: var(--danger); }

/* 日志区 */
.log-terminal { flex-grow: 1; }
.log-content { padding: 15px; overflow-y: auto; font-family: 'Consolas', monospace; font-size: 0.85rem; display: flex; flex-direction: column; gap: 8px; height: 0; flex-grow: 1; background: #0f172a; border-radius: 0 0 12px 12px; }
.log-line { padding: 6px 10px; border-radius: 4px; border-left: 3px solid transparent; word-break: break-all; line-height: 1.4; }
.log-time { color: #64748b; margin-right: 12px; font-size: 0.8rem; }
.log-msg { color: #e2e8f0; }
.log-line.info { border-left-color: #3b82f6; background: rgba(59, 130, 246, 0.1); }
.log-line.success { border-left-color: #10b981; background: rgba(16, 185, 129, 0.1); }
.log-line.warning { border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.1); }
.log-line.error { border-left-color: #ef4444; background: rgba(239, 68, 68, 0.15); color: #fecaca; }

/* 混沌面板 */
.chaos-panel { padding: 25px; border: 1px solid rgba(255,255,255,0.08); background: rgba(15, 20, 40, 0.6); }
.chaos-title { font-family: 'Outfit'; font-weight: 500; margin-bottom: 20px; color: #e2e8f0; font-size: 1.05rem; }
.chaos-buttons { display: flex; gap: 15px; flex-wrap: wrap; }
.btn { flex: 1; min-width: 220px; padding: 14px 20px; border-radius: 6px; border: 1px solid transparent; cursor: pointer; font-family: 'Inter'; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s; color: #fff; background: rgba(255,255,255,0.05); }
.btn:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.2); }
.btn:active { transform: translateY(1px); }

.btn-warning { border-color: rgba(245, 158, 11, 0.3); color: #fbbf24; }
.btn-warning:hover { background: rgba(245, 158, 11, 0.1); border-color: rgba(245, 158, 11, 0.5); }
.btn-orange { border-color: rgba(234, 88, 12, 0.3); color: #fdba74; }
.btn-orange:hover { background: rgba(234, 88, 12, 0.1); border-color: rgba(234, 88, 12, 0.5); }
.btn-danger { border-color: rgba(239, 68, 68, 0.3); color: #fca5a5; }
.btn-danger:hover { background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.5); }
.btn-primary { border-color: rgba(59, 130, 246, 0.3); color: #93c5fd; }
.btn-primary:hover { background: rgba(59, 130, 246, 0.1); border-color: rgba(59, 130, 246, 0.5); }

@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

/* 滚动条美化 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
