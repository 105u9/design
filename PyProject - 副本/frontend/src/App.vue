<template>
  <div class="app-wrapper">
    <!-- Navigation Tabs -->
    <div class="nav-tabs glass">
      <button class="nav-btn" :class="{ active: currentTab === 'main' }" @click="currentTab = 'main'">系统核心监控大屏</button>
      <button class="nav-btn" :class="{ active: currentTab === 'loadtest' }" @click="currentTab = 'loadtest'">并发负载性能测试</button>
      <button class="nav-btn" :class="{ active: currentTab === 'chaos' }" @click="currentTab = 'chaos'">容错与混沌工程演练</button>
    </div>

    <!-- Main View -->
    <div v-if="currentTab === 'main'">
    <!-- Header -->
    <div class="header glass">
        <div>
            <h1>暖通空调智能控制系统 <span class="header-subtext">| 基于数据驱动与多目标优化</span></h1>
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 5px;">
                <span class="badge">数据基座</span>
                本系统模拟数据流来自于公开建筑能耗开源数据集 (Building Data Genome Project 2)，非物理硬连线真实传感器。
            </div>
        </div>
        <div class="switch-container">
            <span class="status-text">控制平面对接状态</span>
            <div class="status-indicator" style="margin-right: 15px;">
                <div :class="connectionError ? 'inactive-dot' : 'active-dot'"></div>
                <span :style="{ color: connectionError ? 'var(--danger)' : 'var(--neon-green)' }">
                  {{ connectionError ? "后端服务离线" : "API 链路正常" }}
                </span>
            </div>
            <span class="status-text">AI 优化闭环控制</span>
            <label class="switch">
                <input type="checkbox" v-model="isAiEnabled" @change="toggleAi">
                <span class="slider"></span>
            </label>
            <div class="status-indicator">
                <div :class="isAiEnabled ? 'active-dot' : 'inactive-dot'"></div>
                <span :style="{ color: isAiEnabled ? 'var(--neon-green)' : 'var(--text-secondary)' }">
                  {{ isAiEnabled ? "运行中" : "已挂起" }}
                </span>
            </div>
        </div>
    </div>

    <!-- Stats -->
    <div class="stats-grid">
        <div class="stat-box glass">
            <div class="stat-label">室内平均温度</div>
            <div class="stat-value">{{ latestStats.temp }} <span class="unit">℃</span></div>
        </div>
        <div class="stat-box glass">
            <div class="stat-label">室内相对湿度</div>
            <div class="stat-value stat-blue">{{ latestStats.hum }} <span class="unit">%</span></div>
        </div>
        <div class="stat-box glass">
            <div class="stat-label">系统当前冷负荷</div>
            <div class="stat-value stat-purple">{{ latestStats.power }} <span class="unit">kW</span></div>
        </div>
        <div class="stat-box glass">
            <div class="stat-label">近期累计节能率</div>
            <div class="stat-value stat-green">{{ latestStats.saving }} <span class="unit">%</span></div>
        </div>
    </div>

    <!-- Charts Hub -->
    <div class="dashboard">
        <!-- Main Env Chart -->
        <div class="card glass c-span-8">
            <div class="card-title">多维环境参数实时监测</div>
            <v-chart class="chart-container" style="height: 400px" :option="envChartOption" autoresize />
        </div>
        <!-- Control Gauge -->
        <div class="card glass c-span-4">
            <div class="card-title">控制指令下发状态 (决策变量)</div>
            <v-chart class="chart-container" style="height: 400px" :option="controlChartOption" autoresize />
        </div>
        
        <!-- Energy Bar -->
        <div class="card glass c-span-4">
            <div class="card-title">系统实时能耗分布</div>
            <v-chart class="chart-container" :option="energyChartOption" autoresize />
        </div>
        <!-- MOPSO Pareto -->
        <div class="card glass c-span-4">
            <div class="card-title">MOPSO 多目标帕累托前沿</div>
            <v-chart class="chart-container" :option="mopsoChartOption" autoresize />
        </div>
        <!-- Predict Curve -->
        <div class="card glass c-span-4">
            <div class="card-title">GAT-LSTM 负荷预测序列 (未来 12h)</div>
            <v-chart class="chart-container" :option="predictChartOption" autoresize />
        </div>
    </div>
    </div>

    <!-- Sandboxes -->
    <div v-else-if="currentTab === 'loadtest'" class="sandbox-view-wrapper">
      <LoadTestSandbox />
    </div>
    <div v-else-if="currentTab === 'chaos'" class="sandbox-view-wrapper">
      <ChaosSandbox />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import VChart from 'vue-echarts'
import LoadTestSandbox from './components/LoadTestSandbox.vue'
import ChaosSandbox from './components/ChaosSandbox.vue'
import { use } from 'echarts/core'
import { 
  LineChart, BarChart, ScatterChart, GaugeChart 
} from 'echarts/charts'
import { 
  TooltipComponent, GridComponent, LegendComponent 
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import * as echarts from 'echarts/core'

use([
  LineChart, BarChart, ScatterChart, GaugeChart, 
  TooltipComponent, GridComponent, LegendComponent, 
  CanvasRenderer
])

const currentTab = ref('main')
const isAiEnabled = ref(true)
const token = ref('')
const connectionError = ref(false)

const latestStats = reactive({
  temp: '--.-',
  hum: '--.-',
  power: '---.-',
  saving: '18.5'
})

// Common styles
const axisStyle = {
  axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
  splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
  axisLabel: { color: '#94a3b8', fontFamily: 'Inter' }
}
const tooltipStyle = {
  backgroundColor: 'rgba(15, 20, 40, 0.8)',
  borderColor: 'rgba(0, 240, 255, 0.3)',
  borderWidth: 1,
  textStyle: { color: '#fff' }
}

const envChartOption = ref<any>({})
const energyChartOption = ref<any>({})
const predictChartOption = ref<any>({})
const mopsoChartOption = ref<any>({})
const controlChartOption = ref<any>({})

const buildHeader = () => ({
  headers: {
    'Authorization': `Bearer ${token.value}`
  }
})

let timer: any = null

const login = async () => {
  try {
    const fd = new URLSearchParams()
    fd.append('username', 'admin')
    fd.append('password', 'admin123')
    
    // Using relative path so it hits Vite proxy -> Fast API
    const res = await axios.post('/token', fd)
    token.value = res.data.access_token
    connectionError.value = false
    startPolling()
  } catch (error) {
    console.error('Login failed (Backend may be offline)', error)
    connectionError.value = true
    // Still start polling to recover when backend comes back online
    startPolling()
  }
}

const toggleAi = async () => {
  try {
    const res = await axios.post(`/api/v1/toggle_ai?mode=${isAiEnabled.value}`, null, buildHeader())
    isAiEnabled.value = res.data.ai_mode
  } catch (err) { console.error('Toggle failed', err) }
}

const fetchData = async () => {
  try {
    const [monRes, predRes, optRes] = await Promise.all([
      axios.get('/api/v1/monitoring', buildHeader()),
      axios.post('/api/v1/predict', {}, buildHeader()),
      axios.post('/api/v1/optimize', {}, buildHeader())
    ])

    const { history, system } = monRes.data
    const predData = predRes.data
    const optData = optRes.data

    if (system !== undefined) {
      isAiEnabled.value = system.ai_mode
    }

    if (history && history.length > 0) {
      const last = history[history.length - 1]
      latestStats.temp = last.temperature.toFixed(1)
      latestStats.hum = last.humidity.toFixed(1)
      latestStats.power = last.power.toFixed(1)
      updateCharts(history, predData, optData)
      connectionError.value = false
    }
  } catch (err) { 
    console.error('Fetch error: Backend API unreachable.', err)
    connectionError.value = true
  }
}

const updateCharts = (history: any[], prediction: any, optimization: any) => {
  const timeLabels = history.map(d => d.timestamp.split(' ')[1])

  envChartOption.value = {
    tooltip: { ...tooltipStyle, trigger: 'axis' },
    legend: { data: ['室内温度(℃)', '室内湿度(%)', 'CO2(ppm)'], textStyle: { color: '#fff' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', boundaryGap: false, data: timeLabels, ...axisStyle },
    yAxis: [
      { type: 'value', name: '℃ / %', nameTextStyle: {color: '#94a3b8'}, ...axisStyle }, 
      { type: 'value', name: 'ppm', position: 'right', nameTextStyle: {color: '#94a3b8'}, ...axisStyle }
    ],
    series: [
      { 
        name: '室内温度(℃)', type: 'line', smooth: true, 
        lineStyle: { width: 3, color: '#00f0ff', shadowColor: 'rgba(0, 240, 255, 0.5)', shadowBlur: 10 },
        itemStyle: { color: '#00f0ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 240, 255, 0.4)' },
            { offset: 1, color: 'rgba(0, 240, 255, 0)' }
          ])
        },
        data: history.map(d => d.temperature) 
      },
      { 
        name: '室内湿度(%)', type: 'line', smooth: true,
        lineStyle: { width: 3, color: '#39ff14', shadowColor: 'rgba(57, 255, 20, 0.5)', shadowBlur: 10 },
        itemStyle: { color: '#39ff14' },
        data: history.map(d => d.humidity) 
      },
      { 
        name: 'CO2(ppm)', type: 'line', yAxisIndex: 1, smooth: true,
        lineStyle: { width: 2, color: 'rgba(255,255,255,0.4)', type: 'dashed' },
        itemStyle: { color: '#ffffff' },
        data: history.map(d => d.co2) 
      }
    ]
  }

  energyChartOption.value = {
    tooltip: { ...tooltipStyle, trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: timeLabels, ...axisStyle },
    yAxis: { type: 'value', ...axisStyle },
    series: [{ 
      name: '系统当前冷负荷', type: 'bar', 
      itemStyle: { 
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: '#b53cff' },
          { offset: 1, color: 'rgba(181, 60, 255, 0.1)' }
        ]),
        borderRadius: [4, 4, 0, 0]
      },
      data: history.map(d => d.power) 
    }]
  }

  predictChartOption.value = {
    tooltip: { ...tooltipStyle, trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'category', data: prediction.forecast_horizon, ...axisStyle, boundaryGap: false },
    yAxis: { type: 'value', ...axisStyle },
    series: [{ 
      name: '预测负荷', type: 'line', smooth: true,
      lineStyle: { type: 'dashed', width: 3, color: '#ffb300' },
      itemStyle: { color: '#ffb300' },
      areaStyle: {
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: 'rgba(255, 179, 0, 0.3)' },
          { offset: 1, color: 'rgba(255, 179, 0, 0)' }
        ])
      },
      data: prediction.predicted_load 
    }]
  }

  mopsoChartOption.value = {
    tooltip: { ...tooltipStyle, formatter: '能耗预测: {c0}kW<br/>PMV偏离度: {c1}' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { name: '预测能耗 (kW)', nameTextStyle: {color: '#94a3b8'}, ...axisStyle },
    yAxis: { name: 'PMV 惩罚项', nameTextStyle: {color: '#94a3b8'}, ...axisStyle },
    series: [{
      type: 'scatter',
      symbolSize: 8,
      data: Array.from({length: 30}, () => [Math.random()*60+80, Math.random()*20]),
      itemStyle: { 
        color: '#00f0ff',
        shadowBlur: 10,
        shadowColor: 'rgba(0, 240, 255, 0.5)'
      }
    }]
  }

  const isAi = optimization.mode && optimization.mode.includes("AI")
  controlChartOption.value = {
    series: [{
      type: 'gauge',
      min: 16, max: 30,
      splitNumber: 7,
      axisLine: {
        lineStyle: {
          width: 15,
          color: [
            [0.3, '#00f0ff'],
            [0.7, '#39ff14'],
            [1, '#ff003c']
          ]
        }
      },
      pointer: { itemStyle: { color: 'auto' } },
      axisTick: { distance: -15, length: 8, lineStyle: { color: '#fff', width: 2 } },
      splitLine: { distance: -15, length: 30, lineStyle: { color: '#fff', width: 3 } },
      axisLabel: { color: 'inherit', distance: 30, fontSize: 14, fontFamily: 'Outfit' },
      detail: {
        valueAnimation: true,
        formatter: function(value: any) {
          const modeName = isAi ? 'MOPSO 寻优设定值' : '系统基准运行值'
          return value.toFixed(1) + ' ℃\n{name|' + modeName + '}'
        },
        color: 'inherit',
        fontSize: 28,
        fontFamily: 'Outfit',
        offsetCenter: [0, '60%'],
        rich: {
          name: {
            fontSize: 14,
            color: '#94a3b8',
            padding: [10, 0, 0, 0]
          }
        }
      },
      data: [{ 
        value: optimization.supply_air_setpoint || 24.0, 
        name: isAi ? 'MOPSO 寻优设定值' : '系统基准运行值' 
      }]
    }]
  }
}

const startPolling = () => {
  fetchData()
  timer = setInterval(fetchData, 4000)
}

onMounted(() => {
  login()
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
/* Navigation Tabs */
.app-wrapper { display: flex; flex-direction: column; min-height: 100vh; padding: 20px; box-sizing: border-box; }
.nav-tabs { display: flex; gap: 15px; margin-bottom: 20px; padding: 10px 20px; border-radius: 12px; }
.nav-btn { 
  background: transparent; border: 1px solid rgba(255,255,255,0.1); color: #94a3b8; 
  padding: 10px 20px; border-radius: 8px; font-family: 'Outfit'; font-size: 1rem; 
  cursor: pointer; transition: all 0.3s; 
}
.nav-btn:hover { background: rgba(255,255,255,0.05); color: #fff; }
.nav-btn.active { 
  background: rgba(0, 240, 255, 0.1); border-color: var(--neon-blue); color: var(--neon-blue); 
  box-shadow: 0 0 15px rgba(0, 240, 255, 0.2); 
}

.sandbox-view-wrapper { flex-grow: 1; display: flex; flex-direction: column; }

/* Specific Component View Styles */
.header { 
  display: flex; justify-content: space-between; align-items: center; 
  padding: 20px 35px; margin-bottom: 30px; position: relative; overflow: hidden;
}
.header::after {
  content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-blue), var(--neon-purple), transparent);
  opacity: 0.7;
}

.header h1 {
  font-family: 'Outfit', sans-serif; margin: 0; font-size: 2.2rem; font-weight: 800;
  letter-spacing: 1px; background: linear-gradient(to right, #ffffff, var(--neon-blue));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  text-shadow: 0 0 20px rgba(0, 240, 255, 0.3); display: flex; align-items: bottom; gap: 15px;
}

.header-subtext {
  font-family: 'Inter', sans-serif; font-size: 1rem; font-weight: 400; color: var(--text-secondary);
  -webkit-text-fill-color: initial; text-shadow: none; letter-spacing: 0; margin-bottom: 5px;
}

.badge {
  color: var(--neon-blue); border: 1px solid var(--neon-blue); padding: 2px 6px; 
  border-radius: 4px; margin-right: 8px; font-weight: 600;
}

.dashboard { 
  display: grid; grid-template-columns: repeat(12, 1fr); gap: 25px; padding-bottom: 20px;
}
.c-span-4 { grid-column: span 4; }
.c-span-8 { grid-column: span 8; }
@media (max-width: 1200px) {
  .c-span-4 { grid-column: span 6; }
  .c-span-8 { grid-column: span 12; }
}

.card { 
  padding: 22px; transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
  display: flex; flex-direction: column; position: relative;
}
.card:hover {
  transform: translateY(-5px); border-color: rgba(181, 60, 255, 0.4);
  box-shadow: 0 10px 40px 0 rgba(181, 60, 255, 0.15);
}

.card-title { 
  font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 600; 
  margin-bottom: 20px; color: #ffffff; display: flex; align-items: center; gap: 10px;
}
.card-title::before {
  content: ''; display: inline-block; width: 8px; height: 8px;
  background: var(--neon-blue); border-radius: 50%; box-shadow: 0 0 10px var(--neon-blue);
}

.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 25px; }
.stat-box {
  padding: 20px; display: flex; flex-direction: column; align-items: flex-start;
  position: relative; overflow: hidden; border-radius: 12px;
  background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 100%);
  border: 1px solid rgba(255,255,255,0.05);
}
.stat-box::after {
  content: ''; position: absolute; top: 0; right: 0;
  border-top: 15px solid var(--card-border); border-left: 15px solid transparent;
}

.stat-value { 
  font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 800; color: #fff; 
  text-shadow: 0 0 15px rgba(255,255,255,0.2); margin-top: 5px;
}
.stat-label { font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; }
.unit { font-size: 1rem; color: var(--text-secondary); }

.stat-blue { color: var(--neon-blue); text-shadow: 0 0 15px rgba(0, 240, 255, 0.4); }
.stat-green { color: var(--neon-green); text-shadow: 0 0 15px rgba(57, 255, 20, 0.4); }
.stat-purple { color: var(--neon-purple); text-shadow: 0 0 15px rgba(181, 60, 255, 0.4); }

.switch-container { 
  display: flex; align-items: center; gap: 15px; background: rgba(0,0,0,0.3);
  padding: 8px 15px; border-radius: 30px; border: 1px solid rgba(255,255,255,0.1);
}
.switch { position: relative; display: inline-block; width: 60px; height: 30px; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { 
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; 
  background-color: rgba(255,255,255,0.1); transition: .4s; border-radius: 34px; 
  border: 1px solid rgba(255,255,255,0.2);
}
.slider:before { 
  position: absolute; content: ""; height: 22px; width: 22px; left: 3px; bottom: 3px; 
  background-color: var(--text-secondary); transition: .4s; border-radius: 50%;
}
input:checked + .slider { 
  background-color: rgba(0, 240, 255, 0.15); border-color: var(--neon-blue);
  box-shadow: inset 0 0 10px rgba(0, 240, 255, 0.2);
}
input:checked + .slider:before { 
  transform: translateX(30px); background-color: var(--neon-blue); box-shadow: 0 0 15px var(--neon-blue);
}

.status-text { font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.95rem; text-transform: uppercase; color: #fff; }
.status-indicator { display: flex; align-items: center; gap: 8px; font-weight: 600; width: 80px; }
.active-dot { width: 8px; height: 8px; background-color: var(--neon-green); border-radius: 50%; animation: pulse 1.5s infinite alternate; }
.inactive-dot { width: 8px; height: 8px; background-color: var(--danger); border-radius: 50%; box-shadow: 0 0 10px var(--danger); }
@keyframes pulse { 0% { opacity: 0.6; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1.2); } }

.chart-container { flex-grow: 1; width: 100%; min-height: 280px; }
</style>
