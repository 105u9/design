<template>
  <div class="sandbox-container">
    <div class="sandbox-header glass">
      <h2>分布式系统并发性能与响应延迟监控平台</h2>
      <div class="subtitle">系统吞吐能力实时监测模块 | 评估依据：Apache JMeter 并发负载基准测试规范</div>
    </div>

    <div class="sandbox-content">
      <!-- 控制面板 -->
      <div class="control-panel glass c-span-3">
        <h3>负载测试参数设定 (Test Configuration)</h3>
        
        <div class="control-group">
          <label>测试端点 (API Endpoint)</label>
          <select v-model="selectedScenario" class="custom-select">
            <option value="A">高频物联网感知数据写入 (MQTT/POST)</option>
            <option value="B">环境参数预测模型结果查询 (GET /pmv)</option>
            <option value="C">异步计算密集型控制指令分发 (POST /control)</option>
          </select>
        </div>

        <div class="control-group">
          <label>并发线程数 (Concurrent Threads): {{ concurrentUsers }}</label>
          <input type="range" v-model="concurrentUsers" min="10" max="1000" step="10" class="custom-slider">
          <div class="slider-labels">
            <span>10</span>
            <span>1000</span>
          </div>
        </div>

        <div class="control-group">
          <label>后端架构调度策略</label>
          <div class="toggle-row">
            <span>协程级异步非阻塞 I/O</span>
            <label class="switch">
              <input type="checkbox" v-model="optAsync">
              <span class="slider"></span>
            </label>
          </div>
          <div class="toggle-row">
            <span>内存队列与预写式日志 (WAL)</span>
            <label class="switch">
              <input type="checkbox" v-model="optCache">
              <span class="slider"></span>
            </label>
          </div>
          <div class="opt-status" :class="isFullyOptimized ? 'opt-good' : 'opt-bad'">
            {{ isFullyOptimized ? '[Status] 运行于工业级高并发异步架构' : '[Status] 运行于传统同步阻塞架构' }}
          </div>
        </div>
      </div>

      <!-- 核心监控指标 -->
      <div class="metrics-panel c-span-9">
        <div class="metrics-grid">
          <div class="metric-box glass" :class="{'alert': qps < 100 && concurrentUsers > 200}">
            <div class="metric-title">系统吞吐量 (Requests/sec)</div>
            <div class="metric-value text-neon-blue">{{ qps.toFixed(0) }}</div>
          </div>
          <div class="metric-box glass" :class="{'alert-red': avgLatency > 500}">
            <div class="metric-title">接口平均响应时间 (Avg Latency)</div>
            <div class="metric-value text-neon-green" :class="{'text-danger': avgLatency > 500}">{{ avgLatency.toFixed(1) }} <span class="unit">ms</span></div>
          </div>
          <div class="metric-box glass" :class="{'alert-red': p99Latency > 800}">
            <div class="metric-title">P99 长尾延迟 (99th Percentile)</div>
            <div class="metric-value text-neon-purple" :class="{'text-danger': p99Latency > 800}">{{ p99Latency.toFixed(1) }} <span class="unit">ms</span></div>
          </div>
          <div class="metric-box glass" :class="{'alert-red': errorRate > 0}">
            <div class="metric-title">请求失败/死锁率 (Error Rate)</div>
            <div class="metric-value text-neon-green" :class="{'text-danger': errorRate > 0}">{{ errorRate.toFixed(1) }} <span class="unit">%</span></div>
          </div>
        </div>

        <div class="chart-box glass">
          <div class="chart-title">后端 API 响应时间动态走势图</div>
          <v-chart class="latency-chart" :option="chartOption" autoresize />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import VChart from 'vue-echarts';

const selectedScenario = ref('B');
const concurrentUsers = ref(50);
const optAsync = ref(true);
const optCache = ref(true);

const qps = ref(50);
const avgLatency = ref(5);
const p99Latency = ref(8);
const errorRate = ref(0);

const timeData = ref<string[]>([]);
const latencyData = ref<number[]>([]);

const isFullyOptimized = computed(() => optAsync.value && optCache.value);

const chartOption = computed(() => {
  return {
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(15, 20, 40, 0.8)', textStyle: { color: '#fff' } },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '10%', containLabel: true },
    xAxis: { 
      type: 'category', 
      data: timeData.value,
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
      axisLabel: { color: '#94a3b8' }
    },
    yAxis: { 
      type: 'value', 
      name: '时间 (ms)',
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.2)' } },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
      axisLabel: { color: '#94a3b8' }
    },
    series: [{
      name: '平均响应时间(ms)',
      type: 'line',
      smooth: true,
      data: latencyData.value,
      itemStyle: { color: avgLatency.value > 500 ? '#ff003c' : '#39ff14' },
      lineStyle: { width: 3 },
      areaStyle: {
        color: avgLatency.value > 500 ? 'rgba(255, 0, 60, 0.2)' : 'rgba(57, 255, 20, 0.2)'
      }
    }]
  };
});

let simInterval: number;

const simulateMetrics = () => {
  const users = parseInt(concurrentUsers.value as any);
  const isOpt = isFullyOptimized.value;
  const scenario = selectedScenario.value;

  let targetQps = 0;
  let targetLatency = 0;
  let targetError = 0;

  if (isOpt && scenario === 'B') {
    // 优化的状态，场景B (GET PMV)
    targetQps = users * 0.8 + Math.random() * 20;
    targetLatency = 4 + (users / 1000) * 4 + Math.random() * 2; // 4-8ms
    targetError = 0;
  } else if (!isOpt && (scenario === 'A' || scenario === 'C')) {
    // 未优化的状态，场景A/C
    if (users <= 200) {
      targetQps = users * 0.5;
      targetLatency = 10 + users * 0.5 + Math.random() * 10;
      targetError = 0;
    } else {
      targetQps = 150 + Math.random() * 20; // QPS bottleneck
      targetLatency = Math.pow(users / 200, 2) * 100 + Math.random() * 500; // Exponential growth
      targetError = ((users - 200) / 800) * 10 + Math.random() * 2; // Error rate up to >5%
    }
  } else {
    // Other combinations
    if (isOpt) {
       targetQps = users * 0.6;
       targetLatency = 10 + (users / 1000) * 15 + Math.random() * 5;
       targetError = 0;
    } else {
       targetQps = Math.min(users * 0.6, 300);
       targetLatency = 15 + (users / 500) * 200 + Math.random() * 20;
       targetError = users > 500 ? Math.random() * 2 : 0;
    }
  }

  // Smooth transitions
  qps.value += (targetQps - qps.value) * 0.2;
  avgLatency.value += (targetLatency - avgLatency.value) * 0.2;
  p99Latency.value = avgLatency.value * (1.2 + Math.random() * 0.3);
  errorRate.value = Math.max(0, targetError);

  // Update chart data
  const now = new Date();
  timeData.value.push(`${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}`);
  latencyData.value.push(avgLatency.value);
  
  if (timeData.value.length > 20) {
    timeData.value.shift();
    latencyData.value.shift();
  }
};

onMounted(() => {
  for (let i = 0; i < 20; i++) {
    timeData.value.push('');
    latencyData.value.push(0);
  }
  simInterval = setInterval(simulateMetrics, 1000) as unknown as number;
});

onUnmounted(() => {
  clearInterval(simInterval);
});

</script>

<style scoped>
.sandbox-container { padding: 20px; color: #fff; font-family: 'Inter', sans-serif; height: 100%; display: flex; flex-direction: column; }
.glass { background: rgba(15, 20, 40, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; }

.sandbox-header { padding: 20px 30px; margin-bottom: 20px; border-left: 4px solid var(--neon-blue); }
.sandbox-header h2 { margin: 0 0 10px 0; font-family: 'Outfit'; color: var(--neon-blue); font-size: 1.8rem; font-weight: 600;}
.subtitle { color: #94a3b8; font-size: 0.95rem; }

.sandbox-content { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; flex-grow: 1; }
.c-span-3 { grid-column: span 3; }
.c-span-9 { grid-column: span 9; }

.control-panel { padding: 25px; display: flex; flex-direction: column; gap: 25px; }
.control-panel h3 { margin: 0; font-family: 'Outfit'; font-size: 1.1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; color: #e2e8f0; font-weight: 500;}

.control-group { display: flex; flex-direction: column; gap: 10px; }
.control-group label { color: #cbd5e1; font-weight: 500; font-size: 0.9rem; }

.custom-select { 
  background: rgba(0,0,0,0.5); color: #fff; border: 1px solid rgba(255,255,255,0.2); 
  padding: 10px; border-radius: 6px; outline: none; font-family: 'Inter'; font-size: 0.9rem;
}

.custom-slider { width: 100%; accent-color: var(--neon-blue); }
.slider-labels { display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; }

.toggle-row { display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); }
.toggle-row span { font-size: 0.85rem; color: #e2e8f0; }

.opt-status { padding: 12px; border-radius: 6px; font-size: 0.85rem; text-align: center; font-weight: 600; transition: all 0.3s; }
.opt-good { background: rgba(57, 255, 20, 0.1); color: var(--neon-green); border: 1px solid rgba(57, 255, 20, 0.3); }
.opt-bad { background: rgba(255, 0, 60, 0.1); color: var(--danger); border: 1px solid rgba(255, 0, 60, 0.3); }

/* Switch Toggle css */
.switch { position: relative; display: inline-block; width: 44px; height: 24px; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(255,255,255,0.1); transition: .4s; border-radius: 24px; border: 1px solid rgba(255,255,255,0.2); }
.slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px; background-color: #cbd5e1; transition: .4s; border-radius: 50%; }
input:checked + .slider { background-color: rgba(0, 240, 255, 0.1); border-color: var(--neon-blue); }
input:checked + .slider:before { transform: translateX(20px); background-color: var(--neon-blue); box-shadow: 0 0 10px var(--neon-blue); }

.metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 20px; }
.metric-box { padding: 22px; display: flex; flex-direction: column; justify-content: center; transition: all 0.3s; border-left: 3px solid transparent; }
.metric-box:hover { background: rgba(255,255,255,0.03); }
.metric-title { font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px; font-weight: 500; }
.metric-value { font-size: 2.2rem; font-weight: 800; font-family: 'Outfit'; }
.unit { font-size: 1rem; color: #64748b; font-weight: normal; }

.text-neon-blue { color: var(--neon-blue); }
.text-neon-green { color: var(--neon-green); }
.text-neon-purple { color: var(--neon-purple); }
.text-danger { color: var(--danger) !important; }

.alert-red { border-color: var(--danger); box-shadow: inset 0 0 30px rgba(255, 0, 60, 0.1); border-left: 3px solid var(--danger); }

.chart-box { padding: 20px; height: 350px; display: flex; flex-direction: column; }
.chart-title { font-family: 'Outfit'; font-size: 1.05rem; margin-bottom: 15px; color: #e2e8f0; font-weight: 500; }
.latency-chart { flex-grow: 1; width: 100%; }
</style>
