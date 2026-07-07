# IEEE RTS24 数据生成逻辑说明

本文档说明 `data.py` 如何生成 `dataset_csv` 中的全部数据文件，并给出关键参数与计算公式。  
适用对象：当前项目中的 TNEP+储能联合规划模型（`solve_tnep.py`）。

---

## 1. 总体流程

`data.py` 的执行流程为：

1. 调用 `build_tnep_data()` 基于内置 RTS24 基础数据构造完整数据字典；
2. 调用 `save_tnep_csv_dataset()` 写入 `dataset_csv/*.csv`；
3. 调用 `load_tnep_data_from_csv()` 再读回，生成 `TNEP_DATA`（保证后续建模直接来自 CSV）。

---

## 2. 基础输入数据来源（内置）

以下基础数据在 `load_rts24_minimal_data()` 中硬编码：

- 系统基准容量：`base_mva = 100`
- 母线及电压等级：`buses_kv`
- 在役支路清单：`branches`（含重复回路）
- 机组容量：`(Pmax, Pmin)` 列表
- 机组母线映射：`gen_bus_list`

---

## 3. 网络集合生成逻辑

### 3.1 `buses.csv`

- 字段：`bus`
- 逻辑：从 `buses_kv` 提取母线编号并排序。

### 3.2 `existing_lines.csv`

- 字段：`from_bus, to_bus`
- 逻辑：
  - 保留 `status == 1` 的支路；
  - 去除自环；
  - 将每条线标准化为 `(min(i,j), max(i,j))`；
  - 使用集合去重（双回路合并为同一走廊）。

### 3.3 `candidate_lines.csv`

- 字段：`from_bus, to_bus, kv, hops, length_mile, c_line_usd_per_year`
- 逻辑：
  1. 生成所有母线对 `C(24,2)`；
  2. 删除已在役走廊；
  3. 仅保留最短路径跳数 `<= 4` 的母线对；
  4. 计算每条候选线的电压等级、长度与年化成本。

---

## 4. 成本参数生成逻辑

### 4.1 线路年化成本 `c_line`

1. 电压等级判定：
   - 两端母线均为高压区（`>=200kV`）则 `kv=230`
   - 否则 `kv=138`

2. 线路长度估算（mile）：

\[
L_l = 20 + 15 \times (hops_l - 1)
\]

3. 单位造价（CAPEX）：
   - 138kV：`2,000,000 USD/mile`
   - 230kV：`2,200,000 USD/mile`

4. 年化系数（资本回收系数）：

\[
CRF(r,n)=\frac{r(1+r)^n}{(1+r)^n-1}
\]

其中线路参数为 `r=0.05, n=40`。

5. 候选线年化成本：

\[
c_l^{line} = CAPEX_{kv(l)} \times CRF(0.05,40) \times L_l
\]

---

### 4.2 储能容量年化成本 `c_cap`

- 储能 CAPEX：`247,000 USD/MWh`
- 年化参数：`r=0.08, n=15`

\[
c^{cap} = 247000 \times CRF(0.08,15)
\]

结果写入 `global_params.csv`。

---

### 4.3 其他全局经济参数（`global_params.csv`）

- `c_shed = 10000 USD/MWh`
- `c_curt = 50 USD/MWh`
- `Gamma = 15000000 USD/year`
- `rho = 0.5 MW/MWh`
- `eta_c = 0.922`
- `eta_d = 0.922`
- `delta_t = 4 hour`
- `base_mva = 100`

并附带 assumptions（折现率、寿命、单位造价等）。

---

## 5. 机组数据生成逻辑（`generators.csv`）

字段：`g, bus, pmax, pmin, c_gen_usd_per_mwh`

- `g`：机组编号（1..33）
- `bus`：来自 `gen_bus_list`
- `pmax/pmin`：来自内置 RTS24 数据
- `c_gen`：依据 `(Pmax,Pmin)` 查表映射

示例映射：

- `(20,16) -> 48.5`
- `(76,15.2) -> 15.3`
- `(155,54.3) -> 12.3`
- `(350,140) -> 11.4`
- `(400,100)` 或 `(400,400) -> 6.3`

若签名未命中，默认 `20.0 USD/MWh`。

---

## 6. 储能站点数据生成逻辑（`storage_sites.csv`）

字段：`h, Ebar_mwh, c_fix_usd_per_year`

当前候选储能站点固定为：

- bus 6:  `Ebar=300`, `c_fix=180000`
- bus 10: `Ebar=300`, `c_fix=180000`
- bus 15: `Ebar=400`, `c_fix=220000`
- bus 18: `Ebar=350`, `c_fix=200000`
- bus 21: `Ebar=350`, `c_fix=200000`

---

## 7. 场景与时段数据生成逻辑

### 7.1 `scenarios.csv`

- 场景：`low_ren`, `base`, `high_ren`
- 权重（days/year）：`120, 160, 85`

### 7.2 `periods.csv`

- 时段：`t = 1..6`
- 每时段长度：`delta_t = 4h`（写入 `global_params.csv`）

---

## 8. 负荷数据生成逻辑（`demand.csv`）

字段：`bus, scenario, t, demand_mw`

计算公式：

\[
D_{b,s,t} = base\_pd_b \times time\_mult_t \times scen\_mult_s
\]

其中：

- `base_pd_b`：RTS24 母线基准负荷
- `time_mult = {1:0.85, 2:0.95, 3:1.05, 4:1.00, 5:0.90, 6:0.80}`
- `scen_mult = {low_ren:0.95, base:1.00, high_ren:1.08}`

---

## 9. 可再生数据生成逻辑

### 9.1 `renewables.csv`

字段：`r, bus, capacity_mw`

- `R1 @ bus8, cap=220`
- `R2 @ bus16, cap=180`
- `R3 @ bus21, cap=200`

### 9.2 `renewable_availability.csv`

字段：`r, scenario, t, wbar_mw`

计算公式：

\[
\bar{W}_{r,s,t} = cap_r \times cf_{s,t}
\]

容量因子矩阵：

- `low_ren`: `[0.20, 0.35, 0.45, 0.35, 0.25, 0.15]`
- `base`:    `[0.35, 0.55, 0.75, 0.60, 0.40, 0.25]`
- `high_ren`:`[0.45, 0.65, 0.85, 0.70, 0.50, 0.35]`

---

## 10. 读回规则（`load_tnep_data_from_csv`）

程序会把 11 个 CSV 全部读回，恢复为建模字典：

- 集合：`buses`, `existing_lines`, `candidate_lines`, `storage_sites`, `scenarios`, `periods`
- 成本：`c_line`, `c_gen`, `c_cap`, `c_shed`, `c_curt`, `Gamma`, `c_fix(h)`
- 运行参数：`demand`, `wbar`, `rho`, `eta_c`, `eta_d`, `delta_t`, `base_mva`
- 元数据：`line_meta`, `gen_meta`, `storage_meta`, `assumptions`

这保证了 `solve_tnep.py` 使用的参数来自 CSV，而不是运行时硬编码。

---

## 11. 可调参数清单（建议用于敏感性分析）

建议优先做敏感性分析的字段：

- `line_discount_rate`, `line_lifetime_years`
- `bess_discount_rate`, `bess_lifetime_years`
- `Gamma`
- `c_shed`, `c_curt`
- `max_graph_hops`（候选线规模）
- `scenarios.csv` 中 `omega_days`
- `demand.csv`、`renewable_availability.csv`（压力场景）

