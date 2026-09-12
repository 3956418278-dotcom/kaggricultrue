# Kaggriculture D5–D10 固定行走机（Codex 直接实现版）

> **替换当前 D5–D10 `_make_programme_plan()` / `try_group()` / intraday。** 运行时只允许查静态 preset，不允许优化、重排、找最近格、自己缩 target。

## 1. 唯一允许的运行时逻辑

```text
(day, shop history, actual cash) -> preset CSV 中最高可支付 LOW/BASE/HIGH
preset_id -> preset JSON 中固定 target、固定坐标、固定 worker tile lists、固定 prebuild
固定 visit list -> 仅用现有 _route(x-first, then y) 机械展开
```

若 cash 连 LOW 都不到，只进入 `SURVIVAL`：maintenance + feed + water + Wheat renewal；不新增永久资产。

## 2. 正确时间轴

- D1–D4：保留现有 `scripted_opening.py`。
- D4 已知 shop1，但 opening 已固定，不改 D4。
- D5–D6：只知道 shop1。
- **D7：shop2 reveal。**
- D8–D9：shop1+shop2。
- **D10：shop3 reveal。**
- D11：另一个固定 liquidation 日，exactly 10 total workers。

Wheat cohort 固定：D5 全部新 Wheat=`wday=5`；D6 不动；D7 重种 12 格为 `D10_RESERVE_W(wday=7)`；D9 清掉所有剩余 `wday=5`；D10 先吃 `wday=7` reserve。

## 3. D5 唯一起点

固定资产：`C2 S1 G1 + 12 Melon + 11 opening Wheat + shed 24 Carrot + weed(8,0) + NW/NE`。

cash **不写死110**；只读取实际值。D5 priority Melon：
`(4,3),(5,3),(6,4),(5,4),(4,4),(3,4)`。

## 4. D5 固定市场时序

```text
t0 farmer: NORTH
t0 market:
  SELL CARROT 24
  HIRE ×7
  BUY_PRODUCT WHEAT 4

t1:
  看真实 K1，只沿 shop1.t1 ladder 从头执行最长可支付前缀

t15:
  SELL FERTILIZER 4
  SELL EGG 3

t16:
  看真实 K16，只沿 shop1.t16 ladder
  同时 BUY_SEED WHEAT = 该 shop 的 unconditional_w
  若 preset 需要9/10 workers，只能在这里额外 HIRE 1/2

t19:
  11格 opening Wheat 已全部 DROP
  保留7 Wheat给D6开场
  SELL 其余 Wheat

t20:
  看真实 K20，只沿 shop1.t20 ladder
  做固定 D6 prebuild
  reserved slot 没被永久资产占用 -> 买 Wheat seed

t21–t23:
  只做 PLACE / PLANT / WATER / Wheat-fill
  禁止再买动物
```

## 5. D5 公共 8-worker 逐 action 路线

### W0
```text
t01  WATER
t02  EAST
t03  WATER
t04  EAST
t05  SOUTH
t06  WATER
t07  WEST
t08  WATER
t09  WEST
t10  WATER
t11  WEST
t12  WATER
t13  EAST
t14  EAST
t15  EAST
t16  EAST
t17  EAST
t18  NORTH
t19  NORTH
t20  NORTH
t21  NORTH
t22  DIG
```

### W1
```text
t01  PICKUP WHEAT 2
t02  NORTH
t03  NORTH
t04  NORTH
t05  FEED
t06  CARE
t07  COLLECT_FERTILIZER
t08  EAST
t09  FEED
t10  CARE
t11  COLLECT_FERTILIZER
t12  SOUTH
t13  SOUTH
t14  SOUTH
t15  DROP
```

### W2
```text
t01  PICKUP WHEAT 1
t02  WEST
t03  WEST
t04  NORTH
t05  NORTH
t06  FEED
t07  CARE
t08  COLLECT_FERTILIZER
t09  HARVEST
t10  EAST
t11  SOUTH
t12  SOUTH
t13  DROP
```

### W3
```text
t01  PICKUP WHEAT 1
t02  WEST
t03  WEST
t04  NORTH
t05  NORTH
t06  FEED
t07  CARE
t08  COLLECT_FERTILIZER
t09  EAST
t10  EAST
t11  SOUTH
t12  DROP
```

### W4
```text
t01  EAST
t02  EAST
t03  EAST
t04  NORTH
t05  NORTH
t06  WATER
t07  HARVEST
t08  SOUTH
t09  WATER
t10  HARVEST
t11  EAST
t12  WATER
t13  HARVEST
t14  WEST
t15  WEST
t16  WEST
t17  WEST
t18  DROP
```

### W5
```text
t01  WEST
t02  WEST
t03  NORTH
t04  NORTH
t05  WATER
t06  HARVEST
t07  EAST
t08  EAST
t09  EAST
t10  EAST
t11  EAST
t12  SOUTH
t13  WATER
t14  HARVEST
t15  WEST
t16  WEST
t17  SOUTH
t18  DROP
```

### W6
```text
t01  EAST
t02  NORTH
t03  NORTH
t04  NORTH
t05  WATER
t06  HARVEST
t07  WEST
t08  NORTH
t09  WATER
t10  HARVEST
t11  WEST
t12  WATER
t13  HARVEST
t14  SOUTH
t15  SOUTH
t16  SOUTH
t17  SOUTH
t18  DROP
```

### W7
```text
t01  WEST
t02  WEST
t03  WEST
t04  NORTH
t05  NORTH
t06  WATER
t07  HARVEST
t08  WEST
t09  SOUTH
t10  WATER
t11  HARVEST
t12  EAST
t13  WATER
t14  HARVEST
t15  EAST
t16  EAST
t17  EAST
t18  DROP
```

固定结果：W1/W2/W3 的 F/Egg 在 t15 market 前回 shed；W4–W7 把 11 格 opening Wheat 全部在 t19 前 DROP；W0 完成6个 priority Melon和 weed。

## 6. D5 八条 shop1 投资链

每个 checkpoint 只允许**从头取前缀**，绝对不能跳过前项买后项。

### BAKERY
**t1**
- `K >= 300`：累计做到 `G@(6, 2)`（本 checkpoint 累计 300）
**t16**
- `K >= 300`：累计做到 `G@(7, 3)`（本 checkpoint 累计 300）
**t20**
- `K >= 300`：累计做到 `G@(1, 4)`（本 checkpoint 累计 300）
- t16 固定 unconditional Wheat zone：**31 格**。
- D6 prebuild：无

### PIZZA_SHOP
**t1**
- `K >= 400`：累计做到 `C@(6, 2)`（本 checkpoint 累计 400）
**t16**
- `K >= 400`：累计做到 `C@(7, 3)`（本 checkpoint 累计 400）
- `K >= 450`：累计做到 `T@(9, 0)`（本 checkpoint 累计 450）
- `K >= 500`：累计做到 `T@(9, 1)`（本 checkpoint 累计 500）
**t20**
- `K >= 400`：累计做到 `C@(8, 4)`（本 checkpoint 累计 400）
- `K >= 450`：累计做到 `T@(9, 2)`（本 checkpoint 累计 450）
- `K >= 500`：累计做到 `T@(9, 3)`（本 checkpoint 累计 500）
- t16 固定 unconditional Wheat zone：**26 格**。
- D6 prebuild：C@(7, 2)

### BRUNCH_SPOT
**t1**
- `K >= 300`：累计做到 `G@(6, 2)`（本 checkpoint 累计 300）
- `K >= 400`：累计做到 `ST@(0, 0)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(9, 0)`（本 checkpoint 累计 500）
**t16**
- `K >= 300`：累计做到 `G@(7, 3)`（本 checkpoint 累计 300）
- `K >= 400`：累计做到 `ST@(0, 1)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(1, 0)`（本 checkpoint 累计 500）
**t20**
- `K >= 100`：累计做到 `ST@(8, 0)`（本 checkpoint 累计 100）
- `K >= 200`：累计做到 `ST@(9, 1)`（本 checkpoint 累计 200）
- `K >= 300`：累计做到 `ST@(0, 2)`（本 checkpoint 累计 300）
- `K >= 400`：累计做到 `ST@(1, 1)`（本 checkpoint 累计 400）
- t16 固定 unconditional Wheat zone：**23 格**。
- D6 prebuild：G@(1, 4)

### YARN_STORE
**t1**
- `K >= 500`：累计做到 `S@(1, 4)`（本 checkpoint 累计 500）
**t16**
- `K >= 500`：累计做到 `S@(2, 2)`（本 checkpoint 累计 500）
**t20**
- `K >= 500`：累计做到 `S@(3, 1)`（本 checkpoint 累计 500）
- t16 固定 unconditional Wheat zone：**30 格**。
- D6 prebuild：S@(1, 3)

### ICE_CREAM_SHOP
**t1**
- `K >= 400`：累计做到 `C@(6, 2)`（本 checkpoint 累计 400）
**t16**
- `K >= 400`：累计做到 `C@(7, 3)`（本 checkpoint 累计 400）
**t20**
- `K >= 400`：累计做到 `C@(8, 4)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(0, 0)`（本 checkpoint 累计 500）
- `K >= 600`：累计做到 `ST@(9, 0)`（本 checkpoint 累计 600）
- `K >= 700`：累计做到 `ST@(0, 1)`（本 checkpoint 累计 700）
- `K >= 800`：累计做到 `ST@(1, 0)`（本 checkpoint 累计 800）
- t16 固定 unconditional Wheat zone：**26 格**。
- D6 prebuild：C@(7, 2)

### PET_CAFE
**t1**
- `K >= 300`：累计做到 `G@(6, 2)`（本 checkpoint 累计 300）
- `K >= 320`：累计做到 `CR@(9, 0)`（本 checkpoint 累计 320）
- `K >= 340`：累计做到 `CR@(9, 1)`（本 checkpoint 累计 340）
- `K >= 360`：累计做到 `CR@(9, 2)`（本 checkpoint 累计 360）
- `K >= 380`：累计做到 `CR@(9, 3)`（本 checkpoint 累计 380）
**t16**
- `K >= 300`：累计做到 `G@(7, 3)`（本 checkpoint 累计 300）
- `K >= 320`：累计做到 `CR@(9, 4)`（本 checkpoint 累计 320）
- `K >= 340`：累计做到 `CR@(8, 0)`（本 checkpoint 累计 340）
- `K >= 360`：累计做到 `CR@(8, 1)`（本 checkpoint 累计 360）
- `K >= 380`：累计做到 `CR@(8, 2)`（本 checkpoint 累计 380）
- `K >= 400`：累计做到 `CR@(8, 3)`（本 checkpoint 累计 400）
- `K >= 420`：累计做到 `CR@(8, 4)`（本 checkpoint 累计 420）
**t20**
- `K >= 20`：累计做到 `CR@(7, 0)`（本 checkpoint 累计 20）
- `K >= 40`：累计做到 `CR@(7, 1)`（本 checkpoint 累计 40）
- `K >= 60`：累计做到 `CR@(7, 2)`（本 checkpoint 累计 60）
- `K >= 80`：累计做到 `CR@(6, 0)`（本 checkpoint 累计 80）
- t16 固定 unconditional Wheat zone：**18 格**。
- D6 prebuild：无

### SMOOTHIE_SHOP
**t1**
- `K >= 400`：累计做到 `C@(6, 2)`（本 checkpoint 累计 400）
**t16**
- `K >= 400`：累计做到 `C@(7, 3)`（本 checkpoint 累计 400）
**t20**
- `K >= 400`：累计做到 `C@(8, 4)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(0, 0)`（本 checkpoint 累计 500）
- `K >= 600`：累计做到 `ST@(9, 0)`（本 checkpoint 累计 600）
- `K >= 700`：累计做到 `ST@(0, 1)`（本 checkpoint 累计 700）
- `K >= 800`：累计做到 `ST@(1, 0)`（本 checkpoint 累计 800）
- t16 固定 unconditional Wheat zone：**26 格**。
- D6 prebuild：C@(7, 2)

### FARMERS_MARKET
**t1**
- `K >= 300`：累计做到 `G@(6, 2)`（本 checkpoint 累计 300）
- `K >= 400`：累计做到 `ST@(0, 0)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(9, 0)`（本 checkpoint 累计 500）
**t16**
- `K >= 300`：累计做到 `G@(7, 3)`（本 checkpoint 累计 300）
- `K >= 400`：累计做到 `ST@(0, 1)`（本 checkpoint 累计 400）
- `K >= 500`：累计做到 `ST@(1, 0)`（本 checkpoint 累计 500）
- `K >= 520`：累计做到 `CR@(9, 2)`（本 checkpoint 累计 520）
- `K >= 540`：累计做到 `CR@(9, 3)`（本 checkpoint 累计 540）
**t20**
- `K >= 100`：累计做到 `ST@(8, 0)`（本 checkpoint 累计 100）
- `K >= 200`：累计做到 `ST@(9, 1)`（本 checkpoint 累计 200）
- `K >= 220`：累计做到 `CR@(9, 4)`（本 checkpoint 累计 220）
- `K >= 240`：累计做到 `CR@(8, 1)`（本 checkpoint 累计 240）
- `K >= 260`：累计做到 `CR@(8, 2)`（本 checkpoint 累计 260）
- `K >= 280`：累计做到 `CR@(8, 3)`（本 checkpoint 累计 280）
- t16 固定 unconditional Wheat zone：**20 格**。
- D6 prebuild：无

## 7. D6 固定日

- BASE=8 total workers，HIGH=9，hard max=10。
- t0 hire；t1起按 JSON `livestock_chains` 固定走。
- 只浇 D6 另外6个 Melon；D5 新 ST/T/CR 用 JSON crop list。
- **所有 `wday=5` 此时 age1，禁止 DIG / HARVEST / CONVERT。**
- t11 SELL F + harvestable animal products。
- t12 只 BUY D7 feed Wheat；不做 discretionary。
- t13 看真实 cash，只允许填 D5 prebuild；没有 prebuild 就不买动物。
- D6 不为 D7 预建结构，因为 shop2 未知。

## 8. D7：shop2 固定转换日

- BASE=10 total workers，HIGH=11。
- t1–t8：maintenance + 12 Melon WATER + JSON 固定 D7 finance/reserve Wheat。
- t8：SELL F / products / 固定 financing Wheat。
- t9：看真实 cash，只查 `D7|shop1+shop2|LOW/BASE/HIGH`。
- `buy_sw_today=true` 就 t9 BUY_LAND；不能重新判断。
- crop seed 在 t9 买；animal structure 只按 preset 坐标 BUILD；结构完成后固定 market turn BUY_ANIMAL，下一 turn PICKUP/PLACE。
- JSON `wheat_replanted` 的12格必须原格 `PLANT WHEAT -> WATER`，形成 D10 reserve。

## 9. D8 固定延续

- BASE=10，HIGH=11。
- 只延续 shop1+shop2 preset，不重新选行业。
- 填 D7 prebuild；其它永久资产只按 JSON suffix。
- Pet/Farmers 的 D5 Carrot cohort：D8 固定 `WATER -> HARVEST -> PLANT CARROT -> WATER`。
- t10 SELL F/products；t11 只买 D9 feed；t12 看 cash 再执行 preset suffix。

## 10. D9 renewal

- BASE=11，HIGH=12。
- 所有仍为 `wday=5` 的 Wheat 当天 `WATER -> HARVEST`。
- 未转永久资产的同格 `PLANT WHEAT -> WATER`，成为 `wday=9`。
- `wday=7` D10 reserve **绝不动**。
- t10 SELL F/products/renewal surplus；t11 只执行 JSON 新增 suffix。
- 不为 D10 预建 animal，因为 shop3 未知。

## 11. D10 shop3

- BASE=11，HIGH=12。
- 先执行 JSON `wheat_replanted` / reserve jobs，吃 `wday=7`。
- t9 SELL F/products/reserve-W surplus。
- t10 看 cash，只查 `D10|shop1+shop2+shop3|variant`。
- shop3 新增资产、SW、坐标都已经在 JSON 里展开；禁止重排。
- 12 Melon 只 WATER，不 liquidation；D11 再 WATER/HARVEST。
- D10 必须保留 D11 exactly-10-worker hire reserve **$88**。

## 12. worker_jobs 是静态数据，不是 runtime 调度

JSON 对每个 preset 已经展开：
- `livestock_chains`：每链最多4只，固定顺序；
- `melon_groups`：固定 Melon tile list；
- `crop_groups`：固定 ST/T/CR tile list；
- `wheat_replanted` / `wheat_converted`：固定 Wheat tile；
- `prebuild`：固定下一日 structure 坐标。

代码只允许把这些 visit list 用 `_route(start, visits)` 展开成 MOVE。唯一机械 guard 是 `HARVEST_IF_PRESENT`；不能因此改经济组合。

## 13. 现金档位

CSV 的 `cash_min_after_mandatory_finance` 是 mandatory sale/feed 已结算后的门槛。LOW / BASE / HIGH 各自是完整 preset。只允许：

```python
rows = PRESETS[(day, shop_history)]
preset = max((r for r in rows if r.cash_min <= actual_cash), key=r.cash_min)
```

不能拿同样的钱重新组合；连 LOW 都不够就 SURVIVAL。`capacity_clamped=true` 表示 HIGH 已被离线容量检查封顶为 BASE，runtime 不处理。


## 13.1 前一日 LOW/HIGH 的固定继承规则

不要求前一日一定是 BASE，也不重新规划。

- 当前 preset 仍然给出**绝对 target count + 绝对 target coordinates**。
- `investment_ladder` 是固定顺序。运行时只按这个顺序扫描：
  1. 该 tile 已经是要求的 asset：视为已完成；
  2. 该 asset 的实际数量已经达到本 preset target：后续同 asset 项直接跳过；
  3. 否则只允许执行当前这一项；钱不够就从这里停止，不能跳过它买后项。
- 前一日 HIGH 的额外永久资产永不拆。HIGH extra 已经离线做过“下一 reveal 不冲突”筛选：它在所有可能下一 branch 中只会落到同类 asset / Wheat / 同类 prebuild 位置。
- D6、D9 这种下一日马上 reveal 的节点，若找不到无冲突 HIGH tile，HIGH 直接 `capacity_clamped=true` 等于 BASE；运行时没有补救逻辑。
- 因此前一日 LOW 只会让今天沿同一固定 ladder 先补缺口；前一日 HIGH 只会让今天更早满足 `target_count_guard`，不会产生新的策略分支。

CSV 的 `cash_min_after_mandatory_finance` 用于 canonical BASE 审计；真正跨日 carry 时，执行 JSON 的固定 `investment_ladder`，逐项成本已经写死，仍然不存在资产替换或优化。

## 14. 五条典型 BASE 终态

### Smoothie→Smoothie→Brunch
|Day|C|S|G|ST|CR|T|SW|
|---|---:|---:|---:|---:|---:|---:|:--:|
|D5|5|1|1|4|0|0|N|
|D6|6|1|1|4|0|0|N|
|D7|8|1|1|22|0|0|Y|
|D8|9|1|1|26|0|0|Y|
|D9|10|1|1|30|0|0|Y|
|D10|11|1|1|36|0|0|Y|

### IceCream→Smoothie→Yarn
|Day|C|S|G|ST|CR|T|SW|
|---|---:|---:|---:|---:|---:|---:|:--:|
|D5|5|1|1|4|0|0|N|
|D6|6|1|1|4|0|0|N|
|D7|8|1|1|22|0|0|Y|
|D8|9|1|1|26|0|0|Y|
|D9|10|1|1|30|0|0|Y|
|D10|11|3|1|32|0|0|Y|

### Yarn→Yarn→Pet
|Day|C|S|G|ST|CR|T|SW|
|---|---:|---:|---:|---:|---:|---:|:--:|
|D5|2|4|1|0|0|0|N|
|D6|2|5|1|0|0|0|N|
|D7|2|7|1|0|0|0|N|
|D8|2|9|1|0|0|0|N|
|D9|2|11|1|0|0|0|N|
|D10|2|13|1|0|14|0|N|

### Pet→Smoothie→Brunch
|Day|C|S|G|ST|CR|T|SW|
|---|---:|---:|---:|---:|---:|---:|:--:|
|D5|2|1|3|0|14|0|N|
|D6|2|1|3|0|14|0|N|
|D7|4|1|3|12|14|0|Y|
|D8|5|1|3|18|14|0|Y|
|D9|6|1|3|22|14|0|Y|
|D10|7|1|3|32|14|0|Y|

### Pizza→Pizza→Smoothie
|Day|C|S|G|ST|CR|T|SW|
|---|---:|---:|---:|---:|---:|---:|:--:|
|D5|5|1|1|0|0|4|N|
|D6|6|1|1|0|0|4|N|
|D7|8|1|1|0|0|8|N|
|D8|9|1|1|0|0|10|N|
|D9|10|1|1|0|0|12|N|
|D10|12|1|1|12|0|12|Y|

## 15. 当前错误实现直接停用

D5–D10 不得经过 `_make_programme_plan`、`try_group`、`programme_priority`、`solve_intraday`、CP-SAT route、dynamic nearest placement。

```python
if state.day <= 3:
    return scripted_opening_action(obs)
if 4 <= state.day <= 9:
    return scripted_midgame_action(obs)
```

## 16. 验收

- 8种 shop1 从真实 D5 opening 跑到 D7；
- 64种 shop1+shop2 检查到 D9；
- 512种 shop1+shop2+shop3 的 D10 preset 全做坐标/容量/cash 静态校验；
- 重点 full replay：Smoothie/Smoothie/Brunch、IceCream/Smoothie/Yarn、Yarn/Yarn/Pet、Pet/Brunch/IceCream、Pet/Smoothie/Brunch、Pizza/Pizza/Smoothie、Bakery/Brunch/Yarn、Farmers/Smoothie/Pet；
- 必须 0 illegal action、0 negative cash、0 escaped animal、0 新 plant 未当日 WATER；
- preset 失败只能改该 preset 的 literal data，不准恢复 planner/intraday。