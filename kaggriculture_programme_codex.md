# Kaggriculture 中后期 Programme：Codex 实现规格

> 目标：在现有 fixed opening、`Plan -> intraday -> execution` 架构上直接实现可运行策略。不要重做 opening，不要重写 intraday。**D4 已经固定完成，Programme 从 D5 开始接管。** D5–D10 是主扩张期；所有数量是 **pace floor，不是上限**。达到后如果 cash/feed/route 仍允许，继续扩。

## 1. 每日基本约束

下面的 worker 数是**当天允许的最大总 worker 数（含 farmer）**，不是默认必须雇满。intraday 应在这个上限内寻找最少可行人数；当前人数无法按 deadline 完成 programme 时才继续加 worker。

| Day | D5 | D6 | D7 | D8 | D9 | D10 | D11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| max total workers | 10 | 10 | 11 | 11 | 12 | 12 | **10** |

D11 是特殊硬约束：**exactly 10 total workers**。

普通日 mandatory <= t20；新增动物 PLACE <= t19；新 crop `PLANT+WATER` <= t20；D11 melon critical <= t19。若 route 过慢，先在当天 worker 上限内增加 worker；达到上限仍失败，再删 optional CARE、边际 harvest、reservation-W 转换、最后一个边际资产。不得删 mandatory FEED/WATER。

## 2. 土地角色与 Wheat

每个 unlocked tile 只能属于：

- `PERMANENT`：Melon / ST / active animal / strategic CR
- `NEXT_DAY_STRUCTURE`：**明天确定买得起**的动物对应 pasture/coop
- `OPERATING_WHEAT`：feed / cash backbone
- `RESERVATION_WHEAT`：未来 1–3 天可能转用途的占位 Wheat

禁止大面积裸地。

### 次日预建

每天估算：

```text
cash_D+1 =
cash_now
+ guaranteed F/product/W/CR sales
- next-day hires
- mandatory feed
- mandatory seeds
- other mandatory costs
```

对候选动物计算 `finance_cap / feed_cap / route_cap / land_cap`，只预建：

```text
prebuild = min(finance_cap, feed_cap, route_cap, land_cap)
```

长期想要 5 Cow，但明天只买得起 2，只建 2 个 pasture；其余未来 Cow 位先种 reservation Wheat。

### Wheat

```text
OpW >= max(8, total_livestock + 2)
```

若有 Wheat shop（Bakery/Pizza/Brunch/IceCream/Farmers），再提高；实际目标可用：

```text
OpW ≈ total_livestock + 2 + 4 * wheat_shop_count
```

释放 OpW 前必须保证：

```text
shed_W + guaranteed harvest >= next_2_days_feed
```

Reservation Wheat 带 `candidate_use + preferred_release_day`。资产 financeable 时同日转换：

```text
W -> ST/CR: HARVEST -> PLANT -> WATER
W -> animal: HARVEST -> BUILD -> BUY -> PICKUP -> PLACE
```

若今天还买不起，就让 Wheat 多长一天。

## 3. Shop counters

```text
milk = Pizza + IceCream + Smoothie
yarn = Yarn
egg  = Bakery + Brunch
st   = IceCream + Smoothie + Brunch + Farmers
wheat_shop = Bakery + Pizza + Brunch + IceCream + Farmers
carrot_units = 2*Pet + Farmers
```

## 4. Goose bridge

若 `milk == 0 and yarn == 0`，不要强行买 Cow/Sheep；用较便宜 Goose 维持 F 资金链：

```text
egg == 0 -> G target 3
egg == 1 -> G target 4
egg >= 2 -> G target 6
```

出现 Milk/Wool 信号后立刻停止新增 bridge Goose；已有 Goose 保留并继续产 F。

## 5. D4 已固定，Programme 从 D5 开始

D4 已经由 opening 完整确定，**这里不做任何 D4 重规划**。

实现时：

- 不修改 D3/D4 的既有 Cow/Sheep/Goose/ST/Wheat 布局；
- 不为了匹配某个理想 D4 target 改 opening；
- D4 结束状态只作为 D5 的真实输入；
- shop1 已经知道，但它只负责选择 D5 起的 programme；
- Goose bridge、aggressive ST、next-day pasture/coop、reservation Wheat 等逻辑全部从 D5 开始执行。

因此所有 programme 的第一个可控目标都从 **D5** 写起。

## 6. D5–D10 动物规模

Milk-only 的 D10 pace floor：

```text
milk=1 -> C7
milk=2 -> C10
milk>=3 -> C12
```

Yarn-only：

```text
yarn=1 -> S8
yarn=2 -> S13
yarn>=3 -> S15
```

Mixed：

```text
milk=1,yarn=1 -> C6 S5
milk=2,yarn=1 -> C9 S5
milk=1,yarn=2 -> C6 S8
milk>=2,yarn>=2 -> C9 S8
```

D5 对所有已有 Milk/Wool signal 的线：

```text
C + S >= 6
```

只是最低线；能买第 7/8 只就继续。

日新增 route cap：

```text
D4 +4
D5 +3
D6 +2
D7 +2
D8 +2
D9 +3
D10 +3
```

实际新增量 = `min(finance, feed, route, land)`。

## 7. Strawberry

不要照 top 的 D6/D7 ST 数；它们第二块地晚开，我们 opening 已有两块地。

一个 ST shop：

```text
D4 6
D5 12
D6 16
D7 20
D8 22
D9 24
D10 24
```

两个 ST shop：

```text
D6 18
D7 24
D8 28
D9 30
D10 32
```

三个及以上：

```text
D9 34
D10 36
```

IceCream 可比表少 2 ST 以保 OpW；Farmers 可少 4 ST 以容纳 CR。ST 以 4 格为扩张 block。D5–D10 不允许以 `TARGET_REACHED` 作为停止理由。

## 8. Carrot

```text
Pet x1 -> CR14
Pet x2 -> CR24
Farmers only -> CR6
Pet + Farmers -> CR18
其他 -> opening 收完后 CR0
```

以 4 格扩张。

## 9. 四类 Programme

### DAIRY_GROWTH
适用 Pizza/Smoothie/IceCream，以及后续 Milk signal。

优先级：

```text
mandatory feed
-> Cow
-> ST（若支持）
-> OpW
-> ResW
```

### WOOL_GROWTH
适用 Yarn：

```text
Sheep -> OpW -> 后续新增需求产业 -> ResW
```

### EGG_ST_GROWTH
Brunch：

```text
ST -> Goose -> OpW -> ResW
```

Bakery：

```text
Goose -> OpW -> ResW
```

### GOOSE_BRIDGE_MIXED
Pet/Farmers 或暂时无 Milk/Wool：

```text
supported crop
-> bridge Goose
-> OpW
-> ResW
```

shop2 出 Milk/Wool 后停止 Goose expansion，并释放 reservation Wheat 转 C/S。

## 10. 典型日表

### Smoothie

| Day | C | G | ST | OpW |
|---|---:|---:|---:|---:|
| D4 | 3 | 1 | 6 | 8 |
| D5 | 6 | 1 | 12 | 9 |
| D6 | 7 | 1 | 16 | 10 |
| D7 | 8 | 1 | 20 | 11 |
| D8 | 9 | 1 | 22 | 12 |
| D9 | 10 | 1 | 24 | 13 |

若 shop2 再给 Milk+ST：D6–D10 ST 改为 `18,24,28,30,32`。

### IceCream

`D4–D9: C=3,6,7,8,9,10；ST=4,10,14,18,22,24；OpW=12,12,13,14,15,16`

### Pizza

`D4–D8: C=4,7,8,9,10；OpW=14,15,16,17,18`

### Yarn

`D4–D8: S=3,6,7,8,8；OpW=12,12,13,14,14`

若第二个 Yarn：`D7 S9, D8 S11, D9 S13`

### Brunch

`D4–D7: G=4,4,4,4；ST=8,14,18,22；OpW=10,10,11,12`

### Bakery

`D4–D6: G4, OpW=18,20,20`；第二个 Egg shop 后可推 G6。

## 11. 第三地

成本 $2000。D7 开始检查，正常购买窗口 D7–D9。

必须同时满足：

1. 两块地连续扩张空间不足；
2. 至少 8 格近期明确永久需求；
3. 买地后仍能支付当日 worker、次日 feed、已计划动物和 seed；
4. 新地当天能落地的永久资产立即落地，其余全部 reservation Wheat。

Wheat filler 解决裸地问题，但不能忽略 $2000 的资本机会成本。

## 12. Fertilizer

Melon 永不占 F。

默认：

```text
COLLECT_F -> shed -> SELL
```

只有明确 fertilize 边际收益高于当前 F sale value 才保留。

D4–D10 F 主要作为发展资本。

## 13. D11

**exactly 10 total workers**。

进入 D11 前必须已接近成熟，不允许把 D11 当发展起点。

顺序：

```text
animal FEED/F
-> mandatory crop WATER
-> 12 Melon liquidation
-> 用 Melon cash 完成一个预先选定 expansion block
```

Melon critical <= t19。

## 14. 后期

D12–D20：维持成熟产业，有限继续扩。

D22 起默认停止新 Cow/Sheep；新 ST 必须还能完成至少 2 次 harvest event，否则不种。

D24–D29：停止长期投资，收现有产品/F，ST 退出后转 Wheat/Carrot 等短周期，最后清仓。

## 15. 每日 Expansion Loop

形成 mandatory Plan 后持续尝试加资产：

```text
while candidate exists:
    candidate = programme_priority()
    if cash/feed/land/horizon/route/water/next-day-operation all pass:
        add candidate
        update simulated ledger
    else:
        try cheaper/next candidate

remaining undecided tiles -> reservation Wheat
```

D4–D10 diagnostics 必须记录未继续扩张的原因，只允许：

```text
CASH
FEED
LAND
ROUTE
WATER_DEADLINE
HORIZON
```

不允许：

```text
TARGET_REACHED
```

## 16. 实现边界

Programme / planner 决定：

- 今天要达到什么结构
- 哪些 tile 是 ST/CR/animal/W
- 哪些 reservation Wheat 今天释放
- 买什么、卖什么、买不买地

intraday 只决定：

- 哪个 worker 执行
- 移动顺序
- pickup/drop 顺序

必须复用现有 `rules.py` 的官方常数/市场曲线和现有 `Plan -> intraday -> execution` 链路，不新建平行 planner。

如果某个数值目标经 exact simulation 证明不可行，不要静默缩小；输出：

```text
day
programme
requested target
max feasible target
binding constraint
```

然后只调整被证明失败的数字。
