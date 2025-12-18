# 数据库设计文档

## 数据库信息

- 数据库名称: `crop_pilot_db`
- 字符集: `utf8mb4`
- 排序规则: `utf8mb4_unicode_ci`

## 表结构

### 1. knowledge_rules (知识规则表)

存储作物生长各阶段的农事建议规则。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| crop_type | VARCHAR(50) | 作物类型 |
| growth_stage | VARCHAR(50) | 生长阶段 |
| rule_type | VARCHAR(50) | 规则类型（施肥建议、灌溉建议等） |
| action | TEXT | 具体建议内容 |
| conditions | JSON | 触发条件（JSON格式，如温度、湿度等阈值） |
| priority | INT | 优先级，数字越大优先级越高 |
| is_active | BOOLEAN | 是否启用 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

**索引:**
- `idx_crop_stage`: (crop_type, growth_stage)
- `idx_active`: (is_active)

---

### 2. sensor_data (传感器数据表)

存储传感器采集的实时数据。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| crop_type | VARCHAR(50) | 作物类型 |
| growth_stage | VARCHAR(50) | 生长阶段 |
| temperature | DECIMAL(5,2) | 温度（摄氏度） |
| humidity | DECIMAL(5,2) | 湿度（%） |
| soil_moisture | DECIMAL(5,2) | 土壤湿度（%） |
| light_intensity | DECIMAL(8,2) | 光照强度（lux） |
| ph_value | DECIMAL(4,2) | 土壤pH值 |
| nitrogen | DECIMAL(6,2) | 氮含量（mg/kg） |
| phosphorus | DECIMAL(6,2) | 磷含量（mg/kg） |
| potassium | DECIMAL(6,2) | 钾含量（mg/kg） |
| location | VARCHAR(100) | 位置/地块编号 |
| recorded_at | TIMESTAMP | 记录时间 |

**索引:**
- `idx_crop_stage`: (crop_type, growth_stage)
- `idx_recorded_at`: (recorded_at)

---

### 3. decision_records (决策记录表)

存储系统生成的决策建议记录。

| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| crop_type | VARCHAR(50) | 作物类型 |
| growth_stage | VARCHAR(50) | 生长阶段 |
| sensor_data_id | INT | 关联的传感器数据ID（外键） |
| advice | TEXT | 系统给出的建议 |
| user_action | TEXT | 用户采取的实际行动 |
| feedback_score | INT | 用户反馈评分（1-5） |
| feedback_comment | TEXT | 用户反馈意见 |
| created_at | TIMESTAMP | 创建时间 |

**索引:**
- `idx_crop_stage`: (crop_type, growth_stage)
- `idx_created_at`: (created_at)

**外键:**
- `sensor_data_id` → `sensor_data(id)` ON DELETE SET NULL

---

## 初始化数据

初始数据存储在 `sql/seed_data.sql` 文件中，包括：

- 水稻各生长阶段的施肥和灌溉建议
- 玉米各生长阶段的施肥和灌溉建议

执行方式：
```bash
mysql -u root -p crop_pilot_db < sql/seed_data.sql
```

---

## 数据库初始化

### 1. 创建数据库和表结构

```bash
mysql -u root -p < sql/schema.sql
```

### 2. 导入初始数据

```bash
mysql -u root -p crop_pilot_db < sql/seed_data.sql
```

---

## 数据关系图

```
sensor_data (1) ──< (0..1) decision_records
```

- 一个传感器数据记录可以对应多个决策记录（可选）
- 决策记录可以独立存在，不依赖传感器数据

