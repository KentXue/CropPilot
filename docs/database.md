# 数据库设计文档

## 数据库信息

- 数据库名称: `crop_pilot_db`
- 字符集: `utf8mb4`
- 排序规则: `utf8mb4_unicode_ci`

## 表结构（当前版本）

### 1. users（用户表，支持多角色）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| username | VARCHAR(50) | 登录用户名，唯一 |
| password_hash | VARCHAR(255) | 密码哈希 |
| role | ENUM('farmer','admin','expert') | 角色 |
| phone | VARCHAR(20) | 联系电话 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

**索引:** `idx_role` (role)

---

### 2. fields（农田/地块表，一对多挂在用户下）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| user_id | INT | 所属用户ID（外键） |
| name | VARCHAR(100) | 地块名称 |
| location | VARCHAR(255) | 地理位置描述 |
| area | DECIMAL(10,2) | 面积 |
| crop_type | VARCHAR(50) | 当前种植作物类型（简化：一块地一种作物） |
| soil_type | VARCHAR(50) | 土壤类型 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

**索引:** `idx_user` (user_id), `idx_crop_type` (crop_type)  
**外键:** `user_id` → `users(id)` ON DELETE CASCADE

---

### 3. knowledge_rules（知识规则表，按作物+生长阶段）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| crop_type | VARCHAR(50) | 作物类型 |
| growth_stage | VARCHAR(50) | 生长阶段 |
| rule_type | VARCHAR(50) | 规则类型（施肥建议、灌溉建议等） |
| action | TEXT | 建议内容 |
| conditions | JSON | 触发条件（可选） |
| priority | INT | 优先级 |
| is_active | BOOLEAN | 是否启用 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

**索引:** `idx_crop_stage` (crop_type, growth_stage), `idx_active` (is_active)

---

### 4. sensor_data（传感器数据表，关联地块）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| field_id | INT | 关联地块ID（外键） |
| crop_type | VARCHAR(50) | 作物类型（冗余，便于查） |
| growth_stage | VARCHAR(50) | 生长阶段 |
| temperature | DECIMAL(5,2) | 温度 |
| humidity | DECIMAL(5,2) | 湿度 |
| soil_moisture | DECIMAL(5,2) | 土壤湿度 |
| light_intensity | DECIMAL(8,2) | 光照强度 |
| ph_value | DECIMAL(4,2) | 土壤pH值 |
| nitrogen | DECIMAL(6,2) | 氮含量 |
| phosphorus | DECIMAL(6,2) | 磷含量 |
| potassium | DECIMAL(6,2) | 钾含量 |
| location | VARCHAR(100) | 位置/编号（可选冗余） |
| recorded_at | TIMESTAMP | 记录时间 |

**索引:** `idx_field` (field_id), `idx_crop_stage` (crop_type, growth_stage), `idx_recorded_at` (recorded_at)  
**外键:** `field_id` → `fields(id)` ON DELETE CASCADE

---

### 5. decision_records（决策记录表，关联地块+传感器数据）
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | INT | 主键，自增 |
| field_id | INT | 关联地块ID（外键） |
| crop_type | VARCHAR(50) | 作物类型（冗余） |
| growth_stage | VARCHAR(50) | 生长阶段 |
| sensor_data_id | INT | 关联的传感器数据ID（外键，可空） |
| advice | TEXT | 系统给出的建议 |
| user_action | TEXT | 用户采取的实际行动 |
| feedback_score | INT | 用户反馈评分（1-5） |
| feedback_comment | TEXT | 用户反馈意见 |
| created_at | TIMESTAMP | 创建时间 |

**索引:** `idx_field` (field_id), `idx_crop_stage` (crop_type, growth_stage), `idx_created_at` (created_at)  
**外键:**  
- `field_id` → `fields(id)` ON DELETE CASCADE  
- `sensor_data_id` → `sensor_data(id)` ON DELETE SET NULL

---

## 初始化数据

`sql/seed_data.sql` 提供示例：
- 用户：`admin`(admin) / `farmer_zhang`(farmer) 等
- 地块：张三名下的水稻田、玉米田
- 知识规则：水稻、玉米各生长阶段的施肥与灌溉建议

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
users (1) ──< fields (1) ──< sensor_data (1) ──< (0..1) decision_records
```

- 一个用户可拥有多块地
- 每条传感器数据必须关联地块
- 决策记录可选关联某条传感器数据，必关联地块

