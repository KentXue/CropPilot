# API 接口文档

## 基础信息

- 基础URL: `http://localhost:5000`
- 所有API返回JSON格式

## 接口列表

### 1. 获取农事建议

**接口地址:** `/api/get_advice`

**请求方式:** `GET`

**请求参数:**
- `field_id` (推荐，可选): 地块ID；如果提供，将优先根据地块确定作物
- `crop` (可选): 作物类型，默认值: `水稻`（当未传 `field_id` 时使用）
- `stage` (可选): 生长阶段，默认值: `分蘖期`

**示例请求:**
```
GET /api/get_advice?field_id=1&stage=拔节期
```

**响应示例:**
```json
{
  "crop": "水稻",
  "stage": "拔节期",
  "advice": [
    "施肥建议: 注意控制氮肥，防止徒长，可适当增施磷钾肥。",
    "灌溉建议: 保持适度水层，避免过深或过浅。"
  ],
  "status": "success"
}
```

---

### 2. 保存传感器数据

**接口地址:** `/api/save_sensor_data`

**请求方式:** `POST`

**请求体 (JSON):**
```json
{
  "field_id": 1,          // 必填，所属地块
  "crop_type": "水稻",    // 可选，缺省时后端会根据 field_id 自动补足
  "growth_stage": "分蘖期",
  "temperature": 25.5,
  "humidity": 65.0,
  "soil_moisture": 70.0,
  "light_intensity": 50000,
  "ph_value": 6.5,
  "nitrogen": 150.0,
  "phosphorus": 80.0,
  "potassium": 120.0,
  "location": "A区-1号田"
}
```

**响应示例:**
```json
{
  "status": "success",
  "data_id": 1,
  "message": "传感器数据保存成功"
}
```

---

### 3. 查询传感器数据

**接口地址:** `/api/get_sensor_data`

**请求方式:** `GET`

**请求参数:**
- `field_id` (可选): 地块ID，用于筛选
- `crop` (可选): 作物类型，用于筛选
- `stage` (可选): 生长阶段，用于筛选
- `limit` (可选): 返回记录数，默认值: `50`

**示例请求:**
```
GET /api/get_sensor_data?field_id=1&limit=10
```

**响应示例:**
```json
{
  "status": "success",
  "count": 10,
  "data": [
    {
      "id": 1,
      "crop_type": "水稻",
      "growth_stage": "分蘖期",
      "temperature": 25.5,
      "humidity": 65.0,
      "recorded_at": "2024-01-01 10:00:00"
    }
  ]
}
```

---

### 4. 查询决策历史记录

**接口地址:** `/api/get_history`

**请求方式:** `GET`

**请求参数:**
- `field_id` (可选): 地块ID，用于筛选
- `crop` (可选): 作物类型，用于筛选
- `stage` (可选): 生长阶段，用于筛选
- `limit` (可选): 返回记录数，默认值: `50`

**示例请求:**
```
GET /api/get_history?field_id=1&stage=分蘖期
```

**响应示例:**
```json
{
  "status": "success",
  "count": 5,
  "records": [
    {
      "id": 1,
      "crop_type": "水稻",
      "growth_stage": "分蘖期",
      "advice": "施肥建议: 每亩追施尿素5-8公斤，促进分蘖。\n灌溉建议: 保持浅水层3-5cm，促进分蘖。",
      "created_at": "2024-01-01 10:00:00"
    }
  ]
}
```

---

### 5. 获取地块列表

**接口地址:** `/api/fields`

**请求方式:** `GET`

**请求参数:**
- `user_id` (可选): 用户ID，用于筛选该用户的地块

**示例请求:**
```
GET /api/fields
```

**响应示例:**
```json
{
  "status": "success",
  "count": 2,
  "fields": [
    {
      "id": 1,
      "user_id": 2,
      "name": "张三水稻田1号",
      "location": "湖北省某县A区",
      "area": "12.50",
      "crop_type": "水稻",
      "soil_type": "壤土",
      "created_at": "2024-01-01 10:00:00"
    }
  ]
}
```

---

### 6. 获取用户列表（演示用）

**接口地址:** `/api/users`

**请求方式:** `GET`

**请求参数:** 无

**响应示例:**
```json
{
  "status": "success",
  "count": 2,
  "users": [
    {
      "id": 1,
      "username": "admin",
      "role": "admin",
      "phone": "13800000000",
      "created_at": "2024-01-01 10:00:00"
    }
  ]
}
```

---

## 错误响应

所有接口在出错时返回以下格式:

```json
{
  "status": "error",
  "message": "错误描述信息"
}
```

HTTP状态码: `500`

