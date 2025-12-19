# 作物生长状态管理与决策支持系统 (CropPilot)

一个基于Flask的智能农业决策支持系统，围绕“用户-地块-作物”管理，帮助农民根据地块的作物和生长阶段获取农事建议，记录传感器数据与决策历史。

## 功能特性

- 🌾 **多作物支持**: 支持水稻、玉米等多种作物
- 📊 **生长阶段管理**: 根据不同生长阶段提供针对性建议
- 💡 **智能决策**: 基于知识库和数据库规则提供农事建议
- 📈 **传感器数据支持**: 可记录和查询传感器监测数据
- 🔄 **决策记录**: 记录历史决策和用户反馈

## 项目结构

```
CropPilot/
├── README.md                    # 项目总说明
├── requirements.txt             # Python依赖包清单
├── .gitignore                   # Git忽略文件
├── env.example                  # 环境变量配置示例
│
├── src/                         # 源代码目录
│   ├── app.py                   # Flask应用主文件
│   ├── database.py              # 数据库连接和配置
│   ├── decision_engine.py       # 决策引擎核心逻辑
│   └── knowledge_base.py        # 知识库（备用数据源/缓存）
│
├── sql/                         # SQL脚本目录
│   ├── schema.sql               # 数据库表结构定义（含 users/fields 等）
│   └── seed_data.sql            # 初始数据（用户/地块/知识规则）
│
├── static/                      # Flask静态文件目录
│   ├── css/                     # 样式表（预留）
│   ├── js/                      # JavaScript文件（预留）
│   └── images/                  # 图片资源（预留）
│
├── templates/                   # Flask模板目录
│   └── index.html               # 主页面模板
│
├── config/                      # 配置目录
│   └── settings.py              # 配置文件（支持环境变量）
│
└── docs/                        # 项目文档
    ├── requirements_specification.md  # 需求规格说明书
    ├── system_architecture.md         # 系统架构设计文档
    ├── ui_prototype_design.md          # UI界面原型设计
    ├── implementation_tasks.md         # 实施任务列表
    ├── api.md                         # API接口文档
    ├── database.md                    # 数据库设计文档
    └── deployment.md                  # 部署指南
```

## 安装步骤

### 1. 环境要求

- Python 3.7+
- MySQL 5.7+ 或 MariaDB 10.2+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 数据库配置

1. 创建数据库和表结构：
```bash
mysql -u root -p < sql/schema.sql
```

2. 导入初始数据（包含示例用户、地块、规则）：
```bash
mysql -u root -p crop_pilot_db < sql/seed_data.sql
```

3. 配置环境变量：

复制 `env.example` 为 `.env` 并修改配置：
```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

编辑 `.env` 文件，设置数据库连接信息：
```env
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=crop_pilot_db
```

### 4. 运行应用

从项目根目录运行：
```bash
python src/app.py
```

应用将在 `http://127.0.0.1:5000` 启动。

> **提示**: 详细部署说明请参考 [docs/deployment.md](docs/deployment.md)

## API接口

详细的API文档请参考 [docs/api.md](docs/api.md)

### 快速示例

**获取农事建议（推荐按地块）：**
```
GET /api/get_advice?field_id=1&stage=拔节期
```

**保存传感器数据（必须带 field_id）：**
```json
POST /api/save_sensor_data
{
  "field_id": 1,
  "growth_stage": "分蘖期",
  "temperature": 25.5,
  "humidity": 65.0
}
```

**上传作物图片：**
```
POST /api/upload_crop_image
FormData:
- field_id: 1
- image: <选择图片文件>
- captured_at: 2024-05-01T08:30 (可选，拍摄时间)
```

**获取地块 / 用户（演示用）：**
```
GET /api/users
GET /api/fields?user_id=2
```

## 数据库设计

详细的数据库设计文档请参考 [docs/database.md](docs/database.md)

### 主要数据表

- **users**: 用户表（角色 farmer/admin/expert）
- **fields**: 农田/地块表（挂在用户下，一块地一种作物）
- **knowledge_rules**: 知识规则表，按作物+生长阶段
- **sensor_data**: 传感器数据表（关联地块，冗余作物/阶段便于查询）
- **decision_records**: 决策记录表（关联地块，可选关联一条传感器数据）

## 开发计划

- [x] 基础决策引擎
- [x] 知识库系统
- [x] 数据库支持
- [ ] 传感器数据输入界面
- [ ] 数据可视化图表
- [ ] 历史记录查询
- [ ] 用户反馈系统
- [ ] 数据导出功能

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

