# 作物生长状态管理与决策支持系统 (CropPilot)

一个基于Flask的智能农业决策支持系统，围绕“用户-地块-作物”管理，帮助农民根据地块的作物和生长阶段获取农事建议，记录传感器数据与决策历史。

## 系统特色

- 🌾 **多作物智能管理**: 支持水稻、玉米等多种作物的全生长周期管理
- 📊 **数据驱动决策**: 基于传感器数据和历史物候数据的科学决策
- 💡 **混合智能引擎**: 规则引擎与智能检索相结合的双重决策机制
- 📈 **实时数据监控**: 支持多源传感器数据采集、存储和实时分析
- 🔄 **全程可追溯**: 完整的决策历史记录和用户反馈机制
- 📷 **图像识别**: 支持作物图片上传和病虫害智能识别（规划中）
- 📊 **数据可视化**: 丰富的图表展示和趋势分析功能

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
│   ├── smart_knowledge.py       # 智能知识库（ChromaDB + 向量检索）
│   ├── knowledge_loader.py      # 多数据源知识加载器
│   ├── image_recognition.py     # 图像识别模块
│   └── knowledge_base.py        # 知识库（备用数据源/缓存）
│
├── data/                        # 数据文件目录
│   └── agriculture_knowledge.json  # JSON格式农业知识库
│
├── manage_knowledge.py          # 知识库管理工具（交互式CLI）
├── demo_knowledge_management.py # 知识管理演示脚本
├── reset_knowledge_db.py        # 向量数据库重置工具
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

- Python 3.11+ (推荐)
- MySQL 8.0+ 或 MariaDB 10.3+
- Git (版本控制)
- 现代Web浏览器 (Chrome, Firefox, Safari, Edge)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**智能功能依赖说明**：
- `chromadb`: 向量数据库，用于语义搜索
- `sentence-transformers`: 文本向量化模型
- 首次运行时会自动下载约50MB的AI模型文件

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

**智能农事咨询（自然语言查询）：**
```json
POST /api/smart_advice
{
  "question": "水稻叶子发黄怎么办",
  "crop_type": "水稻",
  "growth_stage": "分蘖期"
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

## 技术架构

### 系统架构
- **前端**: HTML5 + CSS3 + JavaScript (ES6+)
- **后端**: Python Flask 3.0 + RESTful API
- **数据库**: MySQL 8.0 (关系数据) + ChromaDB (向量数据)
- **AI组件**: sentence-transformers (文本向量化) + 图像识别
- **知识管理**: JSON文件 + 硬编码兜底的混合方案
- **可视化**: Chart.js + D3.js
- **部署**: Docker + Nginx + Gunicorn

### 数据库设计

详细的数据库设计文档请参考 [docs/database.md](docs/database.md)

**核心数据表**:
- **users**: 用户表（支持farmer/admin/expert多角色）
- **fields**: 农田/地块表（用户-地块一对多关系）
- **knowledge_rules**: 知识规则表（按作物+生长阶段组织）
- **sensor_data**: 传感器数据表（时序数据，关联地块）
- **decision_records**: 决策记录表（支持决策追溯）
- **crop_images**: 作物图片表（支持图像识别）

## 项目文档

本项目提供完整的技术文档：

- 📋 [需求规格说明书](docs/requirements_specification.md) - 系统需求与功能定义
- 🏗️ [系统架构设计](docs/system_architecture.md) - 技术架构与模块设计
- 🎨 [UI界面原型设计](docs/ui_prototype_design.md) - 用户界面设计规范
- 📝 [实施任务列表](docs/implementation_tasks.md) - 开发任务与进度管理
- 🔌 [API接口文档](docs/api.md) - RESTful API详细说明
- 🗄️ [数据库设计文档](docs/database.md) - 数据模型与表结构
- 🚀 [部署指南](docs/deployment.md) - 系统部署与运维

## 开发进度

### ✅ 已完成功能
- [x] 基础系统架构设计
- [x] 数据库设计与初始化
- [x] 用户与地块管理
- [x] 传感器数据采集与存储
- [x] 基础决策引擎
- [x] 知识库系统
- [x] RESTful API接口
- [x] Web用户界面
- [x] 图片上传功能
- [x] 历史记录查询

### ✅ 已完成功能（续）
- [x] 智能知识库检索 (ChromaDB + sentence-transformers)
- [x] JSON文件知识管理系统
- [x] 多数据源知识加载器
- [x] 语义搜索与向量检索
- [x] 知识库管理工具
- [x] 数据可视化图表 (Chart.js + 传感器数据趋势图)
- [x] 异常检测与预警 (智能监测 + 多级预警系统)

### 🚧 开发中功能
- [ ] 病虫害图像识别模块
- [ ] 生长阶段自动判断

### 📋 计划功能
- [ ] 用户反馈与评价系统
- [ ] 数据导出功能
- [ ] 移动端适配
- [ ] 系统性能优化
- [ ] 部署与运维工具

## 快速开始

### 克隆项目
```bash
git clone <repository-url>
cd CropPilot
```

### 创建虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 安装依赖并启动
```bash
pip install -r requirements.txt
# 配置数据库（参考上面的数据库配置步骤）
python src/app.py
```

访问 `http://localhost:5000` 开始使用系统。

## 知识库管理

### JSON文件管理
系统支持通过JSON文件管理农业知识库，提供灵活的知识更新方式：

```bash
# 交互式知识管理
python manage_knowledge.py

# 演示知识管理功能
python demo_knowledge_management.py

# 重置向量数据库（更新JSON后使用）
python reset_knowledge_db.py
```

### 知识库特点
- **多数据源支持**: JSON文件、CSV文件、数据库
- **硬编码兜底**: 确保系统在任何情况下都能运行
- **语义搜索**: 基于ChromaDB和sentence-transformers的智能检索
- **离线运行**: 日常使用完全离线，仅初次安装需要网络

## 系统截图

### 主仪表板
- 实时数据监控面板
- 智能农事建议展示
- 地块状态概览

### 数据管理
- 传感器数据录入
- 作物图片上传
- 历史记录查询
- **数据可视化图表** ⭐
- **异常检测与预警** ⭐

### 数据可视化功能
- **多参数趋势图表**: 温度、湿度、土壤参数的时间序列图表
- **营养元素分析**: NPK营养元素含量变化趋势
- **环境监测图表**: 光照强度、pH值等环境参数
- **统计数据摘要**: 自动计算平均值、最值等统计指标
- **交互式图表**: 基于Chart.js的响应式图表，支持缩放和数据点查看

### 异常检测与预警功能
- **智能阈值监测**: 基于作物类型的多参数阈值检测
- **四级预警系统**: 信息、警告、危险、严重四个预警等级
- **实时异常检测**: 传感器数据录入时自动检测异常
- **趋势异常分析**: 检测数据变化趋势中的异常模式
- **自动处理建议**: 根据异常类型自动生成处理建议
- **自动监测功能**: 支持定时自动检查预警状态

*详细的界面设计请参考 [UI原型设计文档](docs/ui_prototype_design.md)*

## 贡献指南

欢迎参与项目开发！请遵循以下步骤：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。