# 🌾 CropPilot - 智能农业决策支持系统

> 基于AI技术的现代化农业管理平台，提供智能决策、数据分析和作物健康监测服务

## 📋 项目概述

CropPilot是一个完整的智能农业决策支持系统，集成了数据管理、AI分析、图像识别和智能咨询等功能，帮助农民做出科学的农业决策。

### 🎯 核心功能

- **👥 用户管理**: 多角色用户系统（农民、管理员、专家）
- **🏞️ 地块管理**: 多地块管理和作物类型配置
- **📊 数据管理**: 传感器数据录入、存储和查询
- **📈 数据可视化**: 实时图表展示和趋势分析
- **🤖 智能咨询**: 基于自然语言的农业问答系统
- **🖼️ AI图像识别**: 植物病害自动识别和诊断
- **🚨 异常预警**: 多参数监测和四级预警系统
- **💡 农事建议**: 基于数据的智能决策建议

## 🚀 快速开始

### 环境要求

- Python 3.8+
- MySQL 8.0+
- CUDA支持的GPU（可选，用于AI加速）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd CropPilot
```

2. **创建虚拟环境**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置数据库**
```bash
# 创建数据库
mysql -u root -p
CREATE DATABASE crop_pilot;

# 导入数据结构和示例数据
mysql -u root -p crop_pilot < sql/schema.sql
mysql -u root -p crop_pilot < sql/seed_data.sql
```

5. **配置环境变量**
```bash
# 复制环境配置文件
cp env.example .env
# 编辑.env文件，配置数据库连接信息
```

6. **启动应用**
```bash
python src/app.py
```

7. **访问系统**
```
http://localhost:5000
```

## 📖 使用指南

### 基本操作流程

1. **选择用户**: 在页面顶部选择用户（如：farmer_zhang）
2. **选择地块**: 在各功能模块中选择对应地块
3. **录入数据**: 在"传感器数据"标签页录入环境数据
4. **查看分析**: 在"数据可视化"标签页查看数据趋势
5. **获取建议**: 在"农事建议"标签页获取专业建议
6. **智能咨询**: 在"智能咨询"标签页进行自然语言问答
7. **监控预警**: 在"异常预警"标签页查看预警信息

### 功能模块说明

#### 🤖 智能咨询
- 支持自然语言问答
- 基于农业知识库的语义搜索
- 提供专业的农事建议

#### 🖼️ AI图像识别
- 支持38种植物病害识别
- 自动生成治疗建议
- GPU加速处理

#### 📊 数据可视化
- 实时环境参数趋势图
- 营养元素分析图表
- 交互式数据展示

#### 🚨 异常预警
- 四级预警系统（信息/警告/危险/严重）
- 多参数阈值监测
- 自动处理建议

## 🏗️ 技术架构

### 后端技术栈
- **Web框架**: Flask 3.0
- **数据库**: MySQL 8.0
- **向量数据库**: ChromaDB
- **AI框架**: PyTorch + torchvision
- **文本处理**: sentence-transformers
- **图像处理**: PIL + OpenCV

### 前端技术栈
- **基础**: HTML5 + CSS3 + JavaScript ES6+
- **图表库**: Chart.js
- **样式**: 响应式CSS Grid + Flexbox

### AI技术
- **深度学习模型**: ResNet18
- **文本嵌入**: all-MiniLM-L6-v2
- **向量搜索**: ChromaDB
- **GPU加速**: CUDA支持

## 📁 项目结构

```
CropPilot/
├── src/                    # 源代码目录
│   ├── app.py             # Flask主应用
│   ├── database.py        # 数据库连接
│   ├── image_recognition.py # AI图像识别
│   ├── smart_knowledge.py # 智能知识库
│   ├── decision_engine.py # 决策引擎
│   └── ...
├── templates/             # HTML模板
│   └── index.html        # 主页面
├── static/               # 静态资源
├── sql/                  # 数据库脚本
│   ├── schema.sql        # 数据库结构
│   └── seed_data.sql     # 示例数据
├── docs/                 # 项目文档
├── config/               # 配置文件
├── data/                 # 数据文件
└── requirements.txt      # Python依赖
```

## 🔧 配置说明

### 环境变量配置 (.env)
```bash
# 数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=crop_pilot

# Flask配置
FLASK_ENV=development
SECRET_KEY=your_secret_key

# 上传配置
UPLOAD_FOLDER=static/uploads
```

### 数据库配置
- 确保MySQL服务正在运行
- 创建数据库和用户权限
- 导入数据结构和示例数据

## 📊 性能指标

- **响应时间**: < 2秒（一般查询）
- **AI识别**: < 5秒（单张图片）
- **智能咨询**: < 10秒（复杂查询）
- **并发支持**: 100+ 用户
- **数据容量**: 支持大规模数据存储

## 🔒 安全特性

- 输入数据验证和清洗
- SQL注入防护
- 文件上传安全检查
- 错误信息安全处理
- 环境变量配置保护

## 🚀 部署建议

### 生产环境部署
1. 使用Gunicorn或uWSGI作为WSGI服务器
2. 配置Nginx作为反向代理
3. 使用Redis进行缓存优化
4. 配置SSL证书启用HTTPS
5. 设置定期数据备份

### 性能优化
- 数据库索引优化
- 静态资源CDN加速
- 图片压缩和缓存
- API响应缓存

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

### 开发规范
- 遵循PEP 8代码规范
- 添加适当的注释和文档
- 编写单元测试
- 提交前进行代码检查

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**CropPilot v1.0** - 让农业更智能 🌾