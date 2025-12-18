# 快速开始指南

## 5分钟快速启动

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2: 配置数据库

1. 确保 MySQL 服务正在运行

2. 创建数据库和表：
```bash
mysql -u root -p < sql/schema.sql
```

3. 导入初始数据：
```bash
mysql -u root -p crop_pilot_db < sql/seed_data.sql
```

### 步骤 3: 配置环境变量

复制环境变量示例文件：
```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

编辑 `.env` 文件，修改数据库密码：
```env
DB_PASSWORD=your_actual_password
```

### 步骤 4: 启动应用

```bash
python src/app.py
```

### 步骤 5: 访问应用

打开浏览器访问：`http://localhost:5000`

## 常见问题

**Q: 数据库连接失败？**  
A: 检查 `.env` 文件中的数据库配置是否正确，确保 MySQL 服务正在运行。

**Q: 模块导入错误？**  
A: 确保从项目根目录运行 `python src/app.py`，而不是从 `src` 目录内运行。

**Q: 端口被占用？**  
A: 修改 `src/app.py` 最后一行的端口号。

## 下一步

- 查看 [API文档](docs/api.md) 了解所有接口
- 查看 [数据库设计](docs/database.md) 了解数据结构
- 查看 [部署指南](docs/deployment.md) 了解生产环境部署

