# 部署指南

## 环境要求

- Python 3.8+
- MySQL 5.7+ 或 MariaDB 10.3+
- pip (Python包管理器)

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd CropPilot
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置数据库

#### 4.1 创建数据库

```bash
mysql -u root -p < sql/schema.sql
```

#### 4.2 导入初始数据

```bash
mysql -u root -p crop_pilot_db < sql/seed_data.sql
```

### 5. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 数据库配置
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=crop_pilot_db

# Flask配置
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# 外部API配置（可选）
SENSECAP_API_KEY=your_api_key
SENSECAP_API_URL=https://api.sensecap.com
```

**重要:** `.env` 文件已添加到 `.gitignore`，不会被提交到版本控制。

### 6. 运行应用

```bash
# 从项目根目录运行
python src/app.py
```

或者：

```bash
cd src
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 生产环境部署

### 使用 Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app
```

### 使用 Waitress (Windows)

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 src.app:app
```

### 使用 Nginx 反向代理

在 Nginx 配置文件中添加：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 常见问题

### 1. 数据库连接失败

- 检查 MySQL 服务是否运行
- 确认 `.env` 文件中的数据库配置正确
- 检查数据库用户权限

### 2. 模块导入错误

确保从项目根目录运行应用，或设置正确的 Python 路径。

### 3. 端口被占用

修改 `src/app.py` 中的端口号：

```python
app.run(debug=app.config['DEBUG'], port=5001, host='0.0.0.0')
```

## 开发环境 vs 生产环境

### 开发环境

- `FLASK_DEBUG=True`
- 使用 Flask 内置服务器
- 详细错误信息

### 生产环境

- `FLASK_DEBUG=False`
- 使用 Gunicorn/Waitress
- 配置 Nginx 反向代理
- 使用 HTTPS
- 设置强密码和密钥

