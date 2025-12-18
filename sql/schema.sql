-- 作物生长状态管理与决策支持系统数据库表结构
-- 设置数据库默认编码
SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS `crop_pilot_db` 
CHARACTER SET = `utf8mb4`
COLLATE = `utf8mb4_unicode_ci`;

USE `crop_pilot_db`;

-- =========================
-- 用户表（支持多角色）
-- =========================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '登录用户名',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希',
    role ENUM('farmer', 'admin', 'expert') DEFAULT 'farmer' COMMENT '用户角色',
    phone VARCHAR(20) DEFAULT NULL COMMENT '联系电话',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_role (role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='系统用户表';

-- =========================
-- 农田/地块表（1个用户管理多块地）
-- =========================
CREATE TABLE IF NOT EXISTS fields (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL COMMENT '所属用户ID',
    name VARCHAR(100) NOT NULL COMMENT '地块名称',
    location VARCHAR(255) DEFAULT NULL COMMENT '地理位置描述',
    area DECIMAL(10,2) DEFAULT NULL COMMENT '面积，单位亩/公顷',
    crop_type VARCHAR(50) NOT NULL COMMENT '当前种植作物类型（简化为一块地一种作物）',
    soil_type VARCHAR(50) DEFAULT NULL COMMENT '土壤类型',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_crop_type (crop_type),
    CONSTRAINT fk_fields_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='农田/地块表';

-- =========================
-- 知识规则表（按作物+生长阶段）
-- =========================
CREATE TABLE IF NOT EXISTS knowledge_rules (
    id INT AUTO_INCREMENT PRIMARY KEY,
    crop_type VARCHAR(50) NOT NULL COMMENT '作物类型',
    growth_stage VARCHAR(50) NOT NULL COMMENT '生长阶段',
    rule_type VARCHAR(50) NOT NULL COMMENT '规则类型（施肥建议、灌溉建议等）',
    action TEXT NOT NULL COMMENT '具体建议内容',
    conditions JSON COMMENT '触发条件（JSON格式，如温度、湿度等阈值）',
    priority INT DEFAULT 0 COMMENT '优先级，数字越大优先级越高',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否启用',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_crop_stage (crop_type, growth_stage),
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识规则表';

-- =========================
-- 传感器数据表（改为关联具体地块）
-- =========================
CREATE TABLE IF NOT EXISTS sensor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    field_id INT NOT NULL COMMENT '关联的农田/地块ID',
    crop_type VARCHAR(50) NOT NULL COMMENT '作物类型（冗余字段，便于查询）',
    growth_stage VARCHAR(50) NOT NULL COMMENT '生长阶段',
    temperature DECIMAL(5,2) COMMENT '温度（摄氏度）',
    humidity DECIMAL(5,2) COMMENT '湿度（%）',
    soil_moisture DECIMAL(5,2) COMMENT '土壤湿度（%）',
    light_intensity DECIMAL(8,2) COMMENT '光照强度（lux）',
    ph_value DECIMAL(4,2) COMMENT '土壤pH值',
    nitrogen DECIMAL(6,2) COMMENT '氮含量（mg/kg）',
    phosphorus DECIMAL(6,2) COMMENT '磷含量（mg/kg）',
    potassium DECIMAL(6,2) COMMENT '钾含量（mg/kg）',
    location VARCHAR(100) COMMENT '位置/地块编号（可选冗余）',
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
    INDEX idx_field (field_id),
    INDEX idx_crop_stage (crop_type, growth_stage),
    INDEX idx_recorded_at (recorded_at),
    CONSTRAINT fk_sensor_field FOREIGN KEY (field_id) REFERENCES fields(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='传感器数据表';

-- =========================
-- 决策记录表（关联地块+传感器数据）
-- =========================
CREATE TABLE IF NOT EXISTS decision_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    field_id INT NOT NULL COMMENT '关联的农田/地块ID',
    crop_type VARCHAR(50) NOT NULL COMMENT '作物类型（冗余字段，便于查询）',
    growth_stage VARCHAR(50) NOT NULL COMMENT '生长阶段',
    sensor_data_id INT COMMENT '关联的传感器数据ID',
    advice TEXT NOT NULL COMMENT '系统给出的建议',
    user_action TEXT COMMENT '用户采取的实际行动',
    feedback_score INT COMMENT '用户反馈评分（1-5）',
    feedback_comment TEXT COMMENT '用户反馈意见',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_field (field_id),
    INDEX idx_crop_stage (crop_type, growth_stage),
    INDEX idx_created_at (created_at),
    CONSTRAINT fk_decision_field FOREIGN KEY (field_id) REFERENCES fields(id) ON DELETE CASCADE,
    CONSTRAINT fk_decision_sensor FOREIGN KEY (sensor_data_id) REFERENCES sensor_data(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='决策记录表';

-- =========================
-- 作物图片表（存储地块图片）
-- =========================
CREATE TABLE IF NOT EXISTS crop_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    field_id INT NOT NULL COMMENT '关联的农田/地块ID',
    image_path VARCHAR(255) NOT NULL COMMENT '图片存储路径（相对路径）',
    captured_at DATETIME NULL COMMENT '拍摄时间',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    INDEX idx_field (field_id),
    INDEX idx_uploaded_at (uploaded_at),
    CONSTRAINT fk_crop_images_field FOREIGN KEY (field_id) REFERENCES fields(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='作物图片表';

SET FOREIGN_KEY_CHECKS = 1;

