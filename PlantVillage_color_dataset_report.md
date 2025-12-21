# PlantVillage COLOR 数据集分析报告

## 基本信息
- **数据集类型**: color
- **数据集路径**: C:\Users\hp\Desktop\作物生长状态管理与决策支持系统\数据\1.图像数据（病虫害识别核心）\plantvillage dataset\color
- **总类别数**: 38
- **总图像数**: 54305
- **输入尺寸**: (224, 224)

## 类别映射统计
- **总类别数**: 38
- **作物类别分布**: {'果树类': 17, '粮食作物': 8, '蔬菜类': 13}
- **主要作物**: 苹果, 蓝莓, 樱桃, 玉米, 葡萄, 橙子, 桃子, 甜椒, 马铃薯, 覆盆子

## 类别详细信息

| 英文名称 | 中文名称 | 作物 | 病害 | 图像数量 |
|----------|----------|------|------|----------|
| Orange___Haunglongbing_(Citrus_greening) | 橙子_黄龙病 | 橙子 | 黄龙病 | 5507 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 番茄_黄化曲叶病毒 | 番茄 | 黄化曲叶病毒 | 5357 |
| Soybean___healthy | 大豆_健康 | 大豆 | 健康 | 5090 |
| Peach___Bacterial_spot | 桃子_细菌性斑点病 | 桃子 | 细菌性斑点病 | 2297 |
| Tomato___Bacterial_spot | 番茄_细菌性斑点病 | 番茄 | 细菌性斑点病 | 2127 |
| Tomato___Late_blight | 番茄_晚疫病 | 番茄 | 晚疫病 | 1909 |
| Squash___Powdery_mildew | 南瓜_白粉病 | 南瓜 | 白粉病 | 1835 |
| Tomato___Septoria_leaf_spot | 番茄_斑点病 | 番茄 | 斑点病 | 1771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 番茄_红蜘蛛 | 番茄 | 红蜘蛛 | 1676 |
| Apple___healthy | 苹果_健康 | 苹果 | 健康 | 1645 |
| Tomato___healthy | 番茄_健康 | 番茄 | 健康 | 1591 |
| Blueberry___healthy | 蓝莓_健康 | 蓝莓 | 健康 | 1502 |
| Pepper,_bell___healthy | 甜椒_健康 | 甜椒 | 健康 | 1478 |
| Tomato___Target_Spot | 番茄_靶斑病 | 番茄 | 靶斑病 | 1404 |
| Grape___Esca_(Black_Measles) | 葡萄_黑麻疹病 | 葡萄 | 黑麻疹病 | 1383 |
| Corn_(maize)___Common_rust_ | 玉米_普通锈病 | 玉米 | 普通锈病 | 1192 |
| Grape___Black_rot | 葡萄_黑腐病 | 葡萄 | 黑腐病 | 1180 |
| Corn_(maize)___healthy | 玉米_健康 | 玉米 | 健康 | 1162 |
| Strawberry___Leaf_scorch | 草莓_叶焦病 | 草莓 | 叶焦病 | 1109 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 葡萄_叶枯病 | 葡萄 | 叶枯病 | 1076 |
| Cherry_(including_sour)___Powdery_mildew | 樱桃_白粉病 | 樱桃 | 白粉病 | 1052 |
| Potato___Early_blight | 马铃薯_早疫病 | 马铃薯 | 早疫病 | 1000 |
| Potato___Late_blight | 马铃薯_晚疫病 | 马铃薯 | 晚疫病 | 1000 |
| Tomato___Early_blight | 番茄_早疫病 | 番茄 | 早疫病 | 1000 |
| Pepper,_bell___Bacterial_spot | 甜椒_细菌性斑点病 | 甜椒 | 细菌性斑点病 | 997 |
| Corn_(maize)___Northern_Leaf_Blight | 玉米_北方叶枯病 | 玉米 | 北方叶枯病 | 985 |
| Tomato___Leaf_Mold | 番茄_叶霉病 | 番茄 | 叶霉病 | 952 |
| Cherry_(including_sour)___healthy | 樱桃_健康 | 樱桃 | 健康 | 854 |
| Apple___Apple_scab | 苹果_苹果黑星病 | 苹果 | 苹果黑星病 | 630 |
| Apple___Black_rot | 苹果_黑腐病 | 苹果 | 黑腐病 | 621 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 玉米_灰斑病 | 玉米 | 灰斑病 | 513 |
| Strawberry___healthy | 草莓_健康 | 草莓 | 健康 | 456 |
| Grape___healthy | 葡萄_健康 | 葡萄 | 健康 | 423 |
| Tomato___Tomato_mosaic_virus | 番茄_花叶病毒 | 番茄 | 花叶病毒 | 373 |
| Raspberry___healthy | 覆盆子_健康 | 覆盆子 | 健康 | 371 |
| Peach___healthy | 桃子_健康 | 桃子 | 健康 | 360 |
| Apple___Cedar_apple_rust | 苹果_雪松苹果锈病 | 苹果 | 雪松苹果锈病 | 275 |
| Potato___healthy | 马铃薯_健康 | 马铃薯 | 健康 | 152 |

## 数据分布分析

### 作物分布
- **苹果**: 4 个类别
- **蓝莓**: 1 个类别
- **樱桃**: 2 个类别
- **玉米**: 4 个类别
- **葡萄**: 4 个类别
- **橙子**: 1 个类别
- **桃子**: 2 个类别
- **甜椒**: 2 个类别
- **马铃薯**: 3 个类别
- **覆盆子**: 1 个类别
- **大豆**: 1 个类别
- **南瓜**: 1 个类别
- **草莓**: 2 个类别
- **番茄**: 10 个类别

### 病害类型分布
- **真菌病害**: 19 个类别
- **细菌病害**: 3 个类别
- **病毒病害**: 2 个类别
- **虫害**: 1 个类别
- **生理病害**: 2 个类别
- **健康**: 12 个类别