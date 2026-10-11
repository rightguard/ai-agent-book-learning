
(vllm_wsl) root@DESKTOP-H4LVG45:/mnt/e/ai-agent-book# python char1/search-codegen/distance.py
项目根目录：/mnt/e/ai-agent-book
配置文件：/mnt/e/ai-agent-book/config.yaml
环境文件：/mnt/e/ai-agent-book/.env
模型：qwen3.7-flash-2026-07-15
API 地址：https://dashscope.aliyuncs.com/compatible-mode/v1/responses
[工具] 托管网络搜索完成
[工具] 托管 Python 计算完成

[验收]
  响应正常完成：通过
  托管搜索成功执行：通过
  托管代码成功执行：通过
  最终回答包含至少两个来源URL：通过
  answer_urls：通过
  source_provenance_verified：通过
  Python日志确认计算45对：通过
  Python最近城市对正确：通过
  Python计算距离接近参考值：通过
  生成了最终回答：通过

[最终回答]

### 1. 十个固定城市的坐标与来源链接

以下是通过 `web_search` 获取的十个城市坐标数据，每个城市均提供了至少两个真实来源链接进行交叉核对。

| 序号 | 国家       | 城市 (中文)  | 城市 (英文)         | 纬度 (Latitude) | 经度 (Longitude) | 来源链接 1                                                                        | 来源链接 2                                                                                                     |
| :--- | :--------- | :----------- | :------------------ | :-------------- | :--------------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| 1    | 文莱       | 斯里巴加湾市 | Bandar Seri Begawan | 4.89035° N     | 114.94006° E    | [geodatos.net](https://www.geodatos.net/en/coordinates/brunei/bandar-seri-begawan) | [mapsofworld.com](https://www.mapsofworld.com/lat_long/bandar-seri-begawan.html)                                |
| 2    | 柬埔寨     | 金边         | Phnom Penh          | 11.56245° N    | 104.91601° E    | [latlong.net](https://www.latlong.net/place/phnom-penh-cambodia-6370.html)         | [geodatos.net](https://www.geodatos.net/en/coordinates/cambodia/phnom-penh)                                     |
| 3    | 印度尼西亚 | 雅加达       | Jakarta             | 6.21462° S     | 106.84513° E    | [latlong.net](https://www.latlong.net/place/jakarta-indonesia-27575.html)          | [openstreetmap.org](https://wiki.openstreetmap.org/wiki/Jakarta)                                                |
| 4    | 老挝       | 万象         | Vientiane           | 17.96667° N    | 102.60000° E    | [latlong.net](https://www.latlong.net/place/vientiane-laos-15526.html)             | [latitudelongitude.org](https://latitudelongitude.org/la/vientiane/)                                            |
| 5    | 马来西亚   | 吉隆坡       | Kuala Lumpur        | 3.140853° N    | 101.693207° E   | [latlong.net](https://www.latlong.net/place/kuala-lumpur-malaysia-27237.html)      | [latitudelongitude.org](https://latitudelongitude.org/my/kuala-lumpur/)                                         |
| 6    | 缅甸       | 内比都       | Naypyidaw           | 19.745000° N   | 96.129720° E    | [latitudelongitude.org](https://latitudelongitude.org/mm/nay-pyi-taw/)             | [database.earth](https://database.earth/countries/myanmar/regions/naypyidaw-union-territory/cities/nay-pyi-taw) |
| 7    | 菲律宾     | 马尼拉       | Manila              | 14.599512° N   | 120.984222° E   | [latlong.net](https://www.latlong.net/place/manila-philippines-9339.html)          | [latitudelongitude.org](https://latitudelongitude.org/ph/manila/)                                               |
| 8    | 新加坡     | 新加坡       | Singapore           | 1.290270° N    | 103.851959° E   | [latlong.net](https://www.latlong.net/place/singapore-788.html)                    | [latitudelongitude.org](https://latitudelongitude.org/sg/singapore/)                                            |
| 9    | 泰国       | 曼谷         | Bangkok             | 13.736717° N   | 100.523186° E   | [latlong.net](https://www.latlong.net/place/bangkok-thailand-778.html)             | [latitudelongitude.org](https://latitudelongitude.org/th/bangkok/)                                              |
| 10   | 越南       | 河内         | Hanoi               | 21.028511° N   | 105.804817° E   | [latlong.net](https://www.latlong.net/place/hanoi-vietnam-2759.html)               | [latitudelongitude.org](https://latitudelongitude.org/vn/hanoi/)                                                |

---

### 2. Python 代码执行日志

以下代码使用 `itertools.combinations` 生成了所有 45 对城市组合，并使用 `math` 库中的 Haversine 公式计算球面距离。地球平均半径 $R$ 设定为 **6371.0088 km**。

```python
import math
from itertools import combinations

# Define cities and their coordinates (latitude, longitude)
cities = {
    "Bandar Seri Begawan": (4.89035, 114.94006),
    "Phnom Penh": (11.56245, 104.91601),
    "Jakarta": (-6.21462, 106.84513),
    "Vientiane": (17.96667, 102.6),
    "Kuala Lumpur": (3.140853, 101.693207),
    "Naypyidaw": (19.745, 96.12972),
    "Manila": (14.599512, 120.984222),
    "Singapore": (1.290270, 103.851959),
    "Bangkok": (13.736717, 100.523186),
    "Hanoi": (21.028511, 105.804817)
}

# Earth radius in km
R = 6371.0088

def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c

# Calculate distances for all pairs
pairs = list(combinations(cities.keys(), 2))
distances = []

for city1, city2 in pairs:
    lat1, lon1 = cities[city1]
    lat2, lon2 = cities[city2]
    dist = haversine(lat1, lon1, lat2, lon2)
    distances.append((city1, city2, dist))

# Find the closest pair
closest_pair = min(distances, key=lambda x: x[2])

print(f"PAIR_COUNT={len(pairs)}")
print(f"CLOSEST_PAIR={closest_pair[0]}|{closest_pair[1]}")
print(f"DISTANCE_KM={closest_pair[2]:.2f}")
```

**执行输出结果：**

```text
PAIR_COUNT=45
CLOSEST_PAIR=Kuala Lumpur|Singapore
DISTANCE_KM=316.03
```

---

### 3. 实验结论

* **比较对数**：共计算了 **45** 对城市间的距离 ($C_{10}^2 = 45$)。
* **最近的一对**：**Kuala Lumpur (吉隆坡)** 和 **Singapore (新加坡)**。
* **最短距离**：**316.03 km**。
* **计算方法**：基于 Haversine 公式的大圆距离计算，假设地球为标准球体，半径 $R = 6371.0088 \text{ km}$。

结果已保存：/mnt/e/ai-agent-book/char1/search-codegen/results/asean_distance_20261010_170517.json
