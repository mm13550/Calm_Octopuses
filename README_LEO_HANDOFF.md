# Leo Handoff

## 0. 你现在主要负责什么

按当前 repo、`PLAN.md`、以及项目 proposal PDF 的实际分工，Leo 现在最适合负责的是：

- embeddings pipeline
- vector DB ingestion / schema
- retrieval backend
- retrieval scoring / filtering

你**不需要负责** Merry 那一层的前端交互式 multimodal search UI。

更准确地说：

- Merry 负责“用户怎么搜、界面怎么呈现”
- Leo 负责“后端怎么 embed、怎么存、怎么查、怎么返回结果”

所以你做的是 **multimodal retrieval backend**，不是完整的 product/UI owner。

## 1. Neil 已经交给你的数据

目前可直接拿来接手的主要文件：

- `data/social_reviews.csv`
- `data/social_images.csv`
- `data/images/`
- `data/extracted_menus/final_parsed_menus.json`
- `data/extracted_bios/restaurant_bios_joinable.json`
- `data/csv/restaurant_lookup.csv`

注意：

- `social_images.csv` 里的 `image_path` 现在已经改成 repo-relative path，例如 `data/images/...`
- 旧的 `data/extracted_bios/restaurant_bios.json` 已经从 GitHub 移除
- 现在应该以 `restaurant_bios_joinable.json` 为准

## 2. 先说最重要的 join key

之后所有表和向量层，**一律以 `rest_id` 为主键中心**。

不要再用 `restaurant_name` 当主 join key，因为：

- menu / reviews / images 里主键稳定的是 `rest_id`
- name 会有别名、重音符、大小写、后缀差异
- Craig / Grace / Merry 后面都需要稳定 join

建议你把 `data/csv/restaurant_lookup.csv` 当 canonical lookup table：

- `rest_id`
- `restaurant_name`
- `homepage`
- `borough`
- `michelin_category`
- `match_source`

## 3. Image cleaning 你应该怎么做

Neil 前面说的 cleaning，不是指简单的 CSV cleaning，而是指：

- 从 scrape 回来的 social food images 里去掉 noise
- 不要把明显无关图片直接送进 embedding

这一步很重要，因为如果脏图太多：

- Merry 的图搜图结果会被带偏
- Craig 后面做 restaurant-level style fusion 会被噪声污染
- 你自己做 mean pooling / aggregation 也会失真

### 3.1 哪些图属于 noise

建议优先过滤这些：

- logo / banner / icon
- 室外门头、纯室内环境图、地图截图
- 人像自拍、多人合照
- 菜单截图、文字海报、QR code
- 极度模糊、过暗、过曝
- 明显不是食物主体的图

### 3.2 最现实的 cleaning 方案

不要一开始就上复杂模型，先做一个 2-stage filter：

#### Stage A: rule-based sanity check

- 文件存在
- 能正常打开
- 尺寸不要过小
- 宽高比不要极端离谱
- 去掉完全重复文件或接近重复文件

#### Stage B: lightweight semantic filter

用一个现成 vision-language model 做二分类/打分，判断：

- food dish close-up
- plated food / dining table food
- restaurant interior
- people / selfie
- menu / text-heavy image
- unrelated / low quality

最简单的方法：

- 直接用 CLIP / SigLIP 做 zero-shot label scoring
- 保留 food-related score 高的图
- 对 borderline 样本人工 spot check 一小批

### 3.3 建议的输出

不要覆盖 raw data。建议新增一个 cleaned metadata 文件，例如：

- `data/social_images_cleaned.csv`

最少加这些字段：

- `image_uid`
- `rest_id`
- `source`
- `image_path`
- `keep_for_embedding` (`0/1`)
- `filter_reason`
- `quality_score`
- `food_score`

这样以后：

- Merry 如果想看全量图还有原始记录
- 你做 embedding 时只取 `keep_for_embedding = 1`
- Craig 可以只聚合 cleaned images

## 4. Embedding 应该做什么层级

这里不要只做一种粒度。最稳的是 **两层结构**：

- document/item-level embeddings: 给检索用
- restaurant-level fused representation: 给 clustering / downstream analytics 用

### 4.1 Social images: image-level embeddings

对每一张保留下来的图片单独做 embedding。

推荐单位：

- 一张图 = 一条向量
- key = `image_uid`

为什么：

- Merry 的 image search 需要细粒度结果
- 聚合前保留最大灵活性
- 后面可做 per-restaurant mean / weighted mean

### 4.2 Menus: menu-item-level text embeddings

不要只做餐厅级 menu embedding。

推荐单位：

- 一条 parsed menu row = 一条向量

建议 embed 文本：

- `dish_name + ingredients + restaurant_name`

为什么：

- 你们产品目标是 search by dish，不只是 search by restaurant
- “uni pasta / duck / omakase / sea urchin” 这类 query 需要 dish-level recall

### 4.3 Reviews: review-level text embeddings

review 最好保留单条 embedding，但我建议把它作为 **secondary retrieval corpus**，不是默认主入口。

推荐单位：

- 一条 review = 一条向量

为什么：

- Grace 需要 review 级文本做 ABSA
- review 语义可以补 menu 稀疏餐厅
- 但 review 很 noisy，不适合一上来压过 official menu

### 4.4 Bios: restaurant-level text embedding

bio 不需要拆太细。

推荐单位：

- 一家餐厅一条 bio embedding

建议把这三段拼起来：

- `description`
- `culinary_style`
- `history`

bio 更适合进入 restaurant-level profile，而不是直接做主检索表核心语料。

## 5. Vector DB 不要怎么设计

不要把所有 modality、所有模型输出、所有维度都硬塞进一张 ANN 表里。

尤其注意：

- image embeddings 和 text embeddings 不一定来自同一个模型
- review embeddings 可能跟 menu embeddings 不是同一向量空间
- 如果维度不一致，根本不能共用一张向量列

所以建议 **分表**。

## 6. 建议的 DB 设计

推荐至少 4 张表。

### Table A: `image_vectors`

用途：

- image-to-image retrieval
- image-side evidence for Merry

一行代表一张 cleaned image。

建议字段：

- `image_uid`
- `rest_id`
- `restaurant_name`
- `source`
- `image_path`
- `vector`
- `quality_score`
- `food_score`
- `created_at` if available
- `metadata`

### Table B: `menu_item_vectors`

用途：

- text-to-menu retrieval
- text query 找 dish

一行代表一个 parsed menu item。

建议字段：

- `doc_id`
- `rest_id`
- `restaurant_name`
- `dish_name`
- `ingredients`
- `price`
- `text_for_embedding`
- `vector`
- `source = official_menu`

### Table C: `review_vectors`

用途：

- semantic backoff
- supplemental retrieval
- later aggregation to restaurant profile

一行代表一条 review。

建议字段：

- `uid`
- `rest_id`
- `source`
- `text`
- `rating`
- `vector`

### Table D: `restaurant_profiles`

用途：

- Craig 的 clustering / fused features
- Merry 的 restaurant card / badges / style tags
- Grace 的 ABSA 回填

一行代表一家餐厅。

建议字段：

- `rest_id`
- `restaurant_name`
- `bio_text`
- `bio_vector`
- `mean_image_vector`
- `mean_menu_vector`
- `mean_review_vector`
- `fused_vector`
- `absa_food`
- `absa_service`
- `absa_ambiance`
- `image_count_clean`
- `review_count`
- `menu_item_count`
- `metadata`

## 7. 给 Grace / Merry / Craig 留什么接口

### 给 Grace

Grace 需要的不是“已经被你过度压缩后的 review blob”，而是：

- review-level text
- stable `rest_id`
- 可回写 restaurant-level ABSA result 的位置

所以你不要太早把 review 全压成一条 string。

### 给 Merry

Merry 需要：

- Top-K retrieval result
- 结果里带 `rest_id`
- 能拿到 `image_path / dish_name / restaurant_name`
- 能过滤 content type

所以 retrieval function 返回结果时别只给 vector score，至少要返回：

- result id
- content type
- restaurant name
- dish name if any
- image path if any
- source
- score

### 给 Craig

Craig 需要的是 **restaurant-level fused features**，不是一堆 raw documents。

你需要给他：

- 一家餐厅一条 profile row
- 可聚类的 fused feature
- image/menu/review 的 count 或 summary stats

这样 Craig 才能稳定做：

- UMAP
- GMM
- style clusters

## 8. 推荐的实现顺序

建议按这个顺序做，不容易返工：

1. 用 `restaurant_lookup.csv` 冻结 canonical restaurant ids
2. 对 `social_images.csv` 做 file existence check
3. 做 image noise filtering，产出 `social_images_cleaned.csv`
4. 先做 `image_vectors`
5. 再做 `menu_item_vectors`
6. 再做 `review_vectors`
7. 建 `restaurant_profiles`
8. 再暴露 retrieval API 给 Merry
9. 再把 restaurant-level fused output 交给 Craig / Grace

## 9. 当前你需要特别注意的两个现实问题

### 问题 A: Yelp 图虽然已经补回来了，但 future runs 仍可能不稳定

项目里已经新增了：

- `pipelines/refetch_missing_yelp_images.py`

用途：

- 只补抓缺失 Yelp 图片
- 不修改 `social_images.csv`

这类补抓逻辑以后可以保留，因为 Yelp / Apify 偶尔会被 403 或 proxy 问题打断。

### 问题 B: 不要直接对 raw images 全量平均

如果不过滤 noise，restaurant-level mean image vector 会很脏。

建议：

- 先 clean
- 再 embed
- 再 aggregate

不要反过来。

## 10. 一句话总结

你现在最该做的是：

**clean image metadata -> filter noisy images -> build item-level embeddings -> ingest into well-separated vector tables -> aggregate restaurant-level profiles for Craig/Grace/Merry**

而不是直接把所有东西糊进一个表，然后再让别人自己想办法 join。
