# M1-1 岩心箱旋转校正算法模块实现方案

## 1. 模块定位

M1-1 岩心箱旋转校正是 Geo-Core AI V2.1 后端核心算法链路中的几何预处理模块，位于原始影像读取之后、岩心柱体前景掩膜和深度分割之前。它的目标是将轨道履带连续采集得到的竖向长条岩心影像，自动拆分为若干独立岩心箱，并将每个岩心箱校正为长边与深度方向一致、边界规整、可被后续算法稳定调用的标准影像单元。

本仓库当前样例数据为 ENVI 标准格式：

| 文件 | 说明 |
| --- | --- |
| `RGB-20230909_141858-00000.dat` | 原始长条 RGB 影像数据 |
| `RGB-20230909_141858-00000.hdr` | ENVI 头文件 |

头文件关键参数：

| 参数 | 值 |
| --- | --- |
| samples | 2048 |
| lines | 22480 |
| bands | 3 |
| data type | 1, uint8 |
| interleave | BIL |

## 2. 输入、输出与核心约束

### 2.1 输入

1. 原始长条岩心影像：`dat + hdr`、TIFF、PNG、JPG 或后续融合后的多波段影像。
2. 可选人工参数：手动旋转角、ROI、箱体裁剪边距、阈值上下限。
3. 可选业务元数据：钻孔编号、回次、箱号、采集时间、深度方向约定。

### 2.2 输出

1. `box_0001`、`box_0002` 等独立岩心箱校正影像。
2. 每个岩心箱的二值掩膜、裁剪框、旋转角、旋转矩阵、置信度和质量指标。
3. 批处理元数据文件，例如 `metadata.json`。
4. QA 预览图，例如原始长条图上叠加箱体框线、校正前后对比图。

### 2.3 关键约束

1. 高光谱或融合影像旋转时必须优先使用最近邻采样，避免插值改变像元光谱值。
2. RGB 可视化预览可以单独生成抗锯齿版本，但算法主输出仍以最近邻校正结果为准。
3. 不改变原始像素辐射值。亮度增强、CLAHE、阈值归一化等操作只用于检测分支，不写入主输出影像。
4. 岩心箱顺序以原始长条影像从上到下排序，默认对应深度方向递增，具体方向可由业务配置覆盖。
5. 每次自动校正都必须保存矩阵和参数，使前端人工微调后可以复现。

## 3. 总体算法流水线

```text
原始长条影像
  -> ENVI/常规影像读取
  -> 低分辨率检测缩略图生成
  -> 光照与背景归一化
  -> 岩心箱候选区域分割
  -> 连通域与投影分析拆分多个岩心箱
  -> 箱体边界线检测
  -> 角度估计与鲁棒融合
  -> 最近邻旋转校正
  -> 标准裁剪与排序编号
  -> 影像、掩膜、元数据、QA 报告输出
```

## 4. 详细实现设计

### 4.1 数据读取层

整个模块以 Python 包形式实现。对 ENVI BIL 数据，优先采用 NumPy 内存映射读取，避免一次性复制大文件。

建议实现：

```python
def read_envi_image(dat_path: str, hdr_path: str) -> tuple[np.ndarray, dict]:
    """Read ENVI image and return array in HWC order plus metadata."""
```

对当前样例数据，读取逻辑为：

1. 从 `.hdr` 解析 `samples=2048`、`lines=22480`、`bands=3`、`interleave=bil`、`data type=1`。
2. 用 `np.memmap` 映射为 `(lines, bands, samples)`。
3. 转置为 `(lines, samples, bands)` 的 HWC 视图。
4. 后续所有几何变换都在 HWC 或分块 HWC 上统一执行。

可选依赖：

| 依赖 | 用途 |
| --- | --- |
| `numpy` | 数组、memmap、矩阵计算 |
| `opencv-python` | 阈值、形态学、轮廓、Hough/LSD、旋转 |
| `scipy` | 平滑、峰值检测、鲁棒统计 |
| `scikit-image` | 可选的形态学和测量工具 |
| `pydantic` | 配置和结果模型 |
| `tifffile` | 多波段 TIFF 输出 |

### 4.2 检测缩略图与光照归一化

原始影像高度较大，应先生成检测缩略图完成候选定位，再回到原分辨率精修。

处理方法：

1. 保持宽高比例，下采样到宽度约 `512` 或 `768` 像素。
2. 将 RGB 转为灰度、HSV 的 V 通道或 Lab 的 L 通道。
3. 使用大核高斯模糊或形态学开运算估计缓变背景。
4. 用 `normalized = gray - background` 或 `gray / (background + eps)` 增强箱体边缘与岩心纹理。
5. 对检测图使用 CLAHE 只增强可分割性，不回写到输出。

### 4.3 岩心箱与履带背景分离

履带背景通常更暗、更连续；岩心箱边框、岩心、隔板和箱底具有更高局部纹理和边缘密度。建议采用传统视觉优先、深度学习可插拔的双路径。

基础传统流程：

1. 基于灰度或 Lab-L 通道执行 Otsu 阈值、自适应阈值或分位数阈值。
2. 叠加边缘密度图：`Canny -> dilate`，将箱体边框和岩心纹理纳入前景候选。
3. 使用形态学闭合连接断裂边界，使用开运算去除履带上的小噪声。
4. 填洞得到箱体候选区域。
5. 连通域筛选：
   - 面积大于最小岩心箱面积；
   - 宽度占原图宽度的合理比例；
   - 高宽比符合岩心箱长条形态；
   - 中心 x 坐标接近轨道中线；
   - 与图像边界保留一定安全距离。

深度学习增强路径：

1. 当传统分割质量不稳定时，接入 `YOLO/RT-DETR` 检测岩心箱外接框，或接入 `Mask R-CNN/SegFormer` 输出箱体实例掩膜。
2. 深度模型输出仍进入同一套几何校正函数，保证后续接口不变。
3. 第一版不依赖深度模型即可运行，深度模型作为后续失败样本库积累后的增强项。

### 4.4 多岩心箱拆分

长条影像中可能包含多个岩心箱，且箱间距较小。拆分策略建议组合使用：

1. 纵向投影：对前景掩膜沿 x 方向求和，得到 `profile_y`。
2. 使用平滑后的 `profile_y` 寻找低谷，低谷通常对应箱体之间的履带间隔。
3. 连通域：若相邻箱体未粘连，直接按连通域排序。
4. 分裂粘连区域：若一个连通域高度明显超过单箱高度，可在 `profile_y` 低谷处切分。
5. 水平边线辅助：用 Hough/LSD 检测近水平线，作为箱体上边缘、下边缘或箱间隔的证据。

输出候选结构：

```python
@dataclass
class CoreBoxCandidate:
    box_id: int
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray
    contour: np.ndarray
    source_scale: float
    split_score: float
```

### 4.5 箱体角度估计

岩心箱长边应与图像 y 轴一致。旋转角估计采用多证据鲁棒融合：

1. `cv2.minAreaRect`：基于候选掩膜外轮廓得到粗角度。
2. Hough 直线检测：提取近竖直长边线，统计直线角度直方图。
3. LSD 线段检测：比 Hough 更适合弱边缘和断续边缘。
4. PCA 主轴估计：对箱体掩膜或边缘点做主轴方向分析。
5. RANSAC 主方向拟合：在边缘点中剔除岩心碎块、反光和噪声线段。

融合规则：

1. 只接收绝对角度在合理范围内的候选，例如 `[-10 deg, 10 deg]`，可配置。
2. 优先采用长边线段支持最多的角度。
3. 若 Hough/LSD 与 PCA 差异小于阈值，使用加权平均。
4. 若差异大，选择质量分最高的方法，并降低总体置信度。
5. 若置信度低于阈值，输出 `needs_manual_review=True`。

角度定义：

```text
theta_deg > 0: 需要逆时针旋转 theta_deg，使箱体长边回到竖直方向
theta_deg < 0: 需要顺时针旋转 abs(theta_deg)
```

### 4.6 最近邻旋转与标准裁剪

核心校正函数：

```python
def rotate_with_nearest(
    image: np.ndarray,
    angle_deg: float,
    center_xy: tuple[float, float],
    fill_value: int | float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image with nearest-neighbor sampling and return image plus 2x3 matrix."""
```

实现要求：

1. 使用 `cv2.getRotationMatrix2D` 计算旋转矩阵。
2. 计算旋转后完整画布大小，避免岩心箱角点被裁掉。
3. 使用 `cv2.warpAffine(..., flags=cv2.INTER_NEAREST)`。
4. 对 RGB、多波段影像、掩膜使用同一矩阵。
5. 校正后根据旋转后的箱体掩膜或边界点裁剪紧致矩形。
6. 裁剪时保留可配置边距，例如 `margin_px=10`。
7. 默认不 resize 岩心箱，避免改变像素尺度；如前端需要统一显示尺寸，另生成 preview。

### 4.7 标准化输出

推荐输出目录：

```text
outputs/m1_1/
  corrected_boxes/
    box_0001.png
    box_0002.png
  masks/
    box_0001_mask.png
    box_0002_mask.png
  previews/
    strip_detection_overlay.png
    box_0001_before_after.png
  metadata.json
  qa_report.md
```

对于多波段或高光谱数据，建议输出：

1. `box_0001.npy`：保存完整波段数组。
2. `box_0001.tif`：可选多波段 TIFF。
3. `box_0001_preview.png`：RGB 快速预览。

元数据示例：

```json
{
  "module": "M1-1",
  "input": "RGB-20230909_141858-00000.dat",
  "boxes": [
    {
      "box_id": "box_0001",
      "order_index": 1,
      "bbox_xyxy_raw": [560, 132, 1510, 3180],
      "angle_deg": -1.24,
      "rotation_matrix_2x3": [[0.9998, -0.0216, 31.2], [0.0216, 0.9998, -12.7]],
      "confidence": 0.91,
      "needs_manual_review": false,
      "output_image": "corrected_boxes/box_0001.png",
      "output_mask": "masks/box_0001_mask.png"
    }
  ]
}
```

## 5. Python 包结构建议

```text
geocore_m1_1/
  __init__.py
  config.py
  io_envi.py
  preprocess.py
  segment_boxes.py
  geometry.py
  correction.py
  quality.py
  pipeline.py
  cli.py
  api.py
tests/
  test_envi_reader.py
  test_angle_estimation.py
  test_pipeline_sample.py
```

模块职责：

| 文件 | 职责 |
| --- | --- |
| `config.py` | 参数模型、默认阈值、路径配置 |
| `io_envi.py` | ENVI hdr/dat 读取和写出 |
| `preprocess.py` | 缩略图、光照归一化、检测增强 |
| `segment_boxes.py` | 岩心箱候选掩膜、连通域、箱体拆分 |
| `geometry.py` | Hough/LSD/PCA/RANSAC 角度估计 |
| `correction.py` | 最近邻旋转、裁剪、坐标变换 |
| `quality.py` | 角度一致性、置信度、QA 图生成 |
| `pipeline.py` | 端到端 M1-1 流程 |
| `cli.py` | 命令行入口 |
| `api.py` | FastAPI 或后端服务适配 |

核心配置模型：

```python
class M11Config(BaseModel):
    thumbnail_width: int = 768
    min_box_area_ratio: float = 0.02
    max_abs_angle_deg: float = 10.0
    crop_margin_px: int = 12
    rotate_interpolation: str = "nearest"
    use_lsd: bool = True
    use_hough: bool = True
    manual_angle_delta_deg: float = 0.0
    confidence_threshold: float = 0.75
```

端到端入口：

```python
def run_m11_rotation_correction(
    input_path: str,
    hdr_path: str | None,
    output_dir: str,
    config: M11Config | None = None,
) -> M11BatchResult:
    """Detect, split, rotate-correct, crop and export core-box images."""
```

## 6. CLI 与后端 API 设计

### 6.1 CLI

```bash
python -m geocore_m1_1.cli ^
  --input RGB-20230909_141858-00000.dat ^
  --hdr RGB-20230909_141858-00000.hdr ^
  --output outputs/m1_1 ^
  --save-preview
```

手动微调：

```bash
python -m geocore_m1_1.cli ^
  --input RGB-20230909_141858-00000.dat ^
  --hdr RGB-20230909_141858-00000.hdr ^
  --output outputs/m1_1_manual ^
  --manual-angle-delta -0.35 ^
  --box-id box_0003
```

### 6.2 后端 API

建议 REST 接口：

```text
POST /api/algorithms/m1-1/rotation-correction
```

请求体：

```json
{
  "input_path": "RGB-20230909_141858-00000.dat",
  "hdr_path": "RGB-20230909_141858-00000.hdr",
  "output_dir": "outputs/m1_1",
  "manual_angle_delta_deg": 0.0,
  "save_preview": true
}
```

响应体：

```json
{
  "status": "success",
  "box_count": 7,
  "metadata_path": "outputs/m1_1/metadata.json",
  "preview_path": "outputs/m1_1/previews/strip_detection_overlay.png",
  "boxes": [
    {
      "box_id": "box_0001",
      "angle_deg": -1.24,
      "confidence": 0.91,
      "needs_manual_review": false,
      "image_url": "/outputs/m1_1/corrected_boxes/box_0001.png"
    }
  ]
}
```

## 7. 质量评价与验收指标

### 7.1 自动检测指标

1. 岩心箱漏检率：样例和标注集上尽量为 0。
2. 岩心箱误检率：履带、反光和阴影不应被输出为独立箱体。
3. 箱体边界完整性：校正后裁剪图应包含完整岩心箱边框，不截断箱体。
4. 顺序正确性：输出箱号必须与原始长条影像从上到下顺序一致。

### 7.2 旋转校正指标

1. 自动校正后长边角度误差：目标 `<= 0.5-1.0 deg`。
2. Hough/LSD/PCA 角度一致性：作为置信度核心依据。
3. 最近邻旋转后像元值检查：输出像元值应来自原始像元集合，不产生插值新值。
4. 人工微调后保存校正矩阵，可复现同一输出。

### 7.3 工程指标

1. 单张 `2048 x 22480 x 3` 影像可在普通工作站内完成处理。
2. 支持大图 memmap 或分块处理，避免内存峰值不可控。
3. 失败样本必须输出 `needs_manual_review=True` 和可视化 QA 图。
4. 所有结果可被 M1-2、M1-3 模块按 `CoreBox` 单元继续调用。

## 8. 失败场景与兜底策略

| 场景 | 风险 | 兜底策略 |
| --- | --- | --- |
| 岩心箱与履带亮度接近 | 阈值分割不稳定 | 叠加边缘密度、局部对比增强、人工 ROI |
| 箱体边缘被遮挡或反光 | 角度估计偏移 | 多方法角度融合，低置信度转人工微调 |
| 相邻岩心箱间距过小 | 连通域粘连 | 纵向投影低谷和水平线检测联合切分 |
| 箱体局部缺失或出画 | 裁剪不完整 | 标记为不完整箱体，不进入自动深度分割 |
| 岩心碎块纹理强于箱边 | 直线检测被干扰 | RANSAC 剔除短线、优先使用长边框线 |
| 多波段数据体量大 | 内存压力高 | memmap、分块 warp、只用 RGB/代表波段检测 |

## 9. 与后续模块的接口关系

M1-1 输出的每个岩心箱应作为统一数据单元 `CoreBox`：

```python
@dataclass
class CoreBox:
    box_id: str
    image_path: str
    mask_path: str
    order_index: int
    angle_deg: float
    rotation_matrix: list[list[float]]
    bbox_xyxy_raw: tuple[int, int, int, int]
    bbox_xyxy_corrected: tuple[int, int, int, int]
    confidence: float
    needs_manual_review: bool
```

后续关系：

1. M1-2 岩心柱体前景掩膜：在已校正的单箱图上提取岩心柱、隔板、空槽等前景。
2. M1-3 岩心柱体分割与深度标记：基于规整箱体建立像素到深度的线性或分段映射。
3. M2 系列智能编录：使用 M1-3 输出的标准图斑进行颜色、粒度、岩性、矿物、结构构造等识别。

## 10. 开发里程碑

### V0.1 最小可运行原型

1. 完成 ENVI BIL 读取。
2. 生成长条缩略图。
3. 用阈值、形态学和连通域检测岩心箱候选。
4. 用 `minAreaRect + PCA` 估计角度。
5. 使用最近邻旋转输出独立箱体 PNG 和 `metadata.json`。

### V0.2 稳定传统视觉版

1. 引入 Hough/LSD 直线检测。
2. 引入纵向投影拆分粘连箱体。
3. 输出 QA 预览图和置信度。
4. 加入人工角度微调参数。
5. 建立单元测试和样例端到端测试。

### V0.3 后端服务集成版

1. 封装为 Python 包。
2. 提供 CLI 和 FastAPI 调用接口。
3. 支持多波段 TIFF 或 NPY 输出。
4. 与前端校正工具联调，支持自动校正和手动微调保存。

### V1.0 可验收版

1. 在多批次岩心箱样本上统计漏检、误检、角度误差。
2. 形成失败样本库和 QA 报告。
3. 提供配置文件、日志和异常处理。
4. 输出符合 M1-2/M1-3 消费格式的 `CoreBox` 数据清单。

## 11. 推荐第一步实现顺序

1. 写 `io_envi.py`，确认样例 `.dat/.hdr` 可读并生成 RGB 预览。
2. 写 `segment_boxes.py`，在缩略图上找到所有岩心箱候选区。
3. 写 `geometry.py`，比较 `minAreaRect`、Hough、PCA 三种角度估计。
4. 写 `correction.py`，实现最近邻旋转和标准裁剪。
5. 写 `pipeline.py` 和 `cli.py`，端到端输出 `outputs/m1_1`。
6. 补充 QA 图和测试，再准备接入后端 API。

