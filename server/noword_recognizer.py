"""
noword/image2sgf CNN 棋盘识别器
基于 https://github.com/noword/image2sgf

使用两个预训练 PyTorch 模型:
  - board.pth: FCOS ResNet50 FPN, 检测棋盘四角
  - stone.pth: EfficientNet B3, 分类每个交叉点 (6类)

石子分类 label bits:
  bit 0: 有无标记 (数字/字母/几何图形)
  bit 1-2: 00=空, 01=黑, 10=白
  → color = label >> 1: 0=空, 1=黑, 2=白
"""

import os
import io
import time
import logging
import numpy as np
import cv2
from PIL import Image
from collections import namedtuple

logger = logging.getLogger(__name__)

# ============== 常量 ==============
DEFAULT_IMAGE_SIZE = 1024
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "image2sgf"
)
BOARD_PTH = os.path.join(MODELS_DIR, "board.pth")
STONE_PTH = os.path.join(MODELS_DIR, "stone.pth")

# 延迟加载 PyTorch
_torch = None
_torchvision = None
_board_model = None
_stone_model = None
_device = None

Point = namedtuple('Point', ['x', 'y'])


def _ensure_torch():
    """延迟加载 torch/torchvision，避免启动时间过长"""
    global _torch, _torchvision
    if _torch is None:
        import torch
        import torchvision
        _torch = torch
        _torchvision = torchvision
    return _torch, _torchvision


def _get_device():
    """获取计算设备 (优先 CUDA)"""
    global _device
    if _device is None:
        torch, _ = _ensure_torch()
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            logger.info(f"noword CNN 使用 CUDA: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device('cpu')
            logger.info("noword CNN 使用 CPU")
    return _device


def _load_models():
    """加载 board + stone 模型（只在首次调用时执行）"""
    global _board_model, _stone_model
    if _board_model is not None and _stone_model is not None:
        return _board_model, _stone_model

    torch, torchvision = _ensure_torch()
    device = _get_device()

    t0 = time.time()

    # Board detection: FCOS ResNet50 FPN
    if os.path.exists(BOARD_PTH):
        _board_model = torchvision.models.detection.fcos_resnet50_fpn(
            num_classes=4 + 1,
            detections_per_img=8,
            score_thresh=0.05,
            weights_backbone=None
        )
        state = torch.load(BOARD_PTH, map_location='cpu', weights_only=False)
        _board_model.load_state_dict(state)
        _board_model.to(device)
        _board_model.eval()
        logger.info(f"board.pth 已加载 ({time.time() - t0:.1f}s)")
    else:
        logger.warning(f"board.pth 未找到: {BOARD_PTH}")

    t1 = time.time()

    # Stone classifier: EfficientNet B3
    if os.path.exists(STONE_PTH):
        _stone_model = torchvision.models.efficientnet_b3(num_classes=6)
        state = torch.load(STONE_PTH, map_location='cpu', weights_only=False)
        _stone_model.load_state_dict(state)
        _stone_model.to(device)
        _stone_model.eval()
        logger.info(f"stone.pth 已加载 ({time.time() - t1:.1f}s)")
    else:
        logger.warning(f"stone.pth 未找到: {STONE_PTH}")

    logger.info(f"noword 模型加载完成，总耗时 {time.time() - t0:.1f}s")
    return _board_model, _stone_model


# ============== 几何工具类 ==============

class GridPosition:
    """棋盘网格位置计算（复刻 noword/image2sgf 的实现）"""

    def __init__(self, width, size=19, board_rate=0.8):
        self.width = width
        self.size = size
        self.board_rate = board_rate

        margin = width * (1 - board_rate) / 2
        self.grid_size = width * board_rate / (size - 1)
        self.half_grid_size = self.grid_size / 2

        # _grid_pos[row][col] = Point(x, y)
        # row 0 = 底部, row 18 = 顶部 (棋盘坐标系)
        self._grid_pos = []
        for row in range(size):
            row_pos = []
            for col in range(size):
                x = int(margin + col * self.grid_size)
                y = int(width - margin - row * self.grid_size)
                row_pos.append(Point(x, y))
            self._grid_pos.append(row_pos)

    def __getitem__(self, index):
        return self._grid_pos[index]


class BoxPosition(GridPosition):
    """在 GridPosition 基础上计算每个交叉点的 bounding box"""

    def __init__(self, width, size=19, board_rate=0.8):
        super().__init__(width, size, board_rate)
        self._boxes = []
        for row in self._grid_pos:
            self._boxes.append([
                [
                    max(x - self.half_grid_size, 0),
                    max(y - self.half_grid_size, 0),
                    min(x + self.half_grid_size, width),
                    min(y + self.half_grid_size, width)
                ]
                for x, y in row
            ])

    def __getitem__(self, index):
        return self._boxes[index]


class NpBoxPosition(BoxPosition):
    """numpy 版 BoxPosition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._boxes = np.array(self._boxes)


# ============== 核心函数 ==============

def expand_image(pil_image):
    """将图片扩展为正方形（灰色填充居中）"""
    w, h = pil_image.size
    size = max(w, h)
    new_image = Image.new('RGB', (size, size), (128, 128, 128))
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    new_image.paste(pil_image, (x_offset, y_offset))
    return new_image, x_offset, y_offset


def detect_board_corners(board_model, image, expand=True):
    """
    用 FCOS 模型检测棋盘四个角点
    返回: (boxes[4,4], scores[4]) 四个角的 bbox 和置信度
    """
    torch, torchvision = _ensure_torch()
    device = _get_device()

    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if expand:
        img, x_offset, y_offset = expand_image(image)
    else:
        img = image
        x_offset = y_offset = 0

    T = _torchvision.transforms
    with torch.no_grad():
        tensor = T.ToTensor()(img).unsqueeze(0).to(device)
        target = board_model(tensor)[0]

    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.1)
    _boxes = target['boxes'].detach()[nms].cpu()
    _labels = target['labels'].detach()[nms].cpu()
    _scores = target['scores'].detach()[nms].cpu()

    if len(set(_labels.tolist())) < 4:
        raise ValueError(f"只检测到 {len(set(_labels.tolist()))} 个角点，需要 4 个")

    boxes = np.zeros((4, 4))
    scores = [0] * 4
    for i, box in enumerate(_boxes):
        label = int(_labels[i]) - 1
        if 0 <= label < 4 and np.count_nonzero(boxes[label]) == 0:
            boxes[label] = box.numpy()
            scores[label] = float(_scores[i])

    if np.any(np.all(boxes == 0, axis=1)):
        raise ValueError("未能检测到所有 4 个角点")

    boxes[:, ::2] -= x_offset
    boxes[:, 1::2] -= y_offset

    return boxes, scores


def perspective_correct(board_model, img, expand=True):
    """
    检测四角并做透视变换，输出 1024x1024 的校正棋盘图
    返回: (corrected_image, boxes, scores)
    """
    boxes, scores = detect_board_corners(board_model, img, expand)
    box_pos = NpBoxPosition(width=DEFAULT_IMAGE_SIZE, size=19)

    startpoints = boxes[:, :2].tolist()
    endpoints = [
        box_pos[18][0][:2].tolist(),  # top left
        box_pos[18][18][:2].tolist(),  # top right
        box_pos[0][0][:2].tolist(),    # bottom left
        box_pos[0][18][:2].tolist()    # bottom right
    ]

    transform = cv2.getPerspectiveTransform(
        np.array(startpoints, np.float32),
        np.array(endpoints, np.float32)
    )
    _img = cv2.warpPerspective(
        np.array(img), transform,
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    )

    return Image.fromarray(_img), boxes, scores


def classify_stones(stone_model, corrected_image):
    """
    将校正后的 1024x1024 图切割为 19x19 格，用 EfficientNet B3 分类
    返回: results[19][19], 每个值 = label (color = label >> 1)
    """
    torch, _ = _ensure_torch()
    device = _get_device()
    T = _torchvision.transforms

    box_pos = NpBoxPosition(width=DEFAULT_IMAGE_SIZE, size=19)
    img_tensor = T.ToTensor()(corrected_image)

    grid_h = int(box_pos.grid_size)
    imgs = torch.empty((19 * 19, 3, grid_h, grid_h))

    for y in range(19):
        for x in range(19):
            x0, y0, x1, y1 = box_pos[y][x].astype(int)
            tile = img_tensor[:, y0:y1, x0:x1]
            # 处理边界情况：tile 尺寸可能不是 grid_h
            if tile.shape[1] != grid_h or tile.shape[2] != grid_h:
                tile = torch.nn.functional.interpolate(
                    tile.unsqueeze(0), size=(grid_h, grid_h),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            imgs[x + y * 19] = tile

    with torch.no_grad():
        imgs = imgs.to(device)
        results = stone_model(imgs).argmax(1).cpu()

    return results.reshape(19, 19)


# ============== 主入口 ==============

def recognize_board_noword(image_bytes, board_size=19):
    """
    用 noword/image2sgf 的 CNN 模型识别棋盘

    Args:
        image_bytes: 图片二进制数据
        board_size: 棋盘大小 (目前仅支持 19)

    Returns:
        dict: {
            "board": [[0,1,2,...], ...],  # 19x19, 0=空 1=黑 2=白
            "confidence": float,
            "method": "noword-cnn",
            "corners_score": [float, ...],
            "time": float
        }
    """
    if board_size != 19:
        return {
            "board": [[0] * board_size for _ in range(board_size)],
            "confidence": 0,
            "method": "noword-cnn",
            "error": "noword CNN 目前仅支持 19 路棋盘"
        }

    t0 = time.time()

    try:
        board_model, stone_model = _load_models()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return {
            "board": [[0] * 19 for _ in range(19)],
            "confidence": 0,
            "method": "noword-cnn",
            "error": f"模型加载失败: {str(e)}"
        }

    if board_model is None or stone_model is None:
        return {
            "board": [[0] * 19 for _ in range(19)],
            "confidence": 0,
            "method": "noword-cnn",
            "error": "模型文件缺失 (需要 board.pth 和 stone.pth)"
        }

    # 解码图片
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Step 1: 检测四角 + 透视校正
    try:
        corrected, boxes, scores = perspective_correct(board_model, pil_image, expand=True)
    except Exception as e1:
        logger.warning(f"带扩展的检测失败 ({e1})，尝试不扩展...")
        try:
            corrected, boxes, scores = perspective_correct(board_model, pil_image, expand=False)
        except Exception as e2:
            logger.error(f"棋盘角点检测失败: {e2}")
            return {
                "board": [[0] * 19 for _ in range(19)],
                "confidence": 0,
                "method": "noword-cnn",
                "error": f"无法检测棋盘角点: {str(e2)}"
            }

    corner_conf = min(scores)
    logger.info(f"四角检测置信度: {[f'{s:.2f}' for s in scores]}")

    # 如果角点置信度低，尝试二次校正
    if corner_conf < 0.7:
        try:
            corrected2, boxes2, scores2 = perspective_correct(board_model, corrected, expand=True)
            if sum(scores2) > sum(scores):
                corrected = corrected2
                scores = scores2
                logger.info(f"二次校正提升: {[f'{s:.2f}' for s in scores2]}")
        except Exception:
            pass  # 二次校正失败就用一次的结果

    # Step 2: 分类每个交叉点
    board_raw = classify_stones(stone_model, corrected)

    # 转换为标准格式
    # board_raw = results.reshape(19, 19), results[y][x]
    # 其中 y 来自 box_pos 的 row 维 (y=0 底部, y=18 顶部)
    #      x 来自 box_pos 的 col 维 (x=0 左侧, x=18 右侧)
    # 我们的格式: board[row][col], row 0=顶部
    # → board[r][c] = results[18-r][c] >> 1
    board = []
    for y in range(18, -1, -1):  # y=18(顶) → y=0(底)
        row = []
        for x in range(19):
            color = int(board_raw[y][x]) >> 1  # results[y][x] 对应 box_pos[y][x]
            row.append(color)
        board.append(row)

    elapsed = time.time() - t0
    avg_conf = sum(scores) / len(scores)

    # 统计
    black_count = sum(1 for row in board for c in row if c == 1)
    white_count = sum(1 for row in board for c in row if c == 2)
    logger.info(f"noword CNN 识别完成: 黑{black_count}子, 白{white_count}子, "
                f"置信度{avg_conf:.2f}, 耗时{elapsed:.1f}s")

    return {
        "board": board,
        "confidence": round(avg_conf, 3),
        "method": "noword-cnn",
        "corners_score": [round(s, 3) for s in scores],
        "time": round(elapsed, 2),
        "stats": {
            "black": black_count,
            "white": white_count,
            "empty": 361 - black_count - white_count
        }
    }


# ============== 检查可用性 ==============

def is_available():
    """检查 noword CNN 识别器是否可用"""
    return os.path.exists(BOARD_PTH) and os.path.exists(STONE_PTH)


NOWORD_AVAILABLE = is_available()

if NOWORD_AVAILABLE:
    logger.info(f"noword CNN 识别器: ✅ 可用 (board.pth + stone.pth)")
else:
    missing = []
    if not os.path.exists(BOARD_PTH):
        missing.append("board.pth")
    if not os.path.exists(STONE_PTH):
        missing.append("stone.pth")
    logger.info(f"noword CNN 识别器: ❌ 缺少 {', '.join(missing)}")
