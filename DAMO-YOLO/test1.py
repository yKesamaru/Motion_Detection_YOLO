import numpy as np
import torch
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.structures.bounding_box import BoxList
from damo.utils import postprocess
from damo.utils.demo_utils import transform_img
from PIL import Image

# クラスIDからクラス名へのマッピング
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class Infer:
    def __init__(self, config, ckpt_path, infer_size=[640, 640], device='cuda', engine_type='torch'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.infer_size = infer_size
        self.engine_type = engine_type  # エンジンタイプをインスタンス変数として追加
        self.model = self._build_model(ckpt_path)

    def _build_model(self, ckpt_path):
        print(f'Building model with {self.engine_type} engine...')
        if self.engine_type == 'torch':
            model = build_local_model(self.config, ckpt=ckpt_path, device=self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model'], strict=True)
            model.eval()
        elif self.engine_type == 'onnx':
            raise NotImplementedError("ONNX engine is not implemented in this example.")
        elif self.engine_type == 'tensorRT':
            raise NotImplementedError("TensorRT engine is not implemented in this example.")
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")
        return model

    def preprocess(self, image_path):
        origin_img = np.asarray(Image.open(image_path).convert('RGB'))
        img = transform_img(origin_img, 0, **self.config.test.augment.transform, infer_size=self.infer_size)
        img = img.tensors.to(self.device)
        return img, origin_img.shape[:2]

    def postprocess(self, preds, origin_shape):
        if self.engine_type == 'torch':
            output = preds
        elif self.engine_type == 'onnx':
            scores = torch.Tensor(preds[0])
            bboxes = torch.Tensor(preds[1])
            output = postprocess(scores, bboxes,
                                 self.config.model.head.num_classes,
                                 self.config.model.head.nms_conf_thre,
                                 self.config.model.head.nms_iou_thre)
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")

        if len(output) > 0:
            output = output[0].resize(origin_shape)
            bboxes = output.bbox.cpu().numpy()
            scores = output.get_field('scores').cpu().numpy()
            cls_inds = output.get_field('labels').cpu().numpy()
        else:
            bboxes, scores, cls_inds = [], [], []

        return bboxes, scores, cls_inds

    def forward(self, image_path):
        with torch.no_grad():  # 推論時に勾配追跡を無効化
            img, origin_shape = self.preprocess(image_path)
            preds = self.model(img)
            return self.postprocess(preds, origin_shape)


# モデルと設定のパス
config_file = "configs/damoyolo_tinynasL20_T.py"
ckpt_path = "pretrained_models/damoyolo_tinynasL20_T_420.pth"

# 設定の読み込み
config = parse_config(config_file)
infer = Infer(config, ckpt_path)

# 画像の推論
# image_path = "assets/dog.jpg"
image_path = "assets/input.png"
bboxes, scores, cls_inds = infer.forward(image_path)

# 出力結果をわかりやすく表示（スコアが0.5以上のもののみ出力）
print("Detected Objects (Score >= 0.5):")
for bbox, score, cls_ind in zip(bboxes, scores, cls_inds):
    if score >= 0.5:  # スコアが0.5以上の場合のみ表示
        class_name = COCO_CLASSES[int(cls_ind)]  # クラス名を取得
        print(f"Object: {class_name}, Score: {score:.2f}, Bounding Box: {bbox}")
