import cv2
import torch
from damo.detectors.detector import build_local_model
from damo.utils import vis, postprocess
from damo.config.base import parse_config

def main():
    # 動画ファイルのパス
    video_path = 'assets/input_1.mp4'  # 入力するMP4ファイルのパス
    # DAMO-YOLOの設定ファイルと学習済みモデルのパス
    config_file = 'configs/damoyolo_tinynasL20_T.py'
    ckpt_path = 'pretrained_models/damoyolo_tinynasL20_T_420.pth'
    # 信頼度の閾値
    conf_threshold = 0.5
    # 推論サイズ
    infer_size = (640, 640)
    # デバイスの設定（CPUまたはGPU）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # モデルの構築
    cfg = parse_config(config_file)
    model = build_local_model(cfg, ckpt_path, device=device)
    model.eval()

    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} を開けませんでした。")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 画像の前処理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, infer_size)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_tensor /= 255.0

        # 推論
        with torch.no_grad():
            outputs = model(img_tensor)

        # 推論結果の後処理
        # `outputs`から`cls_scores`と`bbox_preds`を抽出
        cls_scores = outputs[0].extra_fields['scores']  # クラスのスコア（信頼度）
        bbox_preds = outputs[0].bbox  # バウンディングボックスの座標

        # `postprocess`関数の呼び出し
        outputs = postprocess(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            num_classes=cfg.model.head['num_classes'],
            conf_thre=conf_threshold
        )

        # 検出結果の描画
        if len(outputs) is not 0:
            bboxes = outputs[0][:, :4]
            scores = outputs[0][:, 4]
            cls_inds = outputs[0][:, 5]
            frame = vis(frame, bboxes, scores, cls_inds, conf=conf_threshold, class_names=cfg.dataset.class_names)

        # フレームの表示
        cv2.imshow('DAMO-YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
