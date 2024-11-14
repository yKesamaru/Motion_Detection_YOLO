import cv2
import numpy as np
import torch
from damo.detectors.detector import build_local_model
from damo.utils import vis, postprocess
from damo.config.base import parse_config

def main():
    # 動画ファイルのパス
    video_path = 'assets/input_1.mp4'
    # DAMO-YOLOの設定ファイルと学習済みモデルのパス
    config_file = 'configs/damoyolo_tinynasL20_T.py'
    ckpt_path = 'pretrained_models/damoyolo_tinynasL20_T_420.pth'
    # 信頼度の閾値
    conf_threshold = 0.3
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
            print(outputs)  # `outputs`の中身全体を確認

            if outputs and hasattr(outputs[0], 'bbox') and outputs[0].bbox.shape[0] > 0:  # バウンディングボックスが存在する場合にのみ処理
                cls_scores = outputs[0].get_field('scores')
                bbox_preds = outputs[0].bbox

                # `postprocess`関数への呼び出し
                outputs = postprocess(
                    cls_scores=cls_scores,
                    bbox_preds=bbox_preds,
                    num_classes=cfg.model.head['num_classes'],
                    conf_thre=conf_threshold
                )
            else:
                print("No detections found in the current frame.")


        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(outputs)  # float型をテンソルに変換

        # 結果の描画
        if outputs is not None and outputs.numel() > 0:  # outputsが空でないか確認
            bboxes = outputs[0][:, :4] if outputs[0].numel() > 0 else None
            if bboxes is not None:
                scores = outputs[0][:, 4]
                cls_inds = outputs[0][:, 5]
                frame = vis(frame, bboxes, scores, cls_inds, conf=conf_threshold, class_names=cfg.dataset.class_names)
        else:
            print("No detections found in the current frame.")


        # フレームの表示
        cv2.imshow('DAMO-YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
