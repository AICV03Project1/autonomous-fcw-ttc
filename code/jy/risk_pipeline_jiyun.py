import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict, deque
from pathlib import Path
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TF




# ======================================================================================
# 1. 모델 정의 및 SORT 알고리즘
# ======================================================================================

def build_backbone_resnet50_fpn():
    try:
        from torchvision.models import ResNet50_Weights
        return resnet_fpn_backbone("resnet50", weights=ResNet50_Weights.DEFAULT)
    except:
        return resnet_fpn_backbone("resnet50", pretrained=True)

class ROIClassifier(nn.Module):
    def __init__(self, num_location=5, num_action=4):
        super().__init__()
        self.transform = GeneralizedRCNNTransform(min_size=640, max_size=1024, 
                                                image_mean=[0.485, 0.456, 0.406], 
                                                image_std=[0.229, 0.224, 0.225])
        self.backbone = build_backbone_resnet50_fpn()
        self.roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.box_head = TwoMLPHead(in_channels=self.backbone.out_channels * 7 * 7, representation_size=1024)
        self.location_head = nn.Linear(1024, num_location)
        self.action_head = nn.Linear(1024, num_action)

    def forward(self, images, targets=None):
        image_list, targets = self.transform(images, targets)
        features = self.backbone(image_list.tensors)
        pooled = self.roi_pooler(features, [t["boxes"] for t in targets], image_list.image_sizes)
        rep = self.box_head(pooled.flatten(start_dim=1))
        return F.softmax(self.location_head(rep), dim=1), torch.sigmoid(self.action_head(rep))

# --- 추적을 위한 SORT Tracker (간략 구현) ---
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) 
          + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return iou

class SortTracker:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.trackers = [] # [x1, y1, x2, y2, id]
        self.count = 0

    def update(self, dets):
        if len(dets) == 0: return np.empty((0, 5))
        if len(self.trackers) == 0:
            new_trks = []
            for d in dets:
                new_trks.append(np.append(d, self.count))
                self.count += 1
            self.trackers = np.array(new_trks)
            return self.trackers

        # 단순 IoU 매칭
        iou_matrix = iou_batch(dets, self.trackers[:, :4])
        matched_indices = []
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))

        # 매칭된 정보 업데이트 및 신규 생성
        new_trackers = []
        matched_det_idx = [m[0] for m in matched_indices]
        for r, c in matched_indices:
            new_trackers.append(np.append(dets[r], self.trackers[c, 4]))
        
        for i in range(len(dets)):
            if i not in matched_det_idx:
                new_trackers.append(np.append(dets[i], self.count))
                self.count += 1
        
        self.trackers = np.array(new_trackers)
        return self.trackers

# ======================================================================================
# 2. 파이프라인 각 단계 Wrapper 클래스
# ======================================================================================

class YOLODetector:
    def __init__(self, weights, conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame_bgr):
        # 1. YOLO 예측
        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        
        if not res.boxes: 
            print("[YOLO] 프레임 내 탐지된 객체 없음")
            return np.empty((0, 4))
        
        # 2. 클래스 정보 및 좌표 추출
        cls = res.boxes.cls.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        
        # 3. 마스크 수정: 차(0), 버스(1)만 남기도록 변경
        # 본인 모델의 클래스 인덱스가 0=Car, 1=Bus라면 [0, 1]로 설정합니다.
        mask = np.isin(cls, [0, 1]) 
        
        det_boxes = xyxy[mask]
        
        # 디버깅 출력
        print(f"[YOLO] 원본 탐지 ID들: {cls}") 
        print(f"[YOLO] 필터링(0,1번 클래스) 후: {len(det_boxes)}개 객체 남음")
        
        return det_boxes

class ROIClassifierWrapper:
    def __init__(self, ckpt_path, device):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = ROIClassifier().to(device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.best_th = ckpt.get("best_micro_th", 0.5)
        print(f"[Classifier] 모델 로드 완료 (Threshold: {self.best_th})")

    @torch.no_grad()
    def predict(self, image_pil, boxes):
        if len(boxes) == 0: return [], []
        
        img_t = TF.to_tensor(image_pil).to(self.device)
        boxes_t = torch.tensor(boxes, dtype=torch.float32).to(self.device)
        
        loc_p, act_p = self.model([img_t], [{"boxes": boxes_t}])
        
        locations = loc_p.argmax(1).cpu().numpy()
        actions = (act_p >= self.best_th).int().cpu().numpy()
        
        print(f"[Classifier] 분석 완료: {len(boxes)}개 객체 (Loc 예시: {locations[:3]}...)")
        return locations, actions

from collections import deque

class IntentAwareRiskManager:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = {}  # {track_id: deque[(cx, cy)]}

    def _update_history(self, track_id, center):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
        self.history[track_id].append(center)

    def _is_long_static(self, centers, move_thresh=3.0, min_static_frames=6):
        if len(centers) < min_static_frames + 1:
            return False
        diffs = [
            np.linalg.norm(np.array(centers[i]) - np.array(centers[i-1]))
            for i in range(1, len(centers))
        ]
        return all(d < move_thresh for d in diffs[-min_static_frames:])

    def _compute_2d_traj_metrics(self, centers, fps, img_w, img_h, eps=1e-3):
        if len(centers) < 2:
            return float('inf'), 0.0, float('inf')

        ego = np.array([img_w / 2, img_h])
        p_prev = np.array(centers[-2])
        p_curr = np.array(centers[-1])

        d_prev = np.linalg.norm(p_prev - ego)
        d_curr = np.linalg.norm(p_curr - ego)
        v_rel = (d_prev - d_curr) * fps  # px/sec

        if v_rel < 5.0: # 접근 안 함
            return float('inf'), v_rel, d_curr
        if self._is_long_static(centers): # 정지 상태
            return float('inf'), v_rel, d_curr

        ttc = d_curr / max(v_rel, eps)
        return ttc, v_rel, d_curr

    def calculate_risk(self, track_id, bbox, location, actions, fps, img_w, img_h):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self._update_history(track_id, (cx, cy))
        centers = self.history[track_id]

        # 1. 물리적 위험도 (TTC)
        ttc, v_rel, d_curr = self._compute_2d_traj_metrics(centers, fps, img_w, img_h)

        base_score = 0.0
        if ttc < 5.0:
            base_score = min(50.0, 50.0 * (3.0 / max(ttc, 0.5)))

        # 2. 의미론적 판단 (차선 및 행동 분석)
        semantic_bonus = 0.0
        omega = 1.0
        reason = "Normal"

        # 무관한 지역 (2,3,4번 차선) 무시
        if location in [2, 3, 4]:
            return {"score": 0.0, "ttc": "Safe", "reason": "Ignored", "level": "SAFE"}

        HIGH_SPEED = 80.0
        # Ego lane (내 차선)
        if location == 0:
            if ttc < 2.0 and v_rel > HIGH_SPEED:
                reason = "Rapid Approach"
                semantic_bonus = 10.0
            elif ttc < 2.0:
                reason = "Close Following"
            else:
                reason = "Normal Deceleration"
        # Adjacent lane (옆 차선 - 끼어들기 감지)
        elif location == 1:
            # actions[1] 또는 [2]가 방향지시등/차선변경 의도라고 가정
            if actions[1] == 1 or actions[2] == 1:
                reason = "Dangerous Cut-in"
                semantic_bonus = 10.0
                omega = 1.3
            else:
                reason = "Approaching"
                omega = 0.8

        final_risk = min(base_score * omega + semantic_bonus, 100.0)
        
        # 로그 출력 (선택 사항)
        if final_risk > 0:
             print(f"[Risk] ID:{track_id} | {reason} | TTC:{ttc:.2f} | Score:{final_risk:.1f}")

        return {
            "score": round(final_risk, 2),
            "ttc": round(ttc, 2) if ttc != float('inf') else "Safe",
            "reason": reason,
            "level": self._get_risk_level(final_risk),
        }

    def _get_risk_level(self, score):
        if score > 50: return "DANGER"
        if score > 30: return "CAUTION"
        return "SAFE"

# ======================================================================================
# 3. 통합 실행 루프
# ======================================================================================

class RiskPipeline:
    def __init__(self, detector, tracker, classifier, predictor):
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.predictor = predictor

    def _get_iou(self, box1, box2):
        """박스 간 IoU 계산 (box: [x1, y1, x2, y2])"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def process_frame(self, frame_bgr, fps, gt_list=None):
        h, w = frame_bgr.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 1. 객체 탐지
        dets = self.detector.detect(frame_bgr)
        if len(dets) == 0: return []
        
        # 2. 트래킹
        tracks = self.tracker.update(dets)
        
        # 3. 분류 (차선 및 행동)
        locs, acts = self.classifier.predict(image_pil, tracks[:, :4])
        
        results = []
        for i, trk in enumerate(tracks):
            bbox, tid = trk[:4], int(trk[4])
            
            # 4. 위험도 계산
            risk_info = self.predictor.calculate_risk(tid, bbox, locs[i], acts[i], fps, w, h)
            
            # 5. [핵심] Ground Truth 매칭 로직
            matched_gt = {"loc": -1, "act": [0,0,0,0], "iou": 0.0}
            if gt_list:
                best_iou = 0
                for gt in gt_list:
                    current_iou = self._get_iou(bbox, gt['bbox'])
                    if current_iou > best_iou and current_iou > 0.3: # IoU 0.3 이상만 매칭
                        best_iou = current_iou
                        matched_gt = {
                            "loc": int(gt['loc']),
                            "act": gt['act'],
                            "iou": round(current_iou, 2)
                        }
            
            # 6. 모든 정보를 결과 딕셔너리에 담기
            results.append({
                "tid": tid, 
                "bbox": bbox, 
                "loc": locs[i], 
                "act": acts[i], 
                "risk": risk_info,
                "gt": matched_gt  # 이 부분이 있어야 KeyError가 발생하지 않습니다.
            })
            
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--yolo-conf", type=float, default=0.4) # 추가
    parser.add_argument("--clf-ckpt", required=True)
    parser.add_argument("--fps", type=int, default=20) # 추가
    parser.add_argument("--output-dir", default="env_test_results")
    args = parser.parse_args()

    RISK_COLOR = {
        "DANGER":  (0, 0, 255),      # Red
        "CAUTION": (0, 255, 255),    # Yellow
        "SAFE":    (255, 0, 0)       # Blue
    }
    LOC_LABEL = {0: "Ego", 1: "Adjacent", 2: "Opposite", 3: "Intersection", 4: "Parking"}
    ACTION_TEXT_COLOR = {"BRAKE": (0, 0, 255), "LEFT": (255, 255, 0), "RIGHT": (0, 255, 255), "EMERGENCY": (255, 0, 255)}

    def get_action_texts(actions):
        names = ["BRAKE", "LEFT", "RIGHT", "EMERGENCY"]
        return [names[i] for i, val in enumerate(actions) if val == 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # RiskPredictor 대신 IntentAwareRiskManager로 생성
    pipeline = RiskPipeline(
        YOLODetector(args.yolo_weights, conf=args.yolo_conf), 
        SortTracker(), 
        ROIClassifierWrapper(args.clf_ckpt, device), 
        IntentAwareRiskManager(window_size=10) 
    )

    env_path = Path(args.env_path)
    img_files = sorted(list(env_path.glob("*.png")))
    if not img_files:
        print(f"폴더에 이미지가 없습니다: {env_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    env_name = env_path.parent.name if env_path.name == "img" else env_path.name
    
    # 비디오 및 CSV 설정
    first_img = cv2.imread(str(img_files[0]))
    h, w = first_img.shape[:2]
    video_writer = cv2.VideoWriter(os.path.join(args.output_dir, f"{env_name}_result.mp4"), 
                                  cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
    f_csv = open(os.path.join(args.output_dir, f"{env_name}_result.csv"), "w")
    f_csv.write("frame,id,loc,gt_loc,loc_match,actions,gt_actions,act_match,iou,risk_level\n")

    print(f"시작: {env_name} (총 {len(img_files)} 프레임)")

    for img_p in img_files:
        frame = cv2.imread(str(img_p))
        
        # GT 파일 경로 (img -> new_txt, .png -> .txt 변환 예시)
        label_p = str(img_p).replace("img", "new_txt").replace(".png", ".txt")
        current_gt = []
        if os.path.exists(label_p):
            with open(label_p, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    # [x1, y1, x2, y2, ?, loc, a1, a2, a3, a4] 형식 가정
                    current_gt.append({
                        'bbox': parts[0:4],
                        'loc': int(parts[5]),
                        'act': parts[6:10]
                    })

        res = pipeline.process_frame(frame, args.fps, gt_list=current_gt)

        for r in res:
            tid, bbox, risk, gt = r['tid'], r['bbox'], r['risk'], r['gt']
            
            # Action 문자열화 로직
            action_names = ["BRAKE", "LEFT", "RIGHT", "EMERGENCY"]
            pred_act_str = "|".join([action_names[i] for i, v in enumerate(r['act']) if v == 1]) or "None"
            gt_act_str = "|".join([action_names[i] for i, v in enumerate(gt['act']) if v == 1]) or "None"
            
            # 일치 여부 판정
            loc_match = 1 if int(r['loc']) == gt['loc'] else 0
            act_match = 1 if pred_act_str == gt_act_str else 0

            # CSV 쓰기
            f_csv.write(f"{img_p.name},{tid},{r['loc']},{gt['loc']},{loc_match},"
                        f"{pred_act_str},{gt_act_str},{act_match},{gt['iou']},{risk['level']}\n")
            
            # 시각화
            x1, y1, x2, y2 = map(int, bbox)
            level = risk["level"]
            color = RISK_COLOR.get(level, (0, 255, 0))
            location = r["loc"]
            actions = r["act"]
            loc_text = LOC_LABEL.get(location, "Unknown")
            action_texts = get_action_texts(actions)

            # 1. Bounding Box (위험도에 따라 굵기 조절)
            thickness = 3 if level == "DANGER" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 2. 텍스트 렌더링
            if level == "SAFE":
                cv2.putText(frame, loc_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_action = y1 - 4
                for act in action_texts:
                    cv2.putText(frame, act, (x1, y_action), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACTION_TEXT_COLOR[act], 1)
                    y_action += 14
            else:
                # 위험 등급 및 점수/TTC
                cv2.putText(frame, f"[{level}] {risk['score']} TTC:{risk['ttc']}s", (x1, y1 - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 위험 사유
                cv2.putText(frame, f"Cause: {risk['reason']}", (x1, y1 - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                # 차선 위치
                cv2.putText(frame, loc_text, (x1, y1 - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                # 행동 상태
                y_action = y1 - 20
                for act in action_texts:
                    cv2.putText(frame, act, (x1, y_action), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACTION_TEXT_COLOR[act], 1)
                    y_action += 14

        video_writer.write(frame)

    video_writer.release()
    f_csv.close()
    print(f"완료! 결과 저장: {args.output_dir}")

if __name__ == "__main__":
    main()
