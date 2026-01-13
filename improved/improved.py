#light_system
import os
import re
import cv2
import json
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1
import easyocr
import matplotlib.pyplot as plt

# =========================
# 0) 설정
# =========================
# — ArcFace 임계값: 원본 임베딩/보정 임베딩 분리 (Patch A)
THRESHOLD      = 0.40  # 원본 임베딩 기준
THRESHOLD_ENH  = 0.43  # 보정 임베딩(폴백)일 때 조금 더 보수적으로

# — 전역 밝기 판정(보수적) + 보정 강도 (Patch B)
VIDEO_Y_MEAN_THR = 90     # 95→90: 진짜 어두울 때만 보정 켜기
ROI_BORDER       = 0.10   # 중앙 80% 영역만 측정(레터박스 무시)
FORCE_GLOBAL_ENH = None   # None=자동, True=항상 켜기, False=항상 끄기
GLOBAL_TARGET_Y  = 130    # 150→130: 과노출/노이즈 확대 방지

# — 어두울 때만 조건부 추가 개선 (기본 OFF, 필요 시 켜서 A/B 테스트)
USE_WB                 = False   # Gray-World 화이트밸런스
USE_ADAPTIVE_SHARPEN   = True   # 저선명 프레임 샤픈
USE_HIGHLIGHT_COMPRESS = False   # 하이라이트 압축

SHARP_THR   = 120.0  # Laplacian Var 임계값(100~160)
HI_CLIP_THR = 0.02   # Y>=250 비율이 2%↑면 하이라이트 압축

# — 등록자 평균 임베딩
centroid = np.load(r'C:\Users\User\Desktop\average_centroid_deep_agument.npy')

# 비디오/GT 경로 (ref는 GT 만든 기준 영상)
video_path       = r'C:\Users\User\Desktop\your_video.mp4'   # 테스트 입력
ref_video_path   = r'C:\Users\User\Desktop\your_video.mp4'   # GT 기준
save_video_path  = r'C:\Users\User\Desktop\output_mosaic.mp4'
labels_json_dir  = r'C:\Users\User\Desktop\labels_json'      # ✅ 수동 GT(JSON)

# 산출물 폴더
base_dir          = r'C:\Users\User\Desktop'
before_frame_dir  = os.path.join(base_dir, 'before_blur_frames')
after_frame_dir   = os.path.join(base_dir, 'after_blur_frames')
before_face_dir   = os.path.join(base_dir, 'before_blur_faces')
after_face_dir    = os.path.join(base_dir, 'after_blur_faces')
auto_pred_dir     = os.path.join(base_dir, 'auto_pred_json')     # ✅ 자동 예측 JSON
enh_video_path    = r'C:\Users\User\Desktop\det_input_video.mp4' # 검출/인식 입력(보정본) 미리보기
det_frame_dir     = os.path.join(base_dir, 'det_input_frames')   # (옵션) 프레임 저장

for d in [before_frame_dir, after_frame_dir, before_face_dir, after_face_dir, auto_pred_dir, det_frame_dir]:
    os.makedirs(d, exist_ok=True)

# =========================
# 1) 모델 로드
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo   = YOLO(r'C:\Users\User\Desktop\yolov8n-seg.pt')  # seg 권장 (person=0)
mtcnn  = MTCNN(image_size=224, margin=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
ocr    = easyocr.Reader(['ko', 'en'])

# =========================
# 2) 유틸
# =========================
def mosaic_mask(img, mask, scale=0.05):
    """YOLO seg mask(단일 인스턴스)로 모자이크"""
    h, w = img.shape[:2]
    resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small  = cv2.resize(img, (small_w, small_h))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_3d = (resized_mask > 0.5).astype(np.uint8)[:, :, None]
    img[:] = img * (1 - mask_3d) + mosaic * mask_3d

def blur_polygon(image, polygon, ksize=(51, 51)):
    pts = np.array(polygon, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    blurred = cv2.GaussianBlur(image, ksize, 0)
    mask_3ch = cv2.merge([mask] * 3)
    return np.where(mask_3ch == 255, blurred, image)

def is_sensitive_text(text):
    patterns = [
        r'\d{6}-\d{7}', r'01[0-9]-\d{3,4}-\d{4}',
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        r'\d{1,4}동|\d{1,4}호', r'[\uac00-\ud7a3]+[시군구동읍면로길]',
        r'[\uac00-\ud7a3]{2,20}(아파트|빌라|주택|맨션|오피스텔|연립)',
        r'\d{1,4}-\d{1,4}', r'\d{2,4}-\d{2,4}-\d{4,7}', r'\d{9,14}',
        r'(대학교|중학교|고등학교|회사|직장|소속)',
        r'(이름|성명)[:：]?\s?[가-힣]{2,4}'
    ]
    s = text.replace(" ", "")
    return any(re.search(p, s) for p in patterns)

def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def load_frame_json(frame_idx):
    jf = os.path.join(labels_json_dir, f"frame_{frame_idx:03d}.json")
    if not os.path.exists(jf):
        return None
    with open(jf, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("objects", [])

def center_in_bbox(cx, cy, bbox):
    x1, y1, x2, y2 = bbox
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

# ----- ROI/지표 측정 -----
def central_roi(bgr, border=ROI_BORDER):
    h, w = bgr.shape[:2]
    y0 = int(h * border); y1 = h - y0
    x0 = int(w * border); x1 = w - x0
    return bgr[y0:y1, x0:x1]

def measure_metrics(bgr):
    roi = central_roi(bgr)
    y = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    y_mean = float(y.mean())
    hi_clip = float((y >= 250).mean())
    sharp = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    means = roi.mean(axis=(0,1))  # BGR
    b,g,r = means
    rg = float(r/(g+1e-6)); bg = float(b/(g+1e-6))
    return y_mean, hi_clip, sharp, rg, bg

# ----- 개선 함수들 -----
def gray_world_wb(bgr):
    b,g,r = [bgr[:,:,i].astype(np.float32) for i in range(3)]
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    kb, kg, kr = mean_gray/(mean_b+1e-6), mean_gray/(mean_g+1e-6), mean_gray/(mean_r+1e-6)
    out = bgr.copy().astype(np.float32)
    out[:,:,0] = np.clip(b*kb, 0, 255)
    out[:,:,1] = np.clip(g*kg, 0, 255)
    out[:,:,2] = np.clip(r*kr, 0, 255)
    return out.astype(np.uint8)

def unsharp_mask(bgr, sigma=1.0, amount=0.6, thresh=0):
    blur = cv2.GaussianBlur(bgr, (0,0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1+amount, blur, -amount, 0)
    if thresh > 0:
        low_contrast_mask = np.abs(bgr.astype(np.int16)-blur.astype(np.int16)).max(axis=2) < thresh
        sharp[low_contrast_mask] = bgr[low_contrast_mask]
    return sharp

def highlight_compress(bgr):
    x = bgr.astype(np.float32)/255.0
    knee = 0.85; roll = 0.10
    y = np.where(x < knee, x, knee + (1 - np.exp(-(x-knee)/roll)) * (1-knee))
    return np.clip(y*255.0, 0, 255).astype(np.uint8)

# ----- 전역 보정 (감마+CLAHE) -----
def global_enhance(bgr, target_y=GLOBAL_TARGET_Y):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y0 = ycrcb[:, :, 0].astype(np.uint8)
    m = max(1.0, float(y0.mean()))
    gamma = np.log(target_y / 255.0) / np.log(m / 255.0)   # gamma<1 밝아짐
    gamma = float(np.clip(gamma, 0.6, 1.6))
    table = np.array([(i/255.0)**gamma * 255 for i in range(256)], dtype=np.uint8)
    y1 = cv2.LUT(y0, table)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y2 = clahe.apply(y1)
    ycrcb[:, :, 0] = y2
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# ----- 검출/인식 입력용 파이프라인 (GLOBAL_ENH=True 때만 추가 개선) -----
def build_det_input(bgr, global_enh_flag):
    decisions = {"wb": False, "enh": False, "sharpen": False, "hicomp": False}
    out = bgr

    if global_enh_flag:
        if USE_WB:
            out = gray_world_wb(out); decisions["wb"] = True
        out = global_enhance(out, target_y=GLOBAL_TARGET_Y); decisions["enh"] = True

        _, _, sharp, _, _ = measure_metrics(out)
        if USE_ADAPTIVE_SHARPEN and sharp < SHARP_THR:
            out = unsharp_mask(out, sigma=1.0, amount=0.6); decisions["sharpen"] = True

        _, hi_clip, _, _, _ = measure_metrics(out)
        if USE_HIGHLIGHT_COMPRESS and hi_clip > HI_CLIP_THR:
            out = highlight_compress(out); decisions["hicomp"] = True

    # 디버그 문자열
    y_in, hi_in, sharp_in, rg_in, bg_in   = measure_metrics(bgr)
    y_out, hi_out, sharp_out, rg_out, bg_out = measure_metrics(out)
    overlay = (f"GLOBAL_ENH:{global_enh_flag}  WB:{decisions['wb']} ENH:{decisions['enh']} "
               f"SH:{decisions['sharpen']} HI:{decisions['hicomp']}  "
               f"Y_in:{y_in:.1f}/out:{y_out:.1f}  S:{sharp_out:.0f}  R/G:{rg_out:.2f} B/G:{bg_out:.2f}")
    return out, overlay, decisions

# =========================
# 3) GT 기준 해상도 읽기
# =========================
_ref_cap = cv2.VideoCapture(ref_video_path)
REF_W = int(_ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
REF_H = int(_ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
_ref_cap.release()
if REF_W == 0 or REF_H == 0:
    raise RuntimeError("GT 기준(ref) 영상의 해상도를 읽을 수 없습니다. ref_video_path 확인!")

def map_point_to_ref(cx, cy, cur_w, cur_h, ref_w, ref_h):
    rx = int(round(cx * (ref_w / max(1, cur_w))))
    ry = int(round(cy * (ref_h / max(1, cur_h))))
    return rx, ry

def map_bbox_to_ref(b, cur_w, cur_h, ref_w, ref_h):
    x1, y1, x2, y2 = b
    sx = ref_w / max(1, cur_w); sy = ref_h / max(1, cur_h)
    return [int(round(x1 * sx)), int(round(y1 * sy)),
            int(round(x2 * sx)), int(round(y2 * sy))]

print(f"[GT 기준 해상도] {REF_W} x {REF_H}")

# =========================
# 4) 영상 밝기 사전 점검
# =========================
def video_brightness_probe(path, sample_frames=60, y_thr=VIDEO_Y_MEAN_THR):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(0, total-1), num=min(sample_frames, max(1, total)), dtype=int)

    ys, p75s, hi_clips = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret: continue
        roi = central_roi(frame)
        y = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        ys.append(float(np.mean(y)))
        p75s.append(float(np.percentile(y, 75)))
        hi_clips.append(float((y >= 250).mean()))
    cap.release()

    if not ys:
        return {"level": "unknown", "y_mean": None}

    y_mean   = float(np.mean(ys))
    y_median = float(np.median(ys))
    y_p75    = float(np.median(p75s))
    hi       = float(np.mean(hi_clips))

    is_dark = (y_median < y_thr and y_p75 < (y_thr + 30))
    if hi > 0.03:  # 과노출 많으면 dark로 보지 않음
        is_dark = False

    print(f"[ProbeDbg] mean={y_mean:.1f} med={y_median:.1f} p75={y_p75:.1f} hi={hi:.3f}")
    return {"level": "dark" if is_dark else "normal", "y_mean": y_mean}

probe = video_brightness_probe(video_path, sample_frames=60)
GLOBAL_ENH = (probe["level"] == "dark")
if FORCE_GLOBAL_ENH is True:
    GLOBAL_ENH = True
elif FORCE_GLOBAL_ENH is False:
    GLOBAL_ENH = False
print(f"[Video Brightness] level={probe['level']}  y_mean={probe['y_mean']}  -> GLOBAL_ENH={GLOBAL_ENH}")

# =========================
# 5) 비디오 IO
# =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("❌ 영상 파일 열기 실패")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25
fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
out         = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
det_writer  = cv2.VideoWriter(enh_video_path, fourcc, fps, (width, height))  # 검출/인식 입력(미리보기)

# =========================
# 6) 추론 루프 + 평가 + 자동 GT JSON
# =========================
frame_idx = 0

# 사람 단위 혼동행렬
per_tp = per_fp = per_tn = per_fn = 0
per_total = 0
per_missed = 0
per_extra  = 0

def count_person(res):
    if res.boxes is None: return 0
    cls = res.boxes.cls.cpu().numpy().astype(int)
    return int((cls == 0).sum())

print("▶ 영상 기반 모자이크 시작...")

while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imwrite(os.path.join(before_frame_dir, f"frame_{frame_idx:03d}.jpg"), frame)
    orig = frame.copy()

    # (1) 검출/인식 입력 프레임 만들기
    frame_det, overlay_text, decisions = build_det_input(frame, GLOBAL_ENH)

    # 미리보기(오버레이)
    frame_det_vis = frame_det.copy()
    cv2.rectangle(frame_det_vis, (8, 8), (8+1000, 8+32), (0,0,0), -1)
    cv2.putText(frame_det_vis, overlay_text, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    det_writer.write(frame_det_vis)
    # cv2.imwrite(os.path.join(det_frame_dir, f"frame_{frame_idx:03d}.jpg"), frame_det_vis)

    person_idx = 0
    pred_objs_this_frame = []
    gt_objs = load_frame_json(frame_idx)
    gt_matched = [False] * (len(gt_objs) if gt_objs else 0)

    # (2) YOLO: 보정 입력 먼저, 실패 시 원본 재시도 (Patch C)
    results = yolo(frame_det, conf=0.5, iou=0.45)[0]
    if GLOBAL_ENH and count_person(results) == 0:
        # 원본으로 폴백
        results_orig = yolo(orig, conf=0.5, iou=0.45)[0]
        if count_person(results_orig) > 0:
            results   = results_orig
            frame_det = orig  # 이후 크롭/임베딩은 이 프레임 기준으로

    masks   = results.masks.data.cpu().numpy() if (results.masks is not None and results.masks.data is not None) else None
    classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
    boxes   = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []

    num_masks = len(masks) if masks is not None else 0
    cur_h, cur_w = frame.shape[:2]

    for idx, (cls, box) in enumerate(zip(classes, boxes)):
        if cls != 0:  # 사람만
            continue

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(cur_w, x2), min(cur_h, y2)
        if x2 <= x1 or y2 <= y1:
            person_idx += 1
            continue

        # (3) ArcFace: 원본 먼저, 실패 시 보정본 폴백 (Patch A)
        crop_orig = orig[y1:y2, x1:x2]
        crop_det  = frame_det[y1:y2, x1:x2]

        face_name = f"frame_{frame_idx:03d}_p{person_idx}.jpg"
        cv2.imwrite(os.path.join(before_face_dir, face_name), crop_orig)

        need_mosaic = True
        sim_val = None
        try:
            # 원본 먼저
            pil_orig = Image.fromarray(cv2.cvtColor(crop_orig, cv2.COLOR_BGR2RGB))
            face = mtcnn(pil_orig)
            used_enh_for_embed = False

            # 원본 실패 + 전역보정 켜짐 → 보정본에서 재시도
            if face is None and GLOBAL_ENH:
                pil_det = Image.fromarray(cv2.cvtColor(crop_det, cv2.COLOR_BGR2RGB))
                face = mtcnn(pil_det)
                used_enh_for_embed = face is not None

            if face is not None:
                with torch.no_grad():
                    emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
                sim_val = cos_sim(centroid, emb)
                thr = THRESHOLD_ENH if used_enh_for_embed else THRESHOLD
                need_mosaic = (sim_val < thr)
            else:
                need_mosaic = True
        except Exception:
            need_mosaic = True
            sim_val = None

        # (4) 모자이크 적용 (원본 frame에)
        if need_mosaic:
            if masks is not None and idx < num_masks:
                mosaic_mask(frame, masks[idx])
            else:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    small = cv2.resize(roi, (max(1, (x2-x1)//10), max(1, (y2-y1)//10)))
                    roi_m = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = roi_m

        # 저장
        crop_after = frame[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(after_face_dir, face_name), crop_after)

        # 자동 예측 JSON 기록
        bbox_cur = [int(x1), int(y1), int(x2), int(y2)]
        bbox_ref = map_bbox_to_ref(bbox_cur, cur_w, cur_h, REF_W, REF_H)
        pred_objs_this_frame.append({
            "bbox": bbox_cur,
            "bbox_ref": bbox_ref,
            "label": int(1 if need_mosaic else 0),   # 0=등록자, 1=비등록자
            "sim": None if sim_val is None else float(sim_val),
            "used_global_enh": bool(GLOBAL_ENH),
            "used_wb": bool(GLOBAL_ENH and USE_WB),
            "used_sharpen": False,  # decisions는 frame_det 기준이라 단일 플래그로 기록
            "used_hicomp":  False
        })

        # (5) 사람 단위 평가(중심점 매칭; 좌표 ref로 스케일)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx_ref, cy_ref = map_point_to_ref(cx, cy, cur_w, cur_h, REF_W, REF_H)

        matched = False
        if gt_objs:
            for gi, g in enumerate(gt_objs):
                gb = g.get("bbox"); gl = g.get("label")
                if gb is None or gl is None or gt_matched[gi]:
                    continue
                if center_in_bbox(cx_ref, cy_ref, gb):
                    matched = True
                    gt_matched[gi] = True
                    gt_label  = int(gl)
                    pred_label = 1 if need_mosaic else 0
                    if gt_label == 1 and pred_label == 1: per_tp += 1
                    elif gt_label == 1 and pred_label == 0: per_fn += 1
                    elif gt_label == 0 and pred_label == 1: per_fp += 1
                    else: per_tn += 1
                    per_total += 1
                    break
        if not matched:
            per_extra += 1

        person_idx += 1

    # 자동 예측 JSON 저장
    with open(os.path.join(auto_pred_dir, f"frame_{frame_idx:03d}.json"), "w", encoding="utf-8") as f:
        json.dump({"image": f"frame_{frame_idx:03d}.jpg", "objects": pred_objs_this_frame},
                  f, ensure_ascii=False, indent=2)

    # 매칭되지 않은 GT(검출 실패)
    if gt_objs:
        for gi, _ in enumerate(gt_objs):
            if not gt_matched[gi]:
                per_missed += 1

    # 민감 텍스트 블러: 인식은 frame_det에서, 적용은 원본 frame에
    try:
        texts = ocr.readtext(frame_det)
        for (bbox, text, conf) in texts:
            if is_sensitive_text(text):
                frame = blur_polygon(frame, bbox)
    except Exception:
        pass

    # 프레임 저장 + 비디오 기록
    cv2.imwrite(os.path.join(after_frame_dir, f"frame_{frame_idx:03d}.jpg"), frame)
    out.write(frame)

    frame_idx += 1

cap.release()
out.release()
det_writer.release()

# =========================
# 7) 사람 단위 성능 평가 (JSON GT 기준)
# =========================
print("\n✅ [사람 단위 평가 - 수동 JSON GT 기준]")
print(f" - 매칭 성공 건수: {per_total}")
print(f" - GT 있었는데 매칭 실패(검출X): {per_missed}")
print(f" - 검출은 있었는데 GT 없음/매칭 실패: {per_extra}")

if per_total > 0:
    acc = (per_tp + per_tn) / per_total
    precision = per_tp / (per_tp + per_fp) if (per_tp + per_fp) > 0 else 0.0  # 양성=비등록자
    recall    = per_tp / (per_tp + per_fn) if (per_tp + per_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f" - TP(비등록자 모자이크 성공): {per_tp}")
    print(f" - FP(등록자 오모자이크): {per_fp}")
    print(f" - FN(비등록자 놓침): {per_fn}")
    print(f" - TN(등록자 통과 성공): {per_tn}")
    print(f" - Accuracy : {acc:.3f}")
    print(f" - Precision: {precision:.3f}")
    print(f" - Recall   : {recall:.3f}")
    print(f" - F1-score : {f1:.3f}")
else:
    print(" - 매칭된 샘플이 없어 사람 단위 평가는 생략됩니다.")

print(f"\n🎯 자동 예측 JSON 폴더: {auto_pred_dir}")
print(f"🎬 결과 비디오(원본에 모자이크 적용): {save_video_path}")
print(f"🎬 검출/인식 입력 영상(보정 미리보기): {enh_video_path}")
print(f"🖼  전/후 프레임: {before_frame_dir} | {after_frame_dir}")
print(f"🙂  전/후 얼굴 크롭: {before_face_dir} | {after_face_dir}")
print(f"🖼  (옵션) 검출/인식 입력 프레임 폴더: {det_frame_dir}")

# =========================
# 8) ArcFace 유사도 분포 (참고용)
# =========================
def get_similarity(pil_img):
    face = mtcnn(pil_img)
    if face is None:
        return None
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device))[0].cpu().numpy()
    return cos_sim(centroid, emb)

before_sims, after_sims = [], []
face_files = sorted([f for f in os.listdir(before_face_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

print(f"\n▶ ArcFace 유사도 분석(얼굴 크롭) 중... 총 {len(face_files)}개")
for file in face_files:
    try:
        img_b = Image.open(os.path.join(before_face_dir, file)).convert('RGB')
        img_a = Image.open(os.path.join(after_face_dir,  file)).convert('RGB')
        sb = get_similarity(img_b)
        sa = get_similarity(img_a)
        if sb is not None: before_sims.append(sb)
        if sa is not None: after_sims.append(sa)
    except Exception:
        continue

before_sims = np.array(before_sims, dtype=float)
after_sims  = np.array(after_sims, dtype=float)

if len(before_sims) > 0 and len(after_sims) > 0:
    plt.hist(before_sims, bins=30, alpha=0.5, label='Before Blur')
    plt.hist(after_sims,  bins=30, alpha=0.5, label='After Blur')
    plt.axvline(THRESHOLD, color='red', linestyle='--', label=f'Threshold ({THRESHOLD})')
    plt.xlabel('ArcFace Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('ArcFace Similarity Distribution (Face Crops)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n✅ [ArcFace 유사도 통계] (참고용)")
    print(f" - 평균 유사도 (전) : {before_sims.mean():.4f}")
    print(f" - 평균 유사도 (후) : {after_sims.mean():.4f}")
    print(f" - 평균 감소량     : {(before_sims.mean() - after_sims.mean()):.4f}")
else:
    print("\nℹ️ 유사도 분포를 계산할 충분한 얼굴 크롭이 없습니다.")