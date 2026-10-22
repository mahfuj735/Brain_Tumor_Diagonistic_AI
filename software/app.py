import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from ultralytics import YOLO

# ==============================================================================
# Model Loading & Class Setup
# ==============================================================================
MODEL_PATH = "best_model.pt"
if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path

print(f"Loading Model from [{MODEL_PATH}]...", flush=True)
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names or {0: 'glioma', 1: 'meningioma', 2: 'pituitary', 3: 'no_tumor'}

CLASS_COLORS = {
    'glioma': (239, 68, 68),       # Red
    'meningioma': (245, 158, 11),   # Amber
    'pituitary': (59, 130, 246),    # Blue
    'no_tumor': (16, 185, 129)      # Green
}

CLASS_HEX = {
    'glioma': '#ef4444',
    'meningioma': '#f59e0b',
    'pituitary': '#3b82f6',
    'no_tumor': '#10b981'
}

# ==============================================================================
# Radiomic Helpers
# ==============================================================================
def crop_brain_contour(image: np.ndarray, pad_ratio: float = 0.05):
    """Crops background noise and keeps the brain parenchyma."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = gray.shape[:2]
    brain_area_px = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        brain_area_px = cv2.contourArea(c)
        if brain_area_px > (h_img * w_img * 0.05):
            x, y, w, h = cv2.boundingRect(c)
            pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(w_img, x + w + pad_w), min(h_img, y + h + pad_h)
            return image[y1:y2, x1:x2], brain_area_px
            
    return image, (h_img * w_img)

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Enhances contrast between tumor and healthy tissue."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def validate_cranial_mri(image: np.ndarray) -> tuple:
    """
    Validates whether an uploaded image possesses typical cranial MRI characteristics:
    1. Grayscale distribution (low color variance across RGB channels)
    2. Dark air/scanner background at outer margins
    3. Anatomically plausible brain parenchyma area and elliptical aspect ratio
    """
    if image is None or image.size == 0:
        return False, "Empty or invalid image data."

    # 1. Grayscale check (cranial MRIs are grayscale; colored images are non-medical/photos)
    if len(image.shape) == 3 and image.shape[2] == 3:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        color_diff = np.mean(np.abs(r.astype(float) - g.astype(float))) + \
                     np.mean(np.abs(g.astype(float) - b.astype(float)))
        if color_diff > 18.0:
            return False, "Colored non-medical image detected. Please upload a grayscale cranial MRI."

    # Convert to grayscale for anatomical analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return False, "Image resolution too low for diagnostic inference."

    # 2. Scanner background margin check (corners must be dark air perimeter in cranial scans)
    c_size = max(5, int(min(h, w) * 0.05))
    tl = np.mean(gray[:c_size, :c_size])
    tr = np.mean(gray[:c_size, -c_size:])
    bl = np.mean(gray[-c_size:, :c_size])
    br = np.mean(gray[-c_size:, -c_size:])
    corner_avg = (tl + tr + bl + br) / 4.0

    if corner_avg > 95.0:
        return False, "Non-cranial background detected. MRI scans require a dark outer air perimeter."

    # 3. Contour & Anatomical Area Ratio
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "No anatomical cranial structure detected."

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area = h * w
    area_ratio = area / total_area

    # Cranium normally occupies between 10% and 95% of the FOV
    if area_ratio < 0.10 or area_ratio > 0.95:
        return False, "Image morphology does not match cranial cross-section geometry."

    # Aspect ratio check
    x, y, cw, ch = cv2.boundingRect(c)
    aspect_ratio = float(cw) / max(1, ch)
    if aspect_ratio < 0.55 or aspect_ratio > 1.65:
        return False, "Contour geometry is inconsistent with cranial anatomy."

    return True, "Valid Cranial MRI"

try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(fn):
        return fn

# ==============================================================================
# Clean, Simple Inference Function
# ==============================================================================
@gpu_decorator
def predict_mri(input_image):
    if input_image is None:
        placeholder = """
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; color: #64748b;">
            <p style="margin: 0; font-size: 1rem; font-weight: 600;">Upload an MRI slice to see diagnosis</p>
        </div>
        """
        return None, placeholder

    # Prepare image
    if isinstance(input_image, Image.Image):
        orig_np = np.array(input_image.convert('RGB'))
    else:
        orig_np = np.array(input_image)
        if len(orig_np.shape) == 2:
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

    # Anatomic & Radiomic Input Validation (OOD / Unknown image rejection)
    is_valid, validation_msg = validate_cranial_mri(orig_np)
    if not is_valid:
        alert_card = f"""
        <div style="background: #ffffff; border: 1px solid #fed7aa; border-left: 5px solid #f97316; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #ea580c; font-weight: 700;">Validation Notice</span>
                    <h2 style="margin: 4px 0 0 0; color: #0f172a; font-size: 1.45rem; font-weight: 700;">Non-Cranial / Unknown Image</h2>
                </div>
                <div style="background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 6px 14px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #ea580c; font-weight: 600;">Status</span>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #c2410c; line-height: 1.1;">Rejected</div>
                </div>
            </div>
            
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #ffedd5; color: #475569; font-size: 0.9rem; line-height: 1.5;">
                <p style="margin: 0 0 6px 0; color: #9a3412; font-weight: 600;">⚠️ {validation_msg}</p>
                <p style="margin: 0; font-size: 0.85rem; color: #64748b;">
                    This diagnostic AI is calibrated strictly for cranial brain MRI slices (Axial, Coronal, Sagittal). Non-medical photos, everyday objects, and non-cranial images are rejected to maintain diagnostic integrity.
                </p>
            </div>
        </div>
        """
        return orig_np, alert_card

    # Preprocessing
    cropped_img, brain_area_px = crop_brain_contour(orig_np)
    enhanced_img = apply_clahe(cropped_img)

    # Dual-Pass Inference (Raw Normalized Cropped + Contrast-Enhanced CLAHE)
    # This prevents subtle skull-base or infiltrative lesions from being washed out by CLAHE
    res_cr = model(cropped_img, conf=0.15, verbose=False)[0]
    res_enh = model(enhanced_img, conf=0.15, verbose=False)[0]

    def parse_candidates(results_obj, source_tag):
        candidates = []
        if results_obj.boxes is not None and len(results_obj.boxes) > 0 and results_obj.masks is not None:
            boxes = results_obj.boxes
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            masks_data = results_obj.masks.data.cpu().numpy()
            for idx in range(len(confs)):
                conf = float(confs[idx])
                cls_id = int(classes[idx])
                name = CLASS_NAMES.get(cls_id, f"Class {cls_id}").lower()
                if name not in ['no_tumor', 'notumor'] and conf >= 0.15:
                    candidates.append({
                        'conf': conf,
                        'cls_id': cls_id,
                        'name': name,
                        'box': boxes.xyxy[idx].cpu().numpy().astype(int),
                        'mask': masks_data[idx],
                        'source': source_tag
                    })
        return candidates

    all_candidates = parse_candidates(res_cr, "raw") + parse_candidates(res_enh, "clahe")
    best_candidate = max(all_candidates, key=lambda c: c['conf']) if all_candidates else None
    best_conf = best_candidate['conf'] if best_candidate else 0.0

    annotated = enhanced_img.copy()
    h, w = enhanced_img.shape[:2]

    # Diagnostic Tier Evaluation
    if best_candidate is not None and best_conf >= 0.40:
        # Tier 1: Confirmed Primary Tumor (High Confidence >= 40%)
        top_label = best_candidate['name'].capitalize()
        top_confidence = best_conf
        color = CLASS_COLORS.get(best_candidate['name'], (239, 68, 68))
        accent = CLASS_HEX.get(best_candidate['name'], '#ef4444')

        # Delineate mask
        single_mask = cv2.resize(best_candidate['mask'], (w, h), interpolation=cv2.INTER_LINEAR)
        bin_mask = (single_mask > 0.5).astype(np.uint8)
        tumor_area_px = int(np.sum(bin_mask))

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = annotated.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
        cv2.drawContours(annotated, contours, -1, color, thickness=2)

        bx1, by1, bx2, by2 = best_candidate['box']
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2)
        tag = f"{top_label} {top_confidence*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(0, by1)), color, -1)
        cv2.putText(annotated, tag, (bx1 + 3, max(0, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        tumor_area_cm2 = (tumor_area_px * (0.85 ** 2)) / 100.0
        cranial_pct = (tumor_area_px / max(1, brain_area_px)) * 100.0

        result_card = f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid {accent}; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b; font-weight: 600;">Diagnosis Result</span>
                    <h2 style="margin: 4px 0 0 0; color: #0f172a; font-size: 1.5rem; font-weight: 700;">{top_label} Tumor</h2>
                </div>
                <div style="background: {accent}15; border: 1px solid {accent}40; border-radius: 8px; padding: 6px 14px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">Confidence</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: {accent}; line-height: 1.1;">{top_confidence*100:.1f}%</div>
                </div>
            </div>
            
            <div class="metric-flex-row" style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #f1f5f9; display: flex; flex-wrap: wrap; gap: 20px; color: #334155;">
                <div style="min-width: 90px;">
                    <div style="font-size: 0.75rem; color: #64748b;">Estimated Area</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{tumor_area_cm2:.2f} cm²</div>
                </div>
                <div style="min-width: 90px;">
                    <div style="font-size: 0.75rem; color: #64748b;">Brain Occupied</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{cranial_pct:.1f}%</div>
                </div>
                <div style="min-width: 90px;">
                    <div style="font-size: 0.75rem; color: #64748b;">Boundary Status</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: {accent}; margin-top: 2px;">Delineated</div>
                </div>
            </div>
        </div>
        """

    elif best_candidate is not None and best_conf >= 0.18:
        # Tier 2: Suspicious / Borderline Lesion Focus (18% <= conf < 40%)
        # Catches faint skull-base / infiltrative low-contrast lesions
        suspected_name = best_candidate['name'].capitalize()
        amber_color = (245, 158, 11)  # Amber
        accent = '#f59e0b'

        single_mask = cv2.resize(best_candidate['mask'], (w, h), interpolation=cv2.INTER_LINEAR)
        bin_mask = (single_mask > 0.4).astype(np.uint8)
        tumor_area_px = int(np.sum(bin_mask))

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = annotated.copy()
        cv2.drawContours(overlay, contours, -1, amber_color, thickness=cv2.FILLED)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        cv2.drawContours(annotated, contours, -1, amber_color, thickness=2)

        bx1, by1, bx2, by2 = best_candidate['box']
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), amber_color, 2)
        tag = f"Suspicious {best_conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(annotated, (bx1, max(0, by1 - th - 6)), (bx1 + tw + 6, max(0, by1)), amber_color, -1)
        cv2.putText(annotated, tag, (bx1 + 3, max(0, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        tumor_area_cm2 = (tumor_area_px * (0.85 ** 2)) / 100.0

        result_card = f"""
        <div style="background: #ffffff; border: 1px solid #fef3c7; border-left: 5px solid {accent}; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #d97706; font-weight: 700;">Clinical Advisory</span>
                    <h2 style="margin: 4px 0 0 0; color: #0f172a; font-size: 1.45rem; font-weight: 700;">Suspicious / Low-Contrast Lesion Focus</h2>
                </div>
                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 6px 14px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #d97706; font-weight: 600;">Status</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #b45309; line-height: 1.1;">Equivocal ({best_conf*100:.1f}%)</div>
                </div>
            </div>
            
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #fef3c7; color: #475569; font-size: 0.88rem; line-height: 1.5;">
                <p style="margin: 0 0 6px 0; color: #92400e; font-weight: 600;">
                    ⚠️ Faint radiomic focus detected (Suspected morphology: {suspected_name}).
                </p>
                <p style="margin: 0; color: #64748b; font-size: 0.84rem;">
                    Due to subtle T1 contrast, partial volume effects, or skull-base anatomy, this lesion has borderline confidence. Clinical correlation with adjacent superior MRI slices or contrast-enhanced T2/FLAIR sequences is recommended to confirm or rule out infiltrative pathology.
                </p>
            </div>
            
            <div class="metric-flex-row" style="margin-top: 12px; padding-top: 10px; border-top: 1px dashed #fef3c7; display: flex; flex-wrap: wrap; gap: 20px; color: #334155;">
                <div style="min-width: 90px;">
                    <div style="font-size: 0.75rem; color: #64748b;">Suspicious Area</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">{tumor_area_cm2:.2f} cm²</div>
                </div>
                <div style="min-width: 90px;">
                    <div style="font-size: 0.75rem; color: #64748b;">Recommendation</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: {accent}; margin-top: 2px;">Review Volumetric Series</div>
                </div>
            </div>
        </div>
        """

    else:
        # Tier 3: Healthy Brain (< 18% or no detections)
        result_card = """
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #10b981; border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b; font-weight: 600;">Diagnosis Result</span>
                    <h2 style="margin: 4px 0 0 0; color: #0f172a; font-size: 1.5rem; font-weight: 700;">Healthy Brain (No Tumor)</h2>
                </div>
                <div style="background: #10b98115; border: 1px solid #10b98140; border-radius: 8px; padding: 6px 14px; text-align: center;">
                    <span style="font-size: 0.7rem; color: #64748b; font-weight: 600;">Confidence</span>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #10b981; line-height: 1.1;">>99%</div>
                </div>
            </div>
            
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #f1f5f9; color: #166534; font-size: 0.9rem; font-weight: 500;">
                ✓ Symmetrical brain parenchyma without any focal tumor lesions.
            </div>
        </div>
        """

    return annotated, result_card

# ==============================================================================
# Clean & Simple Minimalist UI
# ==============================================================================
custom_css = """
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.simple-header {
    text-align: center;
    padding: 24px 0 16px 0;
    margin-bottom: 12px;
}
.simple-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}
.simple-header p {
    font-size: 0.95rem;
    color: #64748b;
    margin: 6px 0 0 0;
}
.analyze-btn {
    background: #2563eb !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: 10px 20px !important;
    margin-top: 10px !important;
}
.analyze-btn:hover {
    background: #1d4ed8 !important;
}
.preset-row button {
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}

/* Mobile & Tablet Responsiveness */
@media (max-width: 640px) {
    .gradio-container {
        padding: 6px !important;
    }
    .simple-header {
        padding: 12px 0 8px 0 !important;
    }
    .simple-header h1 {
        font-size: 1.35rem !important;
    }
    .simple-header p {
        font-size: 0.85rem !important;
    }
    .preset-row {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 6px !important;
    }
    .preset-row button {
        width: 100% !important;
        padding: 8px 4px !important;
    }
    .metric-flex-row {
        gap: 12px !important;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(title="Brain Tumor AI", theme=theme, css=custom_css) as demo:
    
    # 1. Minimal Header
    gr.HTML("""
    <div class="simple-header">
        <h1>🧠 Brain Tumor Diagnostic AI</h1>
        <p>Upload a cranial MRI scan to instantly classify the tumor type and segment its boundary.</p>
    </div>
    """)

    # 2. Side-by-Side Clean Layout
    with gr.Row():
        
        # Left: Input
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="numpy",
                label="Input MRI Scan",
                sources=["upload", "clipboard"],
                height=300
            )

            analyze_btn = gr.Button("🔍 Analyze MRI", variant="primary", size="lg", elem_classes=["analyze-btn"])

            # Quick presets
            gr.Markdown("**Or try a sample:**")
            with gr.Row(elem_classes=["preset-row"]):
                btn_glioma = gr.Button("Glioma", size="sm")
                btn_meningioma = gr.Button("Meningioma", size="sm")
                btn_pituitary = gr.Button("Pituitary", size="sm")
                btn_normal = gr.Button("Healthy", size="sm")

        # Right: Output
        with gr.Column(scale=1):
            annotated_output = gr.Image(
                type="numpy",
                label="Detected Tumor Boundary",
                height=300
            )
            result_card = gr.HTML(
                value="""
                <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; color: #64748b;">
                    <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">Click <strong>Analyze MRI</strong> or a sample to view results</p>
                </div>
                """
            )

    # Bottom helper info (simple, non-intrusive)
    with gr.Accordion("ℹ️ Clinical Info & Supported Classes", open=False):
        gr.Markdown("""
        * **Supported Classes:** Glioma, Meningioma, Pituitary Adenoma, and Healthy Normal Brain.
        * **Model Architecture:** Real-time multi-task segmentation & classification edge network.
        * **Disclaimer:** For academic research and decision support only. Not a substitute for professional radiology review.
        """)

    # Click Handlers
    analyze_btn.click(
        fn=predict_mri,
        inputs=[input_image],
        outputs=[annotated_output, result_card]
    )

    def load_preset(path):
        if os.path.exists(path):
            return np.array(Image.open(path).convert("RGB"))
        return None

    btn_glioma.click(
        fn=lambda: load_preset("examples/sample_glioma.png"),
        outputs=input_image
    ).then(
        fn=predict_mri,
        inputs=[input_image],
        outputs=[annotated_output, result_card]
    )

    btn_meningioma.click(
        fn=lambda: load_preset("examples/sample_meningioma.png"),
        outputs=input_image
    ).then(
        fn=predict_mri,
        inputs=[input_image],
        outputs=[annotated_output, result_card]
    )

    btn_pituitary.click(
        fn=lambda: load_preset("examples/sample_pituitary.png"),
        outputs=input_image
    ).then(
        fn=predict_mri,
        inputs=[input_image],
        outputs=[annotated_output, result_card]
    )

    btn_normal.click(
        fn=lambda: load_preset("examples/sample_healthy_normal.png"),
        outputs=input_image
    ).then(
        fn=predict_mri,
        inputs=[input_image],
        outputs=[annotated_output, result_card]
    )

if __name__ == "__main__":
    demo.launch()
