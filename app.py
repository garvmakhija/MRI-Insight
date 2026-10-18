"""MRI Insight - secure local inference, reports, and reconstruction workspace."""
from __future__ import annotations
import base64, gc, io, json, os, re, sqlite3, tempfile, zipfile, zlib
from datetime import datetime, timezone
from pathlib import Path
import cv2, nibabel as nib, numpy as np, tensorflow as tf
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).parent
MODELS, DATABASE = ROOT / "models", ROOT / "mri_insight.db"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
SEGMENT_CHANNELS = {"NCR": 1, "ED": 2, "ET": 3}
REGION_ORDER = ("NCR", "ED", "ET", "TC", "WT")
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=200 * 1024 * 1024, SECRET_KEY=os.getenv("MRI_INSIGHT_SECRET", "replace-this-local-secret"))
loaded_model = None
loaded_model_kind = None

class ModelError(RuntimeError):
    """A saved model is missing, unreadable, or incompatible at inference time."""

def database():
    connection = sqlite3.connect(DATABASE); connection.row_factory = sqlite3.Row; return connection

def init_database():
    with database() as db: db.executescript("""
    CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, analysis_type TEXT NOT NULL, created_at TEXT NOT NULL, result_json TEXT NOT NULL, FOREIGN KEY(user_id) REFERENCES users(id));
    CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY, analysis_id INTEGER NOT NULL, field_name TEXT NOT NULL, original_name TEXT NOT NULL, content BLOB NOT NULL, FOREIGN KEY(analysis_id) REFERENCES analyses(id));
    CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY, analysis_id INTEGER NOT NULL, created_at TEXT NOT NULL, content BLOB NOT NULL, FOREIGN KEY(analysis_id) REFERENCES analyses(id));
    """)

def user_id():
    if "user_id" not in session: abort(401)
    return session["user_id"]

def model_for(kind):
    global loaded_model, loaded_model_kind
    model_specs = {
        "classification": ("brain_disease_model.keras", "The disease classification model", (None, 224, 224, 3), (None, 4)),
        "segmentation": ("attention_unet.keras", "The segmentation model", (None, 128, 128, 4), (None, 128, 128, 4)),
    }
    filename, label, input_shape, output_shape = model_specs[kind]
    if loaded_model_kind != kind:
        if loaded_model is not None:
            loaded_model = None
            loaded_model_kind = None
            tf.keras.backend.clear_session()
            gc.collect()
        try: loaded_model = tf.keras.models.load_model(MODELS / filename, compile=False)
        except Exception as error: raise ModelError(f"{label} could not be loaded: {error}") from error
        loaded_model_kind = kind
    if loaded_model.input_shape != input_shape or loaded_model.output_shape != output_shape:
        raise ModelError(f"{label} shape is incompatible with this application.")
    return loaded_model

def as_data_url(array):
    buffer = io.BytesIO(); Image.fromarray(array.astype(np.uint8), "RGB").save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def normalize(slice_):
    """Current per-slice min-max normalization; training-time normalization is unknown."""
    slice_ = slice_.astype(np.float32)
    if slice_.max() > slice_.min(): slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min())
    return slice_

def format_measurement(value, decimals):
    """Format numeric metrics like JavaScript's compact number display."""
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

def save_analysis(kind, data, files):
    now = datetime.now(timezone.utc).isoformat()
    with database() as db:
        cursor = db.execute("INSERT INTO analyses(user_id, analysis_type, created_at, result_json) VALUES(?,?,?,?)", (user_id(), kind, now, json.dumps(data)))
        analysis_id = cursor.lastrowid
        for field, file in files.items():
            file.stream.seek(0); content = file.read(); file.stream.seek(0)
            db.execute("INSERT INTO uploads(analysis_id,field_name,original_name,content) VALUES(?,?,?,?)", (analysis_id, field, file.filename, content))
    data["analysis_id"] = analysis_id; return data

MODALITY_PATTERNS = {
    "t1": re.compile(r"(?:^|[_\-.])t1(?:[_\-.]|$)"),
    "t1ce": re.compile(r"(?:^|[_\-.])(?:t1ce|t1c|t1gd)(?:[_\-.]|$)"),
    "t2": re.compile(r"(?:^|[_\-.])t2(?:[_\-.]|$)"),
    "flair": re.compile(r"(?:^|[_\-.])flair(?:[_\-.]|$)"),
}

def modality_from_filename(filename):
    name = Path(filename).name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")): return None
    name = name.removesuffix(".gz").removesuffix(".nii")
    matches = [key for key, pattern in MODALITY_PATTERNS.items() if pattern.search(name)]
    if len(matches) != 1: return None
    return matches[0]

def nifti_from_upload(file, temp_dir):
    if not (file.filename.lower().endswith(".nii") or file.filename.lower().endswith(".nii.gz")): raise ValueError("Each segmentation upload must be a .nii or .nii.gz NIfTI file.")
    suffix = ".nii.gz" if file.filename.lower().endswith(".nii.gz") else ".nii"
    path = Path(temp_dir) / f"{file.name}{suffix}"
    file.save(path)
    try: return nib.load(str(path))
    except Exception as error: raise ValueError(f"{file.filename} is not a valid readable NIfTI file.") from error

def nifti_from_zip(upload, temp_dir):
    if not upload.filename.lower().endswith(".zip"): raise ValueError("ZIP upload must be a .zip file.")
    archive_path = Path(temp_dir) / "study.zip"; upload.save(archive_path)
    discovered = {}
    try: archive_context = zipfile.ZipFile(archive_path)
    except (zipfile.BadZipFile, OSError) as error: raise ValueError("The uploaded ZIP file is invalid or unreadable.") from error
    with archive_context as archive:
        members = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(members) > 50 or sum(entry.file_size for entry in members) > 700 * 1024 * 1024: raise ValueError("The ZIP contains too many or excessively large files.")
        for entry in members:
            modality = modality_from_filename(entry.filename)
            if modality is None: continue
            if modality in discovered: raise ValueError(f"Unable to confidently identify a single {modality.upper()} modality from the uploaded ZIP.")
            suffix = ".nii.gz" if entry.filename.lower().endswith(".nii.gz") else ".nii"
            output = Path(temp_dir) / f"{modality}{suffix}"
            try:
                with archive.open(entry) as source, open(output, "wb") as destination: destination.write(source.read())
                discovered[modality] = nib.load(str(output))
            except Exception as error: raise ValueError(f"The {modality.upper()} file in the ZIP is not a valid readable NIfTI file.") from error
    missing = [key.upper() for key in ("t1", "t1ce", "t2", "flair") if key not in discovered]
    if missing: raise ValueError(f"Unable to identify the following modality from the uploaded ZIP: {', '.join(missing)}.")
    return discovered

def best_segmentation(files, zip_upload=None):
    """Run the actual 2D four-channel Attention U-Net across a BraTS study."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if zip_upload and zip_upload.filename:
            images = nifti_from_zip(zip_upload, temp_dir)
        else:
            if any(not item or not item.filename for item in files.values()): raise ValueError("Please upload all four MRI modalities: T1, T1ce, T2 and FLAIR.")
            images = {key: nifti_from_upload(file, temp_dir) for key, file in files.items()}
        ordered = [images[key] for key in ("t1", "t1ce", "t2", "flair")]
        reference = ordered[0]
        if not np.isfinite(reference.affine).all() or abs(float(np.linalg.det(reference.affine[:3, :3]))) < 1e-8: raise ValueError("The reference NIfTI has an invalid or singular voxel-to-world affine.")
        volumes = []
        for key, image in zip(("T1", "T1ce", "T2", "FLAIR"), ordered):
            if not np.isfinite(image.affine).all() or abs(float(np.linalg.det(image.affine[:3, :3]))) < 1e-8: raise ValueError(f"{key} has an invalid or singular voxel-to-world affine.")
            try: volume = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
            except Exception as error: raise ValueError(f"{key} NIfTI data could not be decoded.") from error
            if not np.isfinite(volume).all(): raise ValueError(f"{key} contains invalid non-finite voxel values.")
            volumes.append(volume)
        if any(volume.ndim != 3 for volume in volumes): raise ValueError("Each NIfTI modality must be a three-dimensional MRI volume.")
        if any(any(length <= 0 for length in volume.shape) for volume in volumes): raise ValueError("Each NIfTI modality must contain non-empty 3D voxel data.")
        if len({volume.shape for volume in volumes}) != 1: raise ValueError("The uploaded NIfTI volumes have incompatible dimensions.")
        spatial_unit = reference.header.get_xyzt_units()[0]
        unit_to_mm = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}.get(spatial_unit)
        if unit_to_mm is None: raise ValueError("The reference NIfTI must declare spatial units (mm, meter, or micron) so measurements can be reported in millimeters.")
        if any(image.header.get_xyzt_units()[0] != spatial_unit for image in ordered[1:]): raise ValueError("All four modalities must use the same NIfTI spatial units.")
        if any(not np.allclose(image.affine, reference.affine, atol=1e-3) for image in ordered[1:]): raise ValueError("All four modalities must belong to the same MRI study and orientation.")
        model = model_for("segmentation")
        predicted_volume = np.zeros(volumes[0].shape, dtype=np.uint8); best = (None, None, -1, None)
        # Keep inference memory bounded while calling the same model and preprocessing for every slice.
        # Small batches reduce peak RAM on low-memory CPU hosts such as Render Free.
        for start in range(0, volumes[0].shape[2], 2):
            slice_inputs = []
            for index in range(start, min(start + 2, volumes[0].shape[2])):
                channels = [cv2.resize(normalize(volume[:, :, index]), (128, 128)) for volume in volumes]
                slice_inputs.append(np.stack(channels, axis=-1))
            try: predictions = model.predict(np.asarray(slice_inputs, dtype=np.float32), verbose=0)
            except Exception as error: raise ModelError(f"Segmentation inference failed: {error}") from error
            for offset, prediction in enumerate(predictions):
                index = start + offset
                mask = np.argmax(prediction, axis=-1).astype(np.uint8)
                restored_mask = cv2.resize(mask, (volumes[0].shape[1], volumes[0].shape[0]), interpolation=cv2.INTER_NEAREST)
                predicted_volume[:, :, index] = restored_mask
                # The selected source slice, restored mask, and area all share this index.
                if (area := int(np.count_nonzero(restored_mask))) > best[2]:
                    best = (normalize(volumes[3][:, :, index]), restored_mask, area, index)
        # NIfTI header zooms are the voxel spacing for array axes i,j,k.
        spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float) * unit_to_mm
        if not np.isfinite(spacing).all() or np.any(spacing <= 0): raise ValueError("The reference NIfTI voxel spacing is invalid.")
        ncr = predicted_volume == SEGMENT_CHANNELS["NCR"]
        ed = predicted_volume == SEGMENT_CHANNELS["ED"]
        et = predicted_volume == SEGMENT_CHANNELS["ET"]
        tc = ncr | et
        wt = ncr | ed | et
        region_masks = {"NCR": ncr, "ED": ed, "ET": et, "TC": tc, "WT": wt}
        voxel_volume_mm3 = float(np.prod(spacing))
        regions = {name: {"voxel_count": int(np.count_nonzero(region_masks[name])), "volume_cm3": round(int(np.count_nonzero(region_masks[name])) * voxel_volume_mm3 / 1000, 3)} for name in REGION_ORDER}
        flat_points = np.flatnonzero(wt)
        if len(flat_points) > 5500: flat_points = flat_points[np.linspace(0, len(flat_points) - 1, 5500, dtype=int)]
        voxel_points = np.column_stack(np.unravel_index(flat_points, predicted_volume.shape))
        affine_mm = reference.affine.copy(); affine_mm[:3, :] *= unit_to_mm
        world_points = nib.affines.apply_affine(affine_mm, voxel_points).astype(float)
        world_points = np.column_stack((world_points, predicted_volume[tuple(voxel_points.T)])).tolist()
        slice_counts = np.count_nonzero(wt, axis=(0, 1)); selected_index = int(np.argmax(slice_counts))
        selected_mask_full = predicted_volume[:, :, selected_index]
        selected_flair = normalize(volumes[3][:, :, selected_index])
        return (selected_flair, selected_mask_full, selected_mask_full, selected_index, tuple(float(value) for value in spacing), world_points, list(predicted_volume.shape), int(np.count_nonzero(wt)), regions, base64.b64encode(zlib.compress(predicted_volume.tobytes())).decode())

def report_pdf(analysis, result, name):
    stream = io.BytesIO(); pdf = canvas.Canvas(stream, pagesize=A4); width, height = A4
    pdf.setFillColor(colors.HexColor("#0E3B43")); pdf.rect(0, height - 110, width, 110, fill=1, stroke=0)
    pdf.setFillColor(colors.white); pdf.setFont("Helvetica-Bold", 24); pdf.drawString(46, height - 55, "MRI INSIGHT")
    pdf.setFont("Helvetica", 9); pdf.drawString(46, height - 75, "AI-assisted MRI analysis report | Research use only")
    pdf.setFillColor(colors.HexColor("#173B3D")); pdf.setFont("Helvetica-Bold", 14); pdf.drawString(46, height - 145, "Analysis summary")
    pdf.setFont("Helvetica", 10); y = height - 172
    lines = [("Patient / user", name), ("Analysis type", analysis["analysis_type"].title()), ("Created", analysis["created_at"][:19].replace("T", " "))]
    if analysis["analysis_type"] == "classification": lines += [("Primary finding", result["predicted_class"].replace("notumor", "No tumor").title()), ("Model confidence", f"{result['confidence']}%")]
    else:
        dimensions = result["reconstruction"]["shape"]
        lines += [("Volume dimensions (voxels)", " x ".join(str(value) for value in dimensions)), ("Model input shape", "128 x 128 x 4"), ("Modalities", "T1, T1ce, T2, FLAIR"), ("NIfTI voxel spacing", " x ".join(f"{value:g}" for value in result["voxel_spacing_mm"]) + " mm"), ("Selected slice", str(result["selected_slice_index"])), ("Selected-slice WT pixels", f"{result['selected_tumor_pixel_count']:,}"), ("Estimated WT slice area", f"{format_measurement(result['estimated_area_mm2'], 2)} mm2")]
        for region in REGION_ORDER:
            values = result["regions"][region]
            lines.append((f"{region} voxels / volume", f"{values['voxel_count']:,} / {values['volume_cm3']:.3f} cm3"))
    for label, value in lines:
        pdf.setFillColor(colors.HexColor("#6E7F7E")); pdf.drawString(46, y, label.upper()); pdf.setFillColor(colors.HexColor("#173B3D")); pdf.drawRightString(width - 46, y, value); y -= 25
    image_url = result.get("overlay") or result.get("source")
    if image_url: pdf.drawImage(ImageReader(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))), 46, 88, 180, 180, preserveAspectRatio=True, anchor="c")
    pdf.setFillColor(colors.HexColor("#6E7F7E")); pdf.setFont("Helvetica", 7); pdf.drawString(46, 60, "Mapping inferred from legacy inference code, not independently verified."); pdf.drawString(46, 47, "Research use only. Not a clinical diagnosis or treatment recommendation."); pdf.drawRightString(width - 46, 47, "MRI Insight")
    pdf.save(); stream.seek(0); return stream.read()

@app.before_request
def ensure_database(): init_database()

@app.get("/")
def login(): return redirect(url_for("workspace")) if "user_id" in session else render_template("login.html")

@app.post("/login")
def login_post():
    name, password = request.form.get("name", "").strip(), request.form.get("password", "")
    if len(name) < 2 or len(password) < 4: return render_template("login.html", error="Enter a name and a password with at least 4 characters."), 400
    with database() as db:
        account = db.execute("SELECT * FROM users WHERE name=?", (name,)).fetchone()
        if account is None:
            cursor = db.execute("INSERT INTO users(name,password_hash,created_at) VALUES(?,?,?)", (name, generate_password_hash(password), datetime.now(timezone.utc).isoformat())); session["user_id"], session["name"] = cursor.lastrowid, name
        elif check_password_hash(account["password_hash"], password): session["user_id"], session["name"] = account["id"], name
        else: return render_template("login.html", error="That password does not match this user."), 401
    return redirect(url_for("workspace"))

@app.get("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

@app.get("/workspace")
def workspace(): user_id(); return render_template("index.html", user_name=session["name"])

@app.post("/api/analyze/classification")
def classify():
    user_id(); file = request.files.get("scan")
    if not file or not file.filename: return jsonify(error="Upload an MRI image to continue."), 400
    try:
        with Image.open(file.stream) as uploaded: image = np.asarray(uploaded.convert("RGB"))
    except Exception: return jsonify(error="The uploaded file is not a valid readable MRI image."), 400
    try:
        batch = preprocess_input(cv2.resize(image, (224, 224)).astype(np.float32))[None, ...]
        try: probabilities = model_for("classification").predict(batch, verbose=0)[0]
        except ModelError: raise
        except Exception as error: raise ModelError(f"Disease classification inference failed: {error}") from error
    except ModelError as error: return jsonify(error=str(error)), 503
    if probabilities.shape != (4,) or not np.isfinite(probabilities).all(): return jsonify(error="The disease model returned an invalid prediction."), 503
    order = np.argsort(probabilities)[::-1]
    data = {"analysis":"classification", "source":as_data_url(image), "predicted_class":CLASS_NAMES[int(order[0])], "confidence":round(float(probabilities[order[0]]) * 100, 1), "probabilities":[{"label":CLASS_NAMES[int(i)],"value":round(float(probabilities[i]) * 100, 1)} for i in order]}
    return jsonify(save_analysis("classification", data, {"scan":file}))

@app.post("/api/analyze/segmentation")
def segment():
    user_id(); keys = ("t1", "t1ce", "t2", "flair"); files = {key: request.files.get(key) for key in keys}; study_zip = request.files.get("study_zip")
    if not (study_zip and study_zip.filename) and any(file is None or not file.filename for file in files.values()): return jsonify(error="Please upload all four MRI modalities: T1, T1ce, T2 and FLAIR."), 400
    try: flair, mask, selected_mask_full, selected_index, spacing, points, shape, tumor_voxel_count, regions, compressed_mask = best_segmentation(files, study_zip)
    except ModelError as error: return jsonify(error=str(error)), 503
    except ValueError as error: return jsonify(error=str(error)), 400
    except Exception:
        app.logger.exception("Unexpected segmentation analysis failure")
        return jsonify(error="Segmentation could not complete on this server. It may have run out of memory; try again after the service restarts."), 500
    base = np.repeat((flair * 255).astype(np.uint8)[...,None], 3, axis=-1); overlay = base.copy()
    class_colors = {1: np.array([54, 211, 153], dtype=np.uint8), 2: np.array([246, 207, 89], dtype=np.uint8), 3: np.array([242, 118, 107], dtype=np.uint8)}
    for label, color in class_colors.items():
        selected = mask == label
        overlay[selected] = (.46 * overlay[selected] + .54 * color).astype(np.uint8)
    tumor = mask > 0
    voxel_area_mm2, voxel_volume_mm3 = spacing[0] * spacing[1], spacing[0] * spacing[1] * spacing[2]
    selected_tumor_pixel_count = int(np.count_nonzero(selected_mask_full > 0))
    data = {"analysis":"segmentation","source":as_data_url(base),"overlay":as_data_url(overlay),"tumor_detected":bool(np.any(tumor)),"selected_slice_index":selected_index,"selected_tumor_pixel_count":selected_tumor_pixel_count,"estimated_area_mm2":round(selected_tumor_pixel_count * voxel_area_mm2, 2),"tumor_voxel_count":tumor_voxel_count,"estimated_volume_cm3":round(tumor_voxel_count * voxel_volume_mm3 / 1000, 3),"voxel_spacing_mm":spacing,"regions":regions,"channel_mapping":{"0":"background","1":"NCR (BraTS label 1)","2":"ED (BraTS label 2)","3":"ET (BraTS label 4; model argmax class 3)"},"reconstruction":{"points":points,"shape":shape,"coordinate_system":"NIfTI RAS+ world coordinates (mm)","mask_zlib_base64":compressed_mask}}
    stored_files = {"study_zip": study_zip} if study_zip and study_zip.filename else files
    return jsonify(save_analysis("segmentation", data, stored_files))

@app.get("/report/<int:analysis_id>")
def report(analysis_id):
    with database() as db:
        analysis = db.execute("SELECT * FROM analyses WHERE id=? AND user_id=?", (analysis_id, user_id())).fetchone()
        if not analysis: abort(404)
        content = report_pdf(analysis, json.loads(analysis["result_json"]), session["name"]); db.execute("INSERT INTO reports(analysis_id,created_at,content) VALUES(?,?,?)", (analysis_id, datetime.now(timezone.utc).isoformat(), content))
    return send_file(io.BytesIO(content), mimetype="application/pdf", as_attachment=True, download_name=f"mri-insight-report-{analysis_id}.pdf")

@app.get("/reconstruction/<int:analysis_id>")
def reconstruction(analysis_id):
    with database() as db: analysis = db.execute("SELECT * FROM analyses WHERE id=? AND user_id=? AND analysis_type='segmentation'", (analysis_id,user_id())).fetchone()
    if not analysis: abort(404)
    result = json.loads(analysis["result_json"])
    return render_template("reconstruction.html", analysis_id=analysis_id, data=result["reconstruction"], regions=result["regions"], spacing=result["voxel_spacing_mm"])

@app.errorhandler(413)
def too_large(_): return jsonify(error="The upload is too large. Please use files under 200 MB."), 413

if __name__ == "__main__": app.run(debug=True, host="127.0.0.1", port=5000)
