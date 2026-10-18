# MRI Insight

A local, privacy-first web interface for the supplied models:

- **Attention U-Net**: BraTS-style four-modality NIfTI segmentation (T1, T1ce, T2, FLAIR).
- **EfficientNetV2-B0**: one-image, four-class brain MRI classification (`glioma`, `meningioma`, `notumor`, `pituitary`).

## Added workspace features

- A first-screen account gate. Names and securely hashed passwords are saved in the local SQLite database (`mri_insight.db`).
- Every uploaded scan, completed prediction, reconstruction data point, and generated PDF report is saved to that database under the signed-in user.
- PDF download buttons for both analysis paths.
- A separate, interactive 3D predicted-mask reconstruction screen following segmentation.
- Segmentation accepts either four aligned `.nii` / `.nii.gz` inputs or one ZIP containing all four named modalities. ZIP uploads are safely inspected and must identify T1, T1ce, T2, and FLAIR unambiguously.

## Run locally

Requires Python 3.10+.

```powershell
# Run this from the directory containing the extracted mri-insight folder.
cd .\mri-insight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The first analysis takes longer while the selected model loads. All files and results remain local and are saved in the project's SQLite database.

Set a persistent session secret before use outside local development:

```powershell
$env:MRI_INSIGHT_SECRET = "use-a-long-random-secret"
```

> Research use only — this interface is not a medical device and must not be used as a diagnostic or treatment decision tool.

## Verified model and measurement notes

The facts below were read from the supplied `.keras` model configs and this project's inference code. They do not establish the missing training procedure.

- `attention_unet.keras` is a 2D convolutional encoder/decoder with pooling, upsampling, and attention-style gates. Its saved input is `(None, 128, 128, 4)` and its output is `(None, 128, 128, 4)`. The final layer has four channels with softmax activation.
- `brain_disease_model.keras` accepts `(None, 224, 224, 3)` and returns four softmax outputs. Its final feature layers include EfficientNetV2 and a 4-unit classifier. The code's class order is `glioma`, `meningioma`, `notumor`, `pituitary`; the model output shape alone does not prove that label order.
- Classification keeps the existing RGB resize to 224×224 and EfficientNetV2 preprocessing call. The saved model contains an internal 1/255 rescaling layer. The model and source contain no training notebook or training history, so epochs, batch size, augmentation, and the training-time preprocessing are unknown.
- Segmentation keeps the order already used in this app: T1, T1ce, T2, FLAIR. Each axis-2 slice is normalized per slice using the current min/max code, resized to 128×128, stacked as float32 channels, predicted, and restored to the original in-plane dimensions with nearest-neighbor interpolation. This is the verified current inference behavior; exact training-time preprocessing is unknown.
- The saved Attention U-Net has four output channels with a softmax final layer and output shape `(None, 128, 128, 4)`. Its `.keras` archive does not contain semantic class names. The supplied legacy MRI Insight inference source explicitly maps argmax labels 0/1/2/3 to background/NCR/ED/ET, respectively; this is the application's established channel mapping, not independently recoverable training metadata. The result and report disclose this provenance. Internal ET class 3 corresponds to BraTS semantic label 4.
- The complete restored 3D argmax mask is counted with boolean masks: NCR=`mask==1`, ED=`mask==2`, ET=`mask==3`, TC=`NCR OR ET`, WT=`NCR OR ED OR ET`. Every region volume is `voxel count × spacing_x × spacing_y × spacing_z / 1000` cm³, where spacing comes from reference NIfTI header zooms converted using its declared spatial units (mm, meter, or micron). Unknown units are rejected.
- The overlay colors argmax channels 1/2/3 as NCR/ED/ET. 3D points (up to 5,500) are sampled from the full WT mask for display and transformed to RAS+ world millimeters. The complete argmax mask remains stored with the analysis.
- The PDF and result page read the same saved backend result fields. The existing login, workspace, upload flow, styling, and page structure are preserved.

The measurements describe the model's predicted foreground and voxel geometry. They are not clinical tumor measurements or a clinical diagnosis.

The model configs serialize compile settings: the Attention U-Net lists Adam with learning rate `0.0001`, a custom `combined_loss` identifier, and a `dice_coef` metric; the classifier lists Adam with learning rate `0.001`, categorical cross-entropy with label smoothing `0.1`, and accuracy. These are the saved Keras compile configs, not a training notebook or run history. Training epochs, training batch size, augmentation details, and the full implementation of the custom segmentation loss are not available in the supplied files. The training-time input normalization cannot be established from these artifacts.

For NIfTI input, the app checks equal dimensions, spatial units, and voxel-to-world affines. Those checks detect common cross-study or orientation mismatches, but NIfTI files without study identifiers do not prove that four matching files came from the same acquisition.
