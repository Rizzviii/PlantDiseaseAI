# Dataset Setup Guide

## Source

This project uses the **PlantVillage** dataset, available on Kaggle:  
🔗 https://www.kaggle.com/datasets/emmarex/plantdisease

PlantVillage contains 20,000+ images of healthy and diseased plant leaves across 15 classes.  
We use a **subset of 6 classes** (~60–80 images each).

---

## Step 1 — Download

1. Go to the Kaggle link above
2. Click **Download** (requires free Kaggle account)
3. Extract the ZIP — you will see a `PlantVillage/` folder with 15 subfolders

---

## Step 2 — Select and Rename Folders

From the downloaded dataset, use only these 6 folders and rename them exactly as shown:

| Original Folder Name | Rename To |
|---|---|
| `Tomato_Early_blight` | `Early_Blight` |
| `Tomato_Late_blight` | `Late_Blight` |
| `Tomato_Bacterial_spot` | `Bacterial_Blight` |
| `Tomato__Tomato_mosaic_virus` | `Mosaic_Virus` |
| `Tomato_Septoria_leaf_spot` | `Leaf_Spot` |
| `Tomato_healthy` | `Healthy` |

> **Tip:** Images from `Potato__Early_blight` can be merged into `Early_Blight` —  
> both are caused by the same pathogen (*Alternaria solani*).

---

## Step 3 — Limit Images Per Class

Each folder in PlantVillage contains 1,000+ images. You only need **60–80 per class**.  
Copy 60–80 images from each renamed folder into a new `PlantDataset/` directory:

```
PlantDataset/
  Early_Blight/      ← 60-80 .jpg files
  Late_Blight/       ← 60-80 .jpg files
  Bacterial_Blight/  ← 60-80 .jpg files
  Mosaic_Virus/      ← 60-80 .jpg files
  Leaf_Spot/         ← 60-80 .jpg files
  Healthy/           ← 60-80 .jpg files
```

Total: ~360–480 raw images → ~1,800–2,400 samples after augmentation (×5).

---

## Step 4 — Run Training

```bash
python main.py --train "C:\PlantDataset"
```

The system will:
1. Load and augment all images
2. Extract HSV features for SVM
3. Train SVM baseline
4. Train CNN (MobileNetV2, two-phase, 20 epochs)
5. Evaluate all approaches
6. Save model to `plant_disease_model.keras`
7. Save visualisation to `plant_disease_analysis.png`

---

## Notes

- Folder names must match the KB disease keys **exactly** (case-sensitive)
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- The `venv/` and `PlantDataset/` folders are excluded from the repository via `.gitignore`
