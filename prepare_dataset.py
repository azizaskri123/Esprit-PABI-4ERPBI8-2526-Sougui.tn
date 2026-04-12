import os
import zipfile

# =========================
# CONFIG
# =========================
DATASET = "almique/glass-bangle-defect-detection-classification"
OUTPUT_DIR = "training_data"

# =========================
# DOWNLOAD DATASET
# =========================
print("📥 Téléchargement du dataset...")
os.system(f"kaggle datasets download -d {DATASET}")

# =========================
# UNZIP
# =========================
zip_file = DATASET.split("/")[-1] + ".zip"

print("📂 Extraction...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("dataset_raw")

# =========================
# CREATE STRUCTURE
# =========================
classes_map = {
    "good": "Intact",
    "broken": "Casse",
    "defect": "Endommage"
}

for new_class in classes_map.values():
    os.makedirs(f"{OUTPUT_DIR}/{new_class}", exist_ok=True)

# =========================
# ORGANISATION
# =========================
print("📁 Organisation des images...")

for old_class, new_class in classes_map.items():
    old_path = f"dataset_raw/{old_class}"
    new_path = f"{OUTPUT_DIR}/{new_class}"

    if os.path.exists(old_path):
        for img in os.listdir(old_path):
            src = os.path.join(old_path, img)
            dst = os.path.join(new_path, img)
            os.rename(src, dst)

print("✅ Dataset prêt dans training_data/")