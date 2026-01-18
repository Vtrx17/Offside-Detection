from ultralytics import YOLO

model = YOLO(
    "C:\\Users\\ROG\\Desktop\\VTRX\\Academics_IIUM\\sem7_2026\\MV\\VSCODE\\football_analysis-main\\football_analysis-main\\models\\best.pt"
)

results = model.predict(
    "C:\\Users\\ROG\\Desktop\\VTRX\\Academics_IIUM\\sem7_2026\\MV\\VSCODE\\football_analysis-main\\football_analysis-main\\input_videos\\Football.mp4",
    save=True,
)
print(results[0])
print("=====================================")
for box in results[0].boxes:
    print(box)
