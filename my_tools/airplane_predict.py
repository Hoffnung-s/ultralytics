from ultralytics import YOLO

# Load a model
model = YOLO("/cv/temp/yolo26m-seg.pt")  # load an official model


img_dir = "/cv/project_data/airplane/pics/vlcsnap-2026-08-17-15h12m44s049.png"

# Predict with the model

results = model.predict(source=img_dir, conf=0.3,save=True,
                        save_txt=True, batch=16, device=0,
                        project='/cv/project_data/airplane/runs',
                        name='segment',
                        )

# Access the results
for result in results:
    xy = result.masks.xy  # mask polygons in pixel coordinates
    xyn = result.masks.xyn  # normalized mask polygons
    masks = result.masks.data  # binary masks, shape (N,H,W), dtype torch.uint8
    boxes = result.boxes       #(N)  Instance boxes/classes/confidences.
    cls = result.boxes.cls     #(N,)  Class IDs; cast to int for names.

    print(f'boxes {boxes.data}\n'
          )