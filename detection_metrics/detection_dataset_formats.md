# Object Detection Dataset Format Schemas

---
## 0. Internal flat format 

- Used in `detection-metrics evaluate` and `detection-metrics analyze`

```
[img_id, cat_id, score, x1, y1, x2, y2]
```
- `[x1, y1, x2, y2]` absolute pixel coordinates of top-left and bottom-right corners
- `score` is confidence score for predictions, set to `1.0` for ground truths
- `cat_id` is the integer category ID (0-based)
---

## 1. Standard COCO JSON

```
dataset/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
└── images/
    ├── train2017/
    │   ├── 000001.jpg
    │   └── 000002.jpg
    ├── val2017/
    │   ├── 000003.jpg
    │   └── 000004.jpg
    └── test2017/
        └── 000005.jpg
```

- One JSON file holds ALL annotations for the entire split
- Images and annotations are in separate top-level folders
- Bbox format: `[x, y, w, h]` absolute pixels

---

## 2. Standard YOLO (Darknet / Original)

```
dataset/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── 000003.jpg
├── labels/
│   ├── 000001.txt
│   ├── 000002.txt
│   └── 000003.txt
├── train.txt               # list of train image paths
├── valid.txt               # list of val image paths
├── obj.names               # class names
└── obj.data                # config file
```

```
# obj.data
classes = 3
train  = train.txt
valid  = valid.txt
names  = obj.names
backup = backup/
```

```
# train.txt (image path list)
images/000001.jpg
images/000002.jpg
```

```
# obj.names
cat
dog
person
```

- No split subfolders — completely flat structure
- Splits are defined via path list `.txt` files, not folder hierarchy
- Bbox format: `[cx, cy, w, h]` normalized 0–1

---

## 3. Roboflow COCO Export

```
dataset/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   └── 000002.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   │   └── 000003.jpg
│   └── _annotations.coco.json
├── test/
│   ├── images/
│   │   └── 000005.jpg
│   └── _annotations.coco.json
└── README.dataset.txt
```

- Split-first folder hierarchy
- One `_annotations.coco.json` per split, placed inside the split folder alongside `images/`
- Same standard COCO JSON schema inside (`images`, `annotations`, `categories`)
- Bbox format: `[x, y, w, h]` absolute pixels

---

## 4. Ultralytics YOLO — Version A: Split-first ✅ Most Common

```
dataset/
├── train/
│   ├── images/
│   │   ├── 000001.jpg
│   │   └── 000002.jpg
│   └── labels/
│       ├── 000001.txt
│       └── 000002.txt
├── valid/
│   ├── images/
│   │   └── 000003.jpg
│   └── labels/
│       └── 000003.txt
├── test/
│   ├── images/
│   │   └── 000005.jpg
│   └── labels/
│       └── 000005.txt
└── data.yaml
```

```yaml
# data.yaml
train: train/images
val: valid/images
test: test/images
nc: 3
names: ['cat', 'dog', 'person']
```

- Default export format from Roboflow for YOLO
- Split-first, with `labels/` mirroring `images/` inside each split
- Ultralytics resolves labels by replacing `/images/` → `/labels/` in path
- Bbox format: `[cx, cy, w, h]` normalized 0–1

---

## 5. Ultralytics YOLO — Version B: Modality-first

```
dataset/
├── images/
│   ├── train/
│   │   ├── 000001.jpg
│   │   └── 000002.jpg
│   ├── valid/
│   │   └── 000003.jpg
│   └── test/
│       └── 000005.jpg
├── labels/
│   ├── train/
│   │   ├── 000001.txt
│   │   └── 000002.txt
│   ├── valid/
│   │   └── 000003.txt
│   └── test/
│       └── 000005.txt
└── data.yaml
```

```yaml
# data.yaml
train: images/train
val: images/valid
test: images/test
nc: 3
names: ['cat', 'dog', 'person']
```

- Modality-first, splits are subfolders under `images/` and `labels/`
- Ultralytics resolves labels by replacing `/images/` → `/labels/` in path
- Bbox format: `[cx, cy, w, h]` normalized 0–1

---

## Quick Comparison

| Aspect | Standard COCO | Standard YOLO (Darknet) | Roboflow COCO | YOLO Version A | YOLO Version B |
|---|---|---|---|---|---|
| Top-level grouping | Modality-first | Flat (no splits) | Split-first | Split-first | Modality-first |
| Annotation type | One JSON per split | One TXT per image | One JSON per split | One TXT per image | One TXT per image |
| Annotation location | `annotations/` root | `labels/` flat | Inside split folder | `labels/` in split | `labels/` at root |
| Split definition | Folder names | `train.txt`, `valid.txt` | Folder names | `data.yaml` | `data.yaml` |
| Config file | ❌ | `obj.data` + `obj.names` | ❌ | `data.yaml` ✅ | `data.yaml` ✅ |
| Bbox format | `[x,y,w,h]` absolute | `[cx,cy,w,h]` normalized | `[x,y,w,h]` absolute | `[cx,cy,w,h]` normalized | `[cx,cy,w,h]` normalized |
| Roboflow default | ❌ | ❌ | ✅ | ✅ | ❌ |

> **Standard YOLO (Darknet)** is the oldest and most bare-bones — no folder hierarchy, just flat files + path lists.  
> **Roboflow → Ultralytics YOLO Version A** is the most common end-to-end pipeline in practice today.
