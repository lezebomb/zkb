COCO8 contest smoke dataset workspace.
With --allow-download, train_detect_yolo.py can use --data coco8.yaml and Ultralytics will fetch real COCO8.
This local data.yaml keeps the contest 9-class order for converted/custom data.
台灯 is not covered by COCO and needs extra data or a second-stage strategy.
Ultralytics is not installed, so no COCO8 download was attempted: [WinError 5] 拒绝访问。: 'C:\\Users\\24981\\AppData\\Roaming\\Ultralytics'
Install with: pip install -r requirements-detect.txt
