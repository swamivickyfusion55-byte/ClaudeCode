"""Optional video finishing: contact-only face anonymisation.

No identity replacement or face-swap operations are performed here.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import cv2
import numpy as np
from stable_face_anonymizer import StableFaceAnonymizerTracker


def _face_boxes(result, w, h):
    boxes = []
    for det in (result.detections or []):
        b = det.location_data.relative_bounding_box
        x1 = max(0, int(b.xmin*w)); y1 = max(0, int(b.ymin*h))
        x2 = min(w, int((b.xmin+b.width)*w)); y2 = min(h, int((b.ymin+b.height)*h))
        if x2-x1 >= 12 and y2-y1 >= 12:
            boxes.append((x1,y1,x2,y2))
    return boxes


def _contact(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    aw=max(1,ax2-ax1); bw=max(1,bx2-bx1)
    ah=max(1,ay2-ay1); bh=max(1,by2-by1)
    gapx=max(0, max(ax1,bx1)-min(ax2,bx2))
    gapy=max(0, max(ay1,by1)-min(ay2,by2))
    return gapx < 0.17*min(aw,bw) and gapy < 0.24*min(ah,bh)


def _blur_face(frame, rect):
    x1,y1,x2,y2=rect
    roi=frame[y1:y2,x1:x2]
    if roi.size==0: return
    # Strong, reliable anonymisation with feathered ellipse, avoiding hard rectangle edges.
    h,w=roi.shape[:2]
    k=max(15, (int(min(h,w)*0.55)//2)*2+1)
    blurred=cv2.GaussianBlur(roi,(k,k),0)
    mask=np.zeros((h,w),np.uint8)
    cv2.ellipse(mask,(w//2,h//2),(max(1,int(w*.49)),max(1,int(h*.49))),0,0,360,255,-1)
    mask=cv2.GaussianBlur(mask,(0,0),max(2,min(w,h)*.06)).astype(np.float32)[...,None]/255
    roi[:]=np.clip(roi.astype(np.float32)*(1-mask)+blurred.astype(np.float32)*mask,0,255).astype(np.uint8)


def finish_video(input_path, output_path, *, contact_anonymise=False, progress=None):
    """Decode, optionally finish frames, remux source audio. Raises on failure."""
    if not contact_anonymise:
        raise ValueError('No finishing mode selected')
    import mediapipe as mp
    if contact_anonymise and not hasattr(mp.solutions,'face_detection'):
        raise RuntimeError('Installed MediaPipe does not provide FaceDetection API')
    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise RuntimeError('Cannot open finishing input')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fd, intermediate=tempfile.mkstemp(prefix='phoenix_finish_',suffix='.mp4'); os.close(fd)
    writer=None
    detector=None
    try:
        writer=cv2.VideoWriter(intermediate,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened(): raise RuntimeError('Cannot open finishing video writer')
        if contact_anonymise: detector=mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.5)
        tracker=StableFaceAnonymizerTracker(max_missed=5)
        count=0; hold=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            if detector is not None:
                det=detector.process(rgb)
                tracks=tracker.update(_face_boxes(det,w,h),frame.shape)
                current=[t for t in tracks if t.missed<=2]
                contact=any(_contact(current[i].box,current[j].box)
                            for i in range(len(current)) for j in range(i+1,len(current)))
                hold=5 if contact else max(0,hold-1)
                if hold:
                    # Contact privacy mode: anonymise only nearby faces, not distant bystanders.
                    ids=set()
                    for i in range(len(current)):
                        for j in range(i+1,len(current)):
                            if _contact(current[i].box,current[j].box): ids.update((i,j))
                    if not ids and len(current)==2: ids={0,1}
                    for i in ids:
                        _blur_face(frame,tracker.anonymisation_box(current[i],frame.shape,margin=.20))
            writer.write(frame); count+=1
            if progress and count%30==0: progress(count,total)
        writer.release(); writer=None
        if not count: raise RuntimeError('No frames decoded during finishing')
        cmd=['ffmpeg','-y','-i',intermediate,'-i',input_path,'-map','0:v:0','-map','1:a?',
             '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
             '-c:a','copy','-shortest',output_path]
        p=subprocess.run(cmd,capture_output=True,timeout=3600)
        if p.returncode or not os.path.isfile(output_path) or os.path.getsize(output_path)<1000:
            raise RuntimeError('Finishing remux failed: '+p.stderr.decode('utf8','replace')[-1000:])
        return count
    finally:
        cap.release()
        if writer: writer.release()
        if detector: detector.close()
        try: os.remove(intermediate)
        except OSError: pass
