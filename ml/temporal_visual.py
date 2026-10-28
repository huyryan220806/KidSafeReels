"""Deterministic local video windows and frozen Kinetics R3D-18 embeddings."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights

from kidsafe.errors import InvalidMediaError, ModelNotReadyError

ROOT=Path(__file__).resolve().parents[1]
BACKBONE=ROOT/'models/checkpoints/pretrained/r3d18-kinetics400.pth'
SAMPLING='uniform5-windows16-at15fps-cv2linear171x128-crop112-v1'


def sample_windows(path):
    cap=cv2.VideoCapture(str(path))
    try:
        length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
        if length<=0 or not np.isfinite(fps) or fps<=0:
            raise InvalidMediaError('Không xác định được số frame/fps cho nhánh chuyển động.')
        last=(length-1)/fps
        starts=np.linspace(0,max(0,last-15/15),5)
        indices=np.clip(np.rint((starts[:,None]+np.arange(16)[None,:]/15)*fps).astype(int),0,length-1)
        decoded={}; current=-1
        for index in np.unique(indices):
            if current<0 or index-current>16:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index)); current=int(index)-1
            while current<int(index)-1:
                if not cap.grab():
                    raise InvalidMediaError('Không giải mã được khoảng frame chuyển động.')
                current+=1
            ok,frame=cap.read(); current=int(index)
            if not ok:
                raise InvalidMediaError('Không giải mã được frame chuyển động.')
            rgb=cv2.cvtColor(cv2.resize(frame,(171,128),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            decoded[int(index)]=rgb[8:120,29:141].copy()
        return np.stack([[decoded[int(i)] for i in row] for row in indices])
    finally:
        cap.release()


class TemporalEncoder:
    def __init__(self,path=BACKBONE,device='cpu'):
        if not Path(path).is_file():
            raise ModelNotReadyError('Thiếu backbone R3D-18 local.')
        self.device=device
        self.model=r3d_18(weights=None)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.model.fc=torch.nn.Identity(); self.model.eval().to(device)

    def encode(self,windows):
        if windows.shape!=(5,16,112,112,3) or windows.dtype!=np.uint8:
            raise ValueError('Unexpected temporal input shape/dtype')
        values=torch.from_numpy(windows).permute(0,4,1,2,3).float().to(self.device)/255.
        mean=torch.tensor([.43216,.394666,.37645],device=self.device)[None,:,None,None,None]
        std=torch.tensor([.22803,.22145,.216989],device=self.device)[None,:,None,None,None]
        values=(values-mean)/std
        with torch.inference_mode():
            # Keep inference memory bounded while training may share the GPU.
            embeddings=torch.cat([self.model(values[i:i+1]) for i in range(len(values))]).cpu().numpy()
        return np.concatenate([embeddings.mean(axis=0),embeddings.max(axis=0)])


def prepare_backbone():
    from ml.experiment_metrics import write_json
    from ml.toxic_baseline import sha256
    BACKBONE.parent.mkdir(parents=True,exist_ok=True)
    weights=R3D_18_Weights.KINETICS400_V1
    if not BACKBONE.exists():
        state=weights.get_state_dict(progress=True,check_hash=True)
        temp=BACKBONE.with_suffix('.partial'); torch.save(state,temp); temp.replace(BACKBONE)
    provenance=BACKBONE.with_suffix('.json')
    if provenance.exists():
        import json
        if json.loads(provenance.read_text())['sha256']!=sha256(BACKBONE):
            raise ValueError('R3D backbone checksum mismatch')
    else:
        write_json(provenance,dict(weights='R3D_18_Weights.KINETICS400_V1',source=weights.url,
            sha256=sha256(BACKBONE),sampling=SAMPLING,
            preprocessing='OpenCV INTER_LINEAR resize 171x128, center crop112, official Kinetics normalization; same at train/inference'))
