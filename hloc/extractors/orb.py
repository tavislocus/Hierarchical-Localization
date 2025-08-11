import numpy as np
import torch
import cv2

from ..utils.base_model import BaseModel

EPS = 1e-6


class ORB(BaseModel):
    default_conf = {
        "options": {
            "nfeatures": 5000,
            "scaleFactor": 1.2,
            "nlevels": 8,
            "edgeThreshold": 31,
            "firstLevel": 0,
            "WTA_K": 2,
            "scoreType": cv2.ORB_HARRIS_SCORE,  # or cv2.ORB_FAST_SCORE ??
            "patchSize": 31,
            "fastThreshold": 20,
        },
        "descriptor": "orb",     
        "max_keypoints": -1,
    }
    required_inputs = ["image"]
    detection_noise = 1.0
    max_batch_size = 4096

    def _init(self, conf):
        if conf["descriptor"] != "orb":
            raise ValueError(f'Unknown descriptor: {conf["descriptor"]}')
        self.orb = None 
        self.dummy_param = torch.nn.Parameter(torch.empty(0))

    def _make_orb(self):
        opts = self.conf["options"]
        # OpenCV’s ORB expects 8-bit single-channel input
        self.orb = cv2.ORB_create(
            nfeatures=int(opts.get("nfeatures", 5000)),
            scaleFactor=float(opts.get("scaleFactor", 1.2)),
            nlevels=int(opts.get("nlevels", 8)),
            edgeThreshold=int(opts.get("edgeThreshold", 31)),
            firstLevel=int(opts.get("firstLevel", 0)),
            WTA_K=int(opts.get("WTA_K", 2)),
            scoreType=int(opts.get("scoreType", cv2.ORB_HARRIS_SCORE)),
            patchSize=int(opts.get("patchSize", 31)),
            fastThreshold=int(opts.get("fastThreshold", 20)),
        )

    def _forward(self, data):
        image = data["image"]
        # Expect shape [1,1,H,W], range ~ [0,1]
        image_np = image.cpu().numpy()[0, 0]
        assert image.shape[1] == 1, "ORB expects a single-channel image"
        assert image_np.min() >= -EPS and image_np.max() <= 1 + EPS

        if self.orb is None:
            self._make_orb()

        # OpenCV ORB requires uint8 grayscale
        img_u8 = np.clip(image_np * 255.0 + 0.5, 0, 255).astype(np.uint8)

        keypoints, descriptors = self.orb.detectAndCompute(img_u8, None)
        # Convert keypoints to arrays
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)         
        sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)    
        # Convert size to an approximate scale similar to SIFT scale usage:
        # SIFT keypoints store radius (sigma) *some factor; for ORB we use size/2 as scale proxy.
        scales = sizes / 2.0
        angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)    # degrees in [0,360) or -1 if undefined
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # Descriptors: (N, 32) uint8 binary ORB
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)

        device = image.device
        keypoints = torch.from_numpy(pts).to(device)
        scales = torch.from_numpy(scales).to(device)
        oris = torch.from_numpy(angles).to(device)
        scores = torch.from_numpy(responses).to(device)
        descriptors = torch.from_numpy(descriptors).to(device)  # (N,32) uint8

        if self.conf["max_keypoints"] != -1 and len(keypoints) > self.conf["max_keypoints"]:
            k = int(self.conf["max_keypoints"])
            vals, idxs = torch.topk(scores, k)
            keypoints = keypoints[idxs]
            scales = scales[idxs]
            oris = oris[idxs]
            scores = vals
            descriptors = descriptors[idxs]

        return {
            "keypoints": keypoints[None],          # [1, N, 2] (x, y)
            "scales": scales[None],                # [1, N]
            "oris": oris[None],                    # [1, N] degrees
            "scores": scores[None],                # [1, N]
            "descriptors": descriptors.T[None],    # [1, 32, N] uint8
        }
