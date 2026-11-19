"""Install trial runtime deps while preserving the pod's CUDA PyTorch."""
import subprocess
import sys


def main():
    try:
        import torch
    except ImportError:
        raise SystemExit('Use a Runpod PyTorch GPU template; CUDA PyTorch is required.')
    print('Existing PyTorch:',torch.__version__,flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Select a GPU-enabled PyTorch pod first.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    try:
        import torchvision
        from torchvision.models import resnet18
        print('Existing torchvision:',torchvision.__version__,flush=True)
    except (ImportError,RuntimeError,AttributeError,OSError):
        # Official release pairing from torch2.0/vision0.15 onward. Do not
        # upgrade/downgrade the pod's torch or replace its CUDA build.
        major,minor=map(int,torch.__version__.split('+')[0].split('.')[:2])
        if major!=2 or not 0<=minor<=14:
            raise SystemExit('Unrecognized torch release; install its matching torchvision manually.')
        version=f'0.{minor+15}.0'
        subprocess.check_call([sys.executable,'-m','pip','install','--no-deps',f'torchvision=={version}'])
    subprocess.check_call([sys.executable,'-m','pip','install','numpy>=1.26,<3',
                           'opencv-python-headless>=4.10,<6','pillow>=10,<13'])
    subprocess.check_call([sys.executable,'-c','import torch, torchvision, cv2; from torchvision.models import resnet18; print("Runtime imports OK; CUDA:",torch.cuda.is_available())'])
    print('Setup complete. In the extracted bundle run: python run_racket_ownership.py --verify-only',flush=True)


if __name__=='__main__':main()
