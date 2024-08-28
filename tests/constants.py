import GPUtil

CUDA_AVAILABLE = len(GPUtil.getAvailable()) > 0
