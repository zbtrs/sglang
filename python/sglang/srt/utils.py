"""Common utilities."""

import base64
import multiprocessing
import logging
import os
import random
import socket
import time
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests
import rpyc
import torch
import triton
from rpyc.utils.server import ThreadedServer
from fastapi.responses import JSONResponse
from packaging import version as pkg_version
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


show_time_cost = False
time_infos = {}


def enable_show_time_cost():
    global show_time_cost
    show_time_cost = True


class TimeInfo:
    def __init__(self, name, interval=0.1, color=0, indent=0):
        self.name = name
        self.interval = interval
        self.color = color
        self.indent = indent

        self.acc_time = 0
        self.last_acc_time = 0

    def check(self):
        if self.acc_time - self.last_acc_time > self.interval:
            self.last_acc_time = self.acc_time
            return True
        return False

    def pretty_print(self):
        print(f"\x1b[{self.color}m", end="")
        print("-" * self.indent * 2, end="")
        print(f"{self.name}: {self.acc_time:.3f}s\x1b[0m")


def mark_start(name, interval=0.1, color=0, indent=0):
    global time_infos, show_time_cost
    if not show_time_cost:
        return
    torch.cuda.synchronize()
    if time_infos.get(name, None) is None:
        time_infos[name] = TimeInfo(name, interval, color, indent)
    time_infos[name].acc_time -= time.time()


def mark_end(name):
    global time_infos, show_time_cost
    if not show_time_cost:
        return
    torch.cuda.synchronize()
    time_infos[name].acc_time += time.time()
    if time_infos[name].check():
        time_infos[name].pretty_print()


def calculate_time(show=False, min_cost_ms=0.0):
    def wrapper(func):
        def inner_func(*args, **kwargs):
            torch.cuda.synchronize()
            if show:
                start_time = time.time()
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            if show:
                cost_time = (time.time() - start_time) * 1000
                if cost_time > min_cost_ms:
                    print(f"Function {func.__name__} took {cost_time} ms to run.")
            return result

        return inner_func

    return wrapper


def get_available_gpu_memory(gpu_id, distributed=False):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARNING: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    torch.cuda.empty_cache()
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False


def allocate_init_ports(
    port: Optional[int] = None,
    additional_ports: Optional[List[int]] = None,
    tp_size: int = 1,
    dp_size: int = 1,
):
    """Allocate ports for all connections."""
    if additional_ports:
        ret_ports = [port] + additional_ports
    else:
        ret_ports = [port]

    ret_ports = list(set(x for x in ret_ports if is_port_available(x)))
    cur_port = ret_ports[-1] + 1 if len(ret_ports) > 0 else 10000

    # HTTP + Tokenizer + Controller + Detokenizer + dp_size * (nccl + tp_size)
    num_ports_needed = 6 + dp_size * (1 + tp_size)
    while len(ret_ports) < num_ports_needed:
        if cur_port not in ret_ports and is_port_available(cur_port):
            ret_ports.append(cur_port)
        cur_port += 1

    if port is not None and ret_ports[0] != port:
        logger.warn(
            f"WARNING: Port {port} is not available. Use port {ret_ports[0]} instead."
        )

    return ret_ports[0], ret_ports[1:num_ports_needed]


def get_int_token_logit_bias(tokenizer, vocab_size):
    """Get the logit bias for integer-only tokens."""
    # a bug when model's vocab size > tokenizer.vocab_size
    vocab_size = tokenizer.vocab_size
    logit_bias = np.zeros(vocab_size, dtype=np.float32)
    for t_id in range(vocab_size):
        ss = tokenizer.decode([t_id]).strip()
        if not (ss.isdigit() or len(ss) == 0 or t_id == tokenizer.eos_token_id):
            logit_bias[t_id] = -1e5

    return logit_bias


def wrap_kernel_launcher(kernel):
    """A faster launcher for triton kernels."""
    if int(triton.__version__.split(".")[0]) >= 3:
        return None

    gpu_id = torch.cuda.current_device()
    kernels = kernel.cache[gpu_id].values()
    kernel = next(iter(kernels))

    # Different trition versions use different low-level names
    if hasattr(kernel, "cu_function"):
        kfunction = kernel.cu_function
    else:
        kfunction = kernel.function

    if hasattr(kernel, "c_wrapper"):
        run = kernel.c_wrapper
    else:
        run = kernel.run

    add_cluster_dim = True

    def ret_func(grid, num_warps, *args):
        nonlocal add_cluster_dim

        try:
            if add_cluster_dim:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    1,
                    1,
                    1,
                    1,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
            else:
                run(
                    grid[0],
                    grid[1],
                    grid[2],
                    num_warps,
                    kernel.shared,
                    0,
                    kfunction,
                    None,
                    None,
                    kernel,
                    *args,
                )
        except TypeError:
            add_cluster_dim = not add_cluster_dim
            ret_func(grid, num_warps, *args)

    return ret_func


def is_multimodal_model(model):
    from sglang.srt.model_config import ModelConfig

    if isinstance(model, str):
        model = model.lower()
        return "llava" in model or "yi-vl" in model or "llava-next" in model

    if isinstance(model, ModelConfig):
        model_path = model.path.lower()
        return (
            "llava" in model_path or "yi-vl" in model_path or "llava-next" in model_path
        )

    raise ValueError("unrecognized type")


def decode_video_base64(video_base64):
    from PIL import Image

    # Decode the base64 string
    video_bytes = base64.b64decode(video_base64)

    # Placeholder for the start indices of each PNG image
    img_starts = []

    frame_format = "PNG"  # str(os.getenv('FRAME_FORMAT', "JPEG"))

    assert frame_format in [
        "PNG",
        "JPEG",
    ], "FRAME_FORMAT must be either 'PNG' or 'JPEG'"

    if frame_format == "PNG":
        # Find each PNG start signature to isolate images
        i = 0
        while i < len(video_bytes) - 7:  # Adjusted for the length of the PNG signature
            # Check if we found the start of a PNG file
            if (
                video_bytes[i] == 0x89
                and video_bytes[i + 1] == 0x50
                and video_bytes[i + 2] == 0x4E
                and video_bytes[i + 3] == 0x47
                and video_bytes[i + 4] == 0x0D
                and video_bytes[i + 5] == 0x0A
                and video_bytes[i + 6] == 0x1A
                and video_bytes[i + 7] == 0x0A
            ):
                img_starts.append(i)
                i += 8  # Skip the PNG signature
            else:
                i += 1
    else:
        # Find each JPEG start (0xFFD8) to isolate images
        i = 0
        while (
            i < len(video_bytes) - 1
        ):  # Adjusted for the length of the JPEG SOI signature
            # Check if we found the start of a JPEG file
            if video_bytes[i] == 0xFF and video_bytes[i + 1] == 0xD8:
                img_starts.append(i)
                # Move to the next byte to continue searching for the next image start
                i += 2
            else:
                i += 1

    frames = []
    for start_idx in img_starts:
        # Assuming each image is back-to-back, the end of one image is the start of another
        # The last image goes until the end of the byte string
        end_idx = (
            img_starts[img_starts.index(start_idx) + 1]
            if img_starts.index(start_idx) + 1 < len(img_starts)
            else len(video_bytes)
        )
        img_bytes = video_bytes[start_idx:end_idx]

        # Convert bytes to a PIL Image
        img = Image.open(BytesIO(img_bytes))

        # Convert PIL Image to a NumPy array
        frame = np.array(img)

        # Append the frame to the list of frames
        frames.append(frame)

    # Ensure there's at least one frame to avoid errors with np.stack
    if frames:
        return np.stack(frames, axis=0), img.size
    else:
        return np.array([]), (
            0,
            0,
        )  # Return an empty array and size tuple if no frames were found


def load_image(image_file):
    from PIL import Image

    image = image_size = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    elif image_file.startswith("video:"):
        image_file = image_file.replace("video:", "")
        image, image_size = decode_video_base64(image_file)
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image, image_size


def init_rpyc_service(service: rpyc.Service, port: int):
    t = ThreadedServer(
        service=service,
        port=port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_pickle": True,
            "sync_request_timeout": 3600
        },
    )
    t.logger.setLevel(logging.WARN)
    t.start()


def connect_to_rpyc_service(port, host="localhost"):
    time.sleep(1)

    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect(
                host,
                port,
                config={
                    "allow_public_attrs": True,
                    "allow_pickle": True,
                    "sync_request_timeout": 3600
                },
            )
            break
        except ConnectionRefusedError:
            time.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise RuntimeError("init rpc env error!")

    return con.root


def start_rpyc_process(service: rpyc.Service, port: int):
    # Return the proxy and the process
    proc = multiprocessing.Process(target=init_rpyc_service, args=(service, port))
    proc.start()
    proxy = connect_to_rpyc_service(port)
    assert proc.is_alive()
    return proxy, proc


def suppress_other_loggers():
    from vllm.logger import logger as vllm_default_logger

    vllm_default_logger.setLevel(logging.WARN)
    logging.getLogger("vllm.utils").setLevel(logging.WARN)
    logging.getLogger("vllm.selector").setLevel(logging.WARN)
    logging.getLogger("vllm.config").setLevel(logging.ERROR)


def assert_pkg_version(pkg: str, min_version: str):
    try:
        installed_version = version(pkg)
        if pkg_version.parse(installed_version) < pkg_version.parse(min_version):
            raise Exception(
                f"{pkg} is installed with version {installed_version} which "
                f"is less than the minimum required version {min_version}"
            )
    except PackageNotFoundError:
        raise Exception(
            f"{pkg} with minimum required version {min_version} is not installed"
        )


API_KEY_HEADER_NAME = "X-API-Key"


class APIKeyValidatorMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        # extract API key from the request headers
        api_key_header = request.headers.get(API_KEY_HEADER_NAME)
        if not api_key_header or api_key_header != self.api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API Key"},
            )
        response = await call_next(request)
        return response
