import threading
import time

import ipywidgets as widgets


def create_gpu_status_widget():
    utilization_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        step=0.1,
        bar_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        orientation="horizontal",
        layout=widgets.Layout(width="auto"),
    )

    memory_bar = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        step=0.1,
        bar_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        orientation="horizontal",
        layout=widgets.Layout(width="auto"),
    )

    try:
        monitor_gpu_utilization(utilization_bar, memory_bar)
        error_message = ""
        not_available_message = ""
    except Exception as e:
        error_message = f"({type(e)} - {e}"
        not_available_message = " (not available)"

    gpu_status_widget = widgets.VBox(
        [
            widgets.VBox(
                [
                    widgets.Label(
                        f"GPU core utilization:{not_available_message}",
                        layout=widgets.Layout(margin="0px 0px -5px 0px"),
                    ),
                    utilization_bar,
                ],
                layout=widgets.Layout(width="auto", justify_content="center"),
            ),
            widgets.VBox(
                [
                    widgets.Label(
                        f"GPU memory utilization:{not_available_message}",
                        layout=widgets.Layout(margin="0px 0px -5px 0px"),
                    ),
                    memory_bar,
                ],
                layout=widgets.Layout(width="auto", justify_content="center"),
            ),
        ]
    )

    return gpu_status_widget, error_message


def monitor_gpu_utilization(utilization_bar, memory_bar):
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        raise

    def update_gpu_status():
        while True:
            # cpu_proc = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

            # gpu_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            # num_procs = len(gpu_procs)

            # info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # print("Total memory:", info.total)
            # print("Free memory:", info.free)
            # print("Used memory:", info.used)

            res = pynvml.nvmlDeviceGetUtilizationRates(handle)

            utilization_bar.value = res.gpu
            if res.gpu > 70 and utilization_bar.bar_style == "info":
                utilization_bar.bar_style = "warning"
            elif res.gpu <= 70 and utilization_bar.bar_style == "warning":
                utilization_bar.bar_style = "info"

            memory_bar.value = res.memory
            if res.memory > 70 and memory_bar.bar_style == "info":
                memory_bar.bar_style = "warning"
            elif res.memory <= 70 and memory_bar.bar_style == "warning":
                memory_bar.bar_style = "info"

            time.sleep(1.0)

    # update_gpu_status()
    thread = threading.Thread(target=update_gpu_status, daemon=True)
    thread.start()
