#!/usr/bin/env python3
"""
Ollama Monitor - A system tray application to monitor Ollama AI models.
Original Created by: Yusuf Emre ALBAYRAK
Now with a built-in Health API for ComfyUI, detailed hover text, and a model kill-switch.
This was modified to be a companion app to: https://github.com/pupphelper/Openwebui-ComfyUI-Universal-Advanced
"""
import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
import logging
import multiprocessing
import psutil

from datetime import datetime, timedelta
from functools import partial
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict
from urllib.parse import urlparse

import re
import subprocess

import pystray
import httpx
import aiohttp
from PIL import Image
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
import winreg

import pynvml
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

__version__ = "2.7.2-Debug" # Version updated for heavy debugging
__author__ = "Yusuf Emre ALBAYRAK modified by \"theJSN\""

# --- HELPER to get a consistent timestamp for debug messages ---
def debug_ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- FastAPI Setup ---
api_app = FastAPI()

class SystemHealth(BaseModel):
    available: bool
    comfyui_status: str
    queue_running: Optional[int] = None
    queue_pending: Optional[int] = None
    ollama_model_processor: Optional[str] = None
    ollama_model_expiry: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_used_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    utilization_gpu_percent: Optional[int] = None
    gpu_temperature_celsius: Optional[int] = None
    ram_total_mb: Optional[int] = None
    ram_used_mb: Optional[int] = None
    cpu_utilization_percent: Optional[float] = None
    cpu_temperature_celsius: Optional[float] = None

monitor_instance = None

@api_app.get("/health", response_model=SystemHealth)
async def get_health_endpoint():
    if monitor_instance:
        health_data = await monitor_instance.get_system_health()
        return health_data
    raise HTTPException(status_code=503, detail="Monitor not initialized")

def setup_logging():
    log_dir = os.path.join(os.getenv('APPDATA'), 'OllamaMonitor', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'ollama_monitor_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    logger = logging.getLogger('OllamaMonitor')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

class SettingsWindow:
    def __init__(self, monitor: 'OllamaMonitor'):
        self.monitor = monitor
        self.window = tk.Tk()
        self.window.title("Ollama Monitor - Settings")
        self.window.geometry("400x580")
        self.window.resizable(False, False)
        
        self.style = ttk.Style()
        self.style.theme_use('vista')
        
        self._create_widgets()
        self._center_window()
        
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        self.window.transient()
        self.window.grab_set()
        self.window.focus_set()
        
        self.window.mainloop()
    
    def _create_widgets(self):
        title_label = ttk.Label(self.window, text="Ollama Monitor", font=('Segoe UI', 14, 'bold'))
        title_label.pack(pady=10)
        
        settings_frame = ttk.LabelFrame(self.window, text="General Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        self.startup_var = tk.BooleanVar(value=self.monitor.settings.get('startup', False))
        startup_check = ttk.Checkbutton(settings_frame, text="Run at Windows startup", variable=self.startup_var, command=self.toggle_startup)
        startup_check.pack(anchor="w", pady=2)

        self.health_api_var = tk.BooleanVar(value=self.monitor.settings.get('health_api_enabled', True))
        health_api_check = ttk.Checkbutton(settings_frame, text="Enable Health API Server (requires restart)", variable=self.health_api_var)
        health_api_check.pack(anchor="w", pady=2)
        
        api_frame = ttk.LabelFrame(self.window, text="API Connections", padding=10)
        api_frame.pack(fill="x", padx=10, pady=5)
        
        ollama_url_frame = ttk.Frame(api_frame)
        ollama_url_frame.pack(fill="x", pady=2)
        ttk.Label(ollama_url_frame, text="Ollama URL:").pack(side="left")
        self.api_url_var = tk.StringVar(value=self.monitor.api_url)
        ttk.Entry(ollama_url_frame, textvariable=self.api_url_var, width=30).pack(side="right")
        
        comfy_url_frame = ttk.Frame(api_frame)
        comfy_url_frame.pack(fill="x", pady=2)
        ttk.Label(comfy_url_frame, text="ComfyUI URL:").pack(side="left")
        self.comfyui_url_var = tk.StringVar(value=self.monitor.comfyui_url)
        ttk.Entry(comfy_url_frame, textvariable=self.comfyui_url_var, width=30).pack(side="right")

        port_frame = ttk.Frame(api_frame)
        port_frame.pack(fill="x", pady=2)
        ttk.Label(port_frame, text="Health API Port:").pack(side="left")
        self.api_port_var = tk.StringVar(value=str(self.monitor.api_port))
        ttk.Entry(port_frame, textvariable=self.api_port_var, width=30).pack(side="right")

        self.save_api_btn = ttk.Button(api_frame, text="Save Connection Settings", command=self.save_api_settings)
        self.save_api_btn.pack(pady=5)
        
        about_frame = ttk.LabelFrame(self.window, text="About", padding=10)
        about_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(about_frame, text=f"Version: {__version__}").pack(anchor="w")
        ttk.Label(about_frame, text="Ollama Monitor with Health API\nCreated by Yusuf Emre ALBAYRAK", justify="left").pack(anchor="w", pady=5)
        github_link = ttk.Label(about_frame, text="GitHub Repository", cursor="hand2", foreground="blue")
        github_link.pack(anchor="w")
        github_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/ysfemreAlbyrk/ollama-monitor"))
        
        close_btn = ttk.Button(self.window, text="Close", command=self.close_window)
        close_btn.pack(side="bottom", pady=10)
    
    def _center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def close_window(self):
        if self.save_api_btn['state'] == 'disabled':
            return
        self.window.destroy()

    def toggle_startup(self):
        startup = self.startup_var.get()
        self.monitor.settings['startup'] = startup
        self.monitor.save_settings()
        key_path = r"Software\\Microsoft\\Windows\\CurrentVersion\\Run"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS)
            if startup:
                executable_path = sys.executable if hasattr(sys, 'frozen') else os.path.abspath(sys.argv[0])
                winreg.SetValueEx(key, "OllamaMonitor", 0, winreg.REG_SZ, f'"{executable_path}"')
            else:
                try: winreg.DeleteValue(key, "OllamaMonitor")
                except WindowsError: pass
            winreg.CloseKey(key)
        except Exception as e:
            self.monitor.logger.error(f"Failed to save startup setting: {str(e)}")
    
    def save_api_settings(self):
        try:
            api_url = self.api_url_var.get().strip()
            comfyui_url = self.comfyui_url_var.get().strip()
            api_port = self.api_port_var.get().strip()
            health_api_enabled = self.health_api_var.get()

            if not urlparse(api_url).scheme or not urlparse(api_url).netloc: raise ValueError("Invalid Ollama URL format")
            if comfyui_url and (not urlparse(comfyui_url).scheme or not urlparse(comfyui_url).netloc): raise ValueError("Invalid ComfyUI URL format")
            if not api_port.isdigit() or not 1024 <= int(api_port) <= 65535: raise ValueError("API Port must be a number between 1024 and 65535")

            restart_needed = self.monitor.settings.get('health_api_enabled', True) != health_api_enabled

            self.monitor.settings['api_url'] = api_url
            self.monitor.settings['comfyui_url'] = comfyui_url
            self.monitor.settings['api_port'] = int(api_port)
            self.monitor.settings['health_api_enabled'] = health_api_enabled
            self.monitor.save_settings()
            
            self.save_api_btn.config(state="disabled")
            
            save_thread = threading.Thread(target=self._update_connection_and_close)
            save_thread.daemon = True
            save_thread.start()

            if restart_needed:
                messagebox.showinfo("Restart Required", "Changes to the Health API Server setting will take effect after you restart the application.")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid settings: {str(e)}")

    def _update_connection_and_close(self):
        try:
            self.monitor.logger.info("Settings saved! The new URL will be used on the next poll.")
            self.monitor.icon.notify("Settings saved!")
        except Exception as e:
            self.monitor.logger.error(f"Failed during settings save notification: {e}")
            self.monitor.icon.notify(f"Error saving settings: {e}")
        finally:
            self.window.after(0, self.window.destroy)

class OllamaMonitor:
    DEFAULT_API_HOST = "localhost"
    DEFAULT_API_PORT = "11434"
    DEFAULT_COMFYUI_URL = "http://localhost:8188"
    DEFAULT_HEALTH_API_PORT = 9188
    DETAILED_POLL_INTERVAL = 60
    
    def __init__(self):
        self.logger = setup_logging()
        self.logger.info(f"Starting Ollama Monitor v{__version__}")
        print(f"[{debug_ts()}] [INIT] OllamaMonitor class initializing...")
        
        self.current_model = "Waiting..."
        self.detailed_model_data = {}
        self.last_detailed_fetch_time = 0
        self.icon: Optional[pystray.Icon] = None
        self.should_run = True
        self.last_status = None
        
        self._avail_lock = threading.Lock()
        self.available = True
        self.unavailable_until = None

        self.icon_red = resource_path("icons/icon_red.png")
        self.icon_blue = resource_path("icons/icon_blue.png")
        self.icon_green = resource_path("icons/icon_green.png")
        
        self.settings_file = os.path.join(os.getenv('APPDATA'), 'OllamaMonitor', 'settings.json')
        self.load_settings()

        self.nvml_initialized = False
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            self.logger.info("NVML initialized.")
        except pynvml.NVMLError as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
        
        psutil.cpu_percent(interval=None)
        print(f"[{debug_ts()}] [INIT] OllamaMonitor class initialized successfully.")


    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f: self.settings = json.load(f)
            else:
                self.settings = {}
            
            self.settings.setdefault('startup', False)
            self.settings.setdefault('api_url', f'http://{self.DEFAULT_API_HOST}:{self.DEFAULT_API_PORT}')
            self.settings.setdefault('comfyui_url', self.DEFAULT_COMFYUI_URL)
            self.settings.setdefault('api_port', self.DEFAULT_HEALTH_API_PORT)
            self.settings.setdefault('health_api_enabled', True)

            self.save_settings()
            self.logger.info("Settings loaded/initialized.")
        except Exception as e:
            self.logger.error(f"Error loading settings: {str(e)}")
            self.settings = {
                'startup': False, 'api_url': f'http://{self.DEFAULT_API_HOST}:{self.DEFAULT_API_PORT}',
                'comfyui_url': self.DEFAULT_COMFYUI_URL, 'api_port': self.DEFAULT_HEALTH_API_PORT,
                'health_api_enabled': True
            }

    async def get_system_health(self) -> Dict:
        with self._avail_lock:
            is_temporarily_unavailable = self.unavailable_until and datetime.now() < self.unavailable_until
            is_currently_available = self.available and not is_temporarily_unavailable

        health = {"available": is_currently_available, "comfyui_status": "offline"}
        
        try:
            if self.comfyui_url:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f'{self.comfyui_url}/queue', timeout=2)
                    if response.status_code == 200:
                        health["comfyui_status"] = "online"
                        data = response.json()
                        health["queue_running"] = len(data.get('queue_running', []))
                        health["queue_pending"] = len(data.get('queue_pending', []))
        except Exception as e:
            self.logger.warning(f"Could not connect to ComfyUI: {e}")

        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                health.update({
                    "vram_total_mb": mem_info.total // 1024**2, "vram_used_mb": mem_info.used // 1024**2,
                    "vram_free_mb": mem_info.free // 1024**2, "utilization_gpu_percent": util_info.gpu,
                    "gpu_temperature_celsius": temp
                })
            except pynvml.NVMLError as e:
                self.logger.error(f"Error querying GPU: {e}")
        
        try:
            ram = psutil.virtual_memory()
            health["ram_total_mb"] = ram.total // 1024**2
            health["ram_used_mb"] = ram.used // 1024**2
            health["cpu_utilization_percent"] = psutil.cpu_percent(interval=None)
        except Exception as e:
            self.logger.error(f"Error querying RAM or CPU: {e}")
            
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            for key in temps:
                if key.lower().startswith("cpu") or "coretemp" in key.lower() or "package" in key.lower():
                    for entry in temps[key]:
                        if hasattr(entry, "current"):
                            cpu_temp = entry.current
                            break
                    if cpu_temp is not None:
                        break
        except Exception as e:
            self.logger.debug(f"Could not read CPU temperature: {e}")

        if cpu_temp is not None:
            health["cpu_temperature_celsius"] = cpu_temp

        health.update({
            "ollama_model_processor": self.detailed_model_data.get("processor"),
            "ollama_model_expiry": self.detailed_model_data.get("expiry")
        })

        return health

    def run_api_server(self):
        self.logger.info(f"Starting health API server on port {self.api_port}")
        try:
            uvicorn.run(api_app, host="0.0.0.0", port=self.api_port, log_level="warning")
        except Exception as e:
            self.logger.error(f"FATAL: API server thread crashed!", exc_info=True)
            print(f"[{debug_ts()}] [FATAL] API server thread crashed: {e}")

    def stop(self):
        self.should_run = False
        self.logger.info("Stopping Ollama Monitor...")
        print(f"[{debug_ts()}] [STOP] Stop requested. should_run set to False.")
        if self.nvml_initialized:
            try: pynvml.nvmlShutdown()
            except pynvml.NVMLError as e: self.logger.error(f"Error during NVML shutdown: {e}")
        if self.icon:
            self.icon.visible = False
            self.icon.stop()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
                print(f"[{debug_ts()}] [STOP] Event loop stop requested.")
        except Exception:
            pass

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f: json.dump(self.settings, f, indent=4)
        except Exception as e: self.logger.error(f"Error saving settings: {str(e)}")
            
    def _unload_all_models_sync(self):
        self.logger.info("Attempting to unload all models...")
        try:
            with httpx.Client(verify=False, timeout=20) as client:
                response = client.get(f'{self.api_url}/api/ps')
                response.raise_for_status()
                data = response.json()
                running_models = data.get('models', [])

                if not running_models:
                    self.logger.info("No models were running to unload.")
                    if self.icon: self.icon.notify("No models were running.")
                    return

                unloaded_count = 0
                for model in running_models:
                    model_name = model.get("name")
                    if model_name:
                        self.logger.info(f"Unloading model: {model_name}")
                        unload_payload = {"model": model_name, "keep_alive": 0}
                        client.post(f'{self.api_url}/api/generate', json=unload_payload)
                        unloaded_count += 1
                
                if unloaded_count > 0:
                    self.logger.info(f"Successfully sent unload requests for {unloaded_count} model(s).")
                    if self.icon: self.icon.notify(f"Unloaded {unloaded_count} model(s).")

        except httpx.RequestError as e:
            self.logger.error(f"Failed to unload models due to a network error: {e}")
            if self.icon: self.icon.notify("Error: Could not contact Ollama server.")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model unload: {e}", exc_info=True)
            if self.icon: self.icon.notify(f"Error during unload: {e}")
            
    def _update_detailed_model_info(self):
        try:
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(
                ['ollama', 'ps'], capture_output=True, text=True, check=True,
                startupinfo=startupinfo, encoding='utf-8'
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                data_line = lines[1]
                parts = re.split(r'\s{2,}', data_line)
                if len(parts) >= 6:
                    name, _, size, processor, _, expiry = parts
                    self.detailed_model_data = {
                        "name": name.strip(), "size": size.strip(),
                        "processor": processor.strip(), "expiry": expiry.strip()
                    }
                    return
        except FileNotFoundError:
            self.logger.error("'ollama' command not found. Is it in your system's PATH?")
            self.detailed_model_data = {"error": "Ollama command not found"}
        except Exception as e:
            self.logger.error(f"Error running 'ollama ps': {e}")
        
        self.detailed_model_data = {}

    def create_icon(self, status: str) -> Image.Image:
        if "Ollama Not Running" in status: icon_path = self.icon_red
        elif "No Model Running" in status: icon_path = self.icon_blue
        else: icon_path = self.icon_green
        return Image.open(icon_path)

    def create_menu(self) -> pystray.Menu:
        is_model_running = "No Model Running" not in self.current_model and "Ollama Not Running" not in self.current_model
        
        with self._avail_lock:
            is_temporarily_unavailable = self.unavailable_until and datetime.now() < self.unavailable_until
            is_currently_available = self.available and not is_temporarily_unavailable
        availability_text = "Set to Unavailable" if is_currently_available else "Set to Available"

        return pystray.Menu(
            pystray.MenuItem(lambda text: self.current_model, lambda _: None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(availability_text, self.toggle_availability),
            pystray.MenuItem(
                "Temporarily Unavailable",
                pystray.Menu(
                    pystray.MenuItem("for 1 Minute", partial(self.set_temporarily_unavailable, 1/60)),
                    pystray.MenuItem("for 1 Hour", partial(self.set_temporarily_unavailable, 1)),
                    pystray.MenuItem("for 2 Hours", partial(self.set_temporarily_unavailable, 2)),
                    pystray.MenuItem("for 4 Hours", partial(self.set_temporarily_unavailable, 4)),
                    pystray.MenuItem("for 8 Hours", partial(self.set_temporarily_unavailable, 8)),
                )
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Unload All Models", self._unload_all_models_sync, enabled=is_model_running),
            pystray.MenuItem("Settings", self.show_settings),
            pystray.MenuItem("Exit", self.stop)
        )

    def set_temporarily_unavailable(self, hours: float, icon=None, item=None):
        with self._avail_lock:
            self.unavailable_until = datetime.now() + timedelta(hours=hours)
        
        if hours < 1:
            mins = int(hours * 60)
            expiry_str = (self.unavailable_until).strftime('%H:%M:%S')
            self.logger.info(f"Status set to temporarily unavailable for {mins} minute(s). Available again at {expiry_str}")
            notify_msg = f"Status is temporarily unavailable. Resets at {expiry_str}"
        else:
            expiry_str = (self.unavailable_until).strftime('%H:%M:%S')
            self.logger.info(f"Status set to temporarily unavailable for {hours} hour(s). Available again at {expiry_str}")
            notify_msg = f"Status is temporarily unavailable. Resets at {expiry_str}"
        if self.icon:
            self.icon.notify(notify_msg)
            self.icon.menu = self.create_menu()
            self.icon.icon = self.create_icon(self.current_model)

    def set_available(self, icon=None, item=None):
        with self._avail_lock:
            self.available = True
            self.unavailable_until = None
        self.logger.info("Status set to Available.")
        if self.icon:
            self.icon.notify("Status is now Available.")
            self.icon.menu = self.create_menu()
            self.icon.icon = self.create_icon(self.current_model)

    def set_unavailable(self, icon=None, item=None):
        with self._avail_lock:
            self.available = False
            self.unavailable_until = None
        self.logger.info("Status set to Unavailable.")
        if self.icon:
            self.icon.notify("Status is now Unavailable.")
            self.icon.menu = self.create_menu()
            self.icon.icon = self.create_icon(self.current_model)

    def toggle_availability(self, icon=None, item=None):
        with self._avail_lock:
            is_temporarily_unavailable = self.unavailable_until and datetime.now() < self.unavailable_until
            is_currently_available = self.available and not is_temporarily_unavailable
        if is_currently_available:
            self.set_unavailable()
        else:
            self.set_available()

    # --- START OF HEAVY DEBUGGING SECTION ---
    async def update_status(self):
        print(f"[{debug_ts()}] [LOOP] Starting update_status async loop.")
        async with aiohttp.ClientSession() as session:
            while self.should_run:
                print(f"[{debug_ts()}] [LOOP] >> New cycle started. Previous status: '{self.last_status}'")
                basic_status = ""
                try:
                    # Make the API call using aiohttp session
                    print(f"[{debug_ts()}] [AIOHTTP] Making GET request to {self.api_url}/api/ps...")
                    async with session.get(f'{self.api_url}/api/ps', timeout=2) as response:
                        print(f"[{debug_ts()}] [AIOHTTP] Received response with status code: {response.status}")
                        response.raise_for_status()
                        data = await response.json()
                        print(f"[{debug_ts()}] [AIOHTTP] JSON response data: {data}")
                        running_models = data.get('models', [])
                        
                        if running_models:
                            model = running_models[0]
                            basic_status = f"{model['name']} ({model['details']['parameter_size']})"
                        else:
                            basic_status = "No Model Running"
                        print(f"[{debug_ts()}] [AIOHTTP] Determined basic_status: '{basic_status}'")

                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                    print(f"[{debug_ts()}] [AIOHTTP-ERROR] Connection/Timeout error: {e}")
                    basic_status = "Ollama Not Running"
                except Exception as e:
                    print(f"[{debug_ts()}] [AIOHTTP-ERROR] Unexpected error during request: {e}")
                    self.logger.error(f"Error in aiohttp update loop: {e}", exc_info=True)
                    basic_status = "Ollama Not Running"
                
                # --- The original update logic now follows, with its own debugging ---
                try:
                    if basic_status != self.last_status:
                        print(f"[{debug_ts()}] [STATE-CHANGE] Status changed from '{self.last_status}' to '{basic_status}'")
                        self.logger.info(f"Status changed from '{self.last_status}' to '{basic_status}'")
                        
                        is_model_running = "No Model Running" not in basic_status and "Ollama Not Running" not in basic_status
                        
                        if is_model_running and self.icon:
                            print(f"[{debug_ts()}] [NOTIFY] Notifying: Model is running.")
                            self.icon.notify(basic_status)
                        elif "No Model Running" == basic_status and self.icon:
                            print(f"[{debug_ts()}] [NOTIFY] Notifying: Model Stopped.")
                            self.icon.notify("Model Stopped")
                        elif "Ollama Not Running" == basic_status and self.icon:
                            print(f"[{debug_ts()}] [NOTIFY] Notifying: Ollama Service Stopped.")
                            self.icon.notify("Ollama Service Stopped")
                        
                        self.last_status = basic_status
                        self.last_detailed_fetch_time = 0
                    else:
                        print(f"[{debug_ts()}] [STATE-SAME] Status '{basic_status}' has not changed.")

                    self.current_model = basic_status
                    
                    hover_text = f"Ollama: {basic_status}" # Default hover text
                    try:
                        # This block is for building the detailed hover text
                        is_model_running = "No Model Running" not in basic_status and "Ollama Not Running" not in basic_status
                        if is_model_running:
                            system_health = await self.get_system_health()
                            # (Your detailed hover text logic is preserved here)
                            if self.detailed_model_data and "name" in self.detailed_model_data:
                                details = self.detailed_model_data
                                hover_text = f"{details['name']}\nProcessor: {details['processor']} | Expires: {details['expiry']}"
                            gpu_text = "GPU: N/A"
                            if system_health.get('utilization_gpu_percent') is not None:
                                gpu_util = system_health['utilization_gpu_percent']
                                vram_used = system_health.get('vram_used_mb', 0)
                                vram_total = system_health.get('vram_total_mb', 0)
                                gpu_text = f"GPU: {gpu_util}% | VRAM: {vram_used}/{vram_total} MB"
                            hover_text += f"\n{gpu_text}"

                        if not self.available: # Check overall availability
                             hover_text = "MANUALLY UNAVAILABLE\n" + hover_text

                    except Exception as e:
                        print(f"[{debug_ts()}] [ERROR] Failed to build hover text: {e}")
                        hover_text = "Error building hover text"


                    if self.icon:
                        print(f"[{debug_ts()}] [UI-UPDATE] Preparing to update pystray icon and menu.")
                        self.icon.icon = self.create_icon(basic_status)
                        self.icon.title = (hover_text[:124] + '...') if len(hover_text) > 127 else hover_text
                        self.icon.menu = self.create_menu()
                        print(f"[{debug_ts()}] [UI-UPDATE] pystray properties updated.")

                except Exception as e:
                    print(f"[{debug_ts()}] [FATAL-UI] FATAL ERROR in update_status loop's UI section: {e}")
                    self.logger.error(f"FATAL ERROR in update_status loop's UI section: {e}", exc_info=True)

                print(f"[{debug_ts()}] [LOOP] << Cycle finished. Sleeping for 2 seconds.")
                await asyncio.sleep(2)
        print(f"[{debug_ts()}] [LOOP] Loop has exited because should_run is False.")
    # --- END OF HEAVY DEBUGGING SECTION ---

    def show_settings(self):
        print(f"[{debug_ts()}] [ACTION] Show settings window requested.")
        settings_thread = threading.Thread(target=lambda: SettingsWindow(self))
        settings_thread.daemon = True
        settings_thread.start()

    def start_polling(self):
        print(f"[{debug_ts()}] [INIT] Starting polling thread.")
        def runner():
            print(f"[{debug_ts()}] [RUNNER] Polling thread started. Running asyncio event loop.")
            try:
                asyncio.run(self.update_status())
            except Exception as e:
                print(f"[{debug_ts()}] [RUNNER-FATAL] Asyncio runner crashed: {e}")
            print(f"[{debug_ts()}] [RUNNER] Asyncio event loop has finished.")
        t = threading.Thread(target=runner, daemon=True)
        t.start()

    @property
    def api_url(self) -> str:
        return self.settings.get('api_url', f'http://{self.DEFAULT_API_HOST}:{self.DEFAULT_API_PORT}')
    
    @property
    def comfyui_url(self) -> str:
        return self.settings.get('comfyui_url', self.DEFAULT_COMFYUI_URL)

    @property
    def api_port(self) -> int:
        return self.settings.get('api_port', self.DEFAULT_HEALTH_API_PORT)

if __name__ == "__main__":
    multiprocessing.freeze_support()

    log_dir = os.path.join(os.getenv('APPDATA'), 'OllamaMonitor', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    output_log_path = os.path.join(log_dir, 'debug_output.txt')
    
    if os.path.exists(output_log_path):
        try:
            os.remove(output_log_path)
        except OSError:
            output_log_path = os.path.join(log_dir, f'debug_output_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt')

    try:
        # This context manager ensures the file is closed properly
        with open(output_log_path, 'w', encoding='utf-8', buffering=1) as log_file: # buffering=1 means line-buffered
            sys.stdout = log_file
            sys.stderr = log_file

            print(f"--- Debug session started at {datetime.now()} ---")
            
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

            monitor = OllamaMonitor()
            monitor_instance = monitor

            if monitor.settings.get('health_api_enabled'):
                api_thread = threading.Thread(target=monitor.run_api_server)
                api_thread.daemon = True
                api_thread.start()
            else:
                monitor.logger.info("Health API server is disabled in settings.")

            monitor.start_polling()

            monitor.icon = pystray.Icon("ollama-monitor", monitor.create_icon("Starting..."), "Ollama Monitor", menu=monitor.create_menu())
            print(f"[{debug_ts()}] [MAIN] pystray icon created. Calling icon.run()...")
            monitor.icon.run()
            print(f"[{debug_ts()}] [MAIN] pystray icon.run() has exited.")

    except (KeyboardInterrupt, SystemExit):
        print(f"[{debug_ts()}] [MAIN-EXIT] Shutdown requested via KeyboardInterrupt/SystemExit.")
    except Exception as e:
        print(f"\n[{debug_ts()}] [MAIN-FATAL] --- A FATAL EXCEPTION OCCURRED ---")
        import traceback
        traceback.print_exc()
        time.sleep(10)
    finally:
        if 'monitor' in locals() and monitor:
            monitor.stop()
        print(f"[{debug_ts()}] [MAIN-EXIT] Application has finished shutting down.")