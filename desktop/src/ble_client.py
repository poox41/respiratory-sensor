from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from bleak import BleakClient, BleakScanner
from PySide2.QtCore import QObject, Signal

from config import DATA_CHAR_UUID, DEFAULT_SAMPLE_RATE_HZ, SERVICE_UUID
from models import Sample


@dataclass
class BleDevice:
    name: str | None
    address: str
    rssi: int | None
    device: object | None = None


class BleWorker(QObject):
    devices_updated = Signal(list)  # list[BleDevice]
    connection_changed = Signal(bool)
    sample_received = Signal(object)  # Sample
    raw_hex = Signal(str)
    error = Signal(str)

    def __init__(self, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ) -> None:
        super().__init__()
        self.sample_rate_hz = sample_rate_hz
        self.sample_period_ms = 1000.0 / sample_rate_hz

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._scanner: BleakScanner | None = None
        self._client: BleakClient | None = None
        self._devices: Dict[str, BleDevice] = {}
        self._notify_buf = bytearray()
        self._last_sample_ms: float | None = None
        self._log_enabled = True
        self._sample_count = 0

    def _log(self, msg: str) -> None:
        if self._log_enabled:
            print(f"[BLE] {msg}", flush=True)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def start_scan(self) -> None:
        self._submit(self._start_scan())

    def stop_scan(self) -> None:
        self._submit(self._stop_scan())

    def connect_device(self, address: str) -> None:
        self._submit(self._connect(address))

    def disconnect_device(self) -> None:
        self._submit(self._disconnect())

    async def _start_scan(self) -> None:
        if self._scanner is not None:
            return

        def on_detect(device, adv_data):
            if SERVICE_UUID not in (adv_data.service_uuids or []):
                return
            entry = BleDevice(
                name=device.name,
                address=device.address,
                rssi=device.rssi,
                device=device,
            )
            self._devices[device.address] = entry
            self._log(f"found {entry.name or 'Unknown'} {entry.address} rssi={entry.rssi}")
            self.devices_updated.emit(list(self._devices.values()))

        self._devices.clear()
        self._log("scan start")
        self._scanner = BleakScanner(detection_callback=on_detect)
        await self._scanner.start()

    async def _stop_scan(self) -> None:
        if self._scanner is None:
            return
        await self._scanner.stop()
        self._scanner = None
        self._log("scan stop")

    async def _connect(self, address: str) -> None:
        self._log(f"connect attempt {address}")
        await self._disconnect()
        self._notify_buf.clear()
        self._last_sample_ms = None
        entry = self._devices.get(address)
        if entry is not None and entry.device is not None:
            self._log("connect using Bleak device instance")
            self._client = BleakClient(entry.device)
        else:
            self._log("connect using address string")
            self._client = BleakClient(address)
        try:
            await self._client.connect()
            try:
                if hasattr(self._client, "pair"):
                    paired = await self._client.pair()
                    self._log(f"pair result: {paired}")
            except Exception as e:
                self._log(f"pair failed: {e!r}")

            await self._client.get_services()
            services = self._client.services
            self._log("services:")
            for svc in services:
                self._log(f"  svc {svc.uuid}")
                for ch in svc.characteristics:
                    props = ",".join(ch.properties)
                    self._log(f"    char {ch.uuid} props={props}")

            char = services.get_characteristic(DATA_CHAR_UUID)
            if char is None:
                raise RuntimeError(f"DATA_CHAR_UUID not found: {DATA_CHAR_UUID}")
            if "notify" not in char.properties:
                raise RuntimeError(
                    f"Characteristic {DATA_CHAR_UUID} has no notify property: {char.properties}"
                )

            await self._client.start_notify(DATA_CHAR_UUID, self._on_notify)
            self.connection_changed.emit(True)
            self._log(f"connect ok {address}")
        except Exception as e:
            self._log(f"connect failed {address}: {e!r}")
            self.error.emit(f"BLE connect failed: {e}")
            await self._disconnect()

    async def _disconnect(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.stop_notify(DATA_CHAR_UUID)
        except Exception:
            pass
        try:
            await self._client.disconnect()
        except Exception:
            pass
        self._client = None
        self.connection_changed.emit(False)
        self._log("disconnect")

    def _on_notify(self, _sender: int, data: bytearray) -> None:
        if not data:
            return
        self._notify_buf.extend(data)
        hex_str = " ".join(f"{b:02x}" for b in data)
        self.raw_hex.emit(hex_str)

        # parse byte stream: every 2 bytes -> one little-endian signed sample
        while len(self._notify_buf) >= 2:
            lo = self._notify_buf[0]
            hi = self._notify_buf[1]
            del self._notify_buf[:2]

            sample_signed = int.from_bytes(bytes([lo, hi]), "little", signed=True)

            now = time.time() * 1000.0
            if self._last_sample_ms is None:
                self._last_sample_ms = now
            t_ms = int(self._last_sample_ms)
            self._last_sample_ms += self.sample_period_ms

            sample = Sample(t_ms=t_ms, x=float(sample_signed))
            self._sample_count += 1
            if self._sample_count % 50 == 0:
                print(f"[BLE] sample#{self._sample_count} t_ms={sample.t_ms} x={sample.x}", flush=True)
            self.sample_received.emit(sample)
