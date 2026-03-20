package com.example.breathheartdemo

import android.Manifest
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import android.os.ParcelUuid
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.UUID

data class BleDevice(
    val name: String?,
    val address: String,
    val rssi: Int,
    val device: BluetoothDevice
)

sealed class ConnectionState {
    data object Disconnected : ConnectionState()
    data object Scanning : ConnectionState()
    data object Connecting : ConnectionState()
    data class Connected(val device: BleDevice) : ConnectionState()
}

class BleClient(
    private val context: Context,
    private val sampleRateHz: Int
) {
    private val logTag = "BleClient"
    private val bluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val adapter = bluetoothManager.adapter
    private val scanner get() = adapter?.bluetoothLeScanner

    private val deviceMap = LinkedHashMap<String, BleDevice>()

    private var gatt: BluetoothGatt? = null
    private var currentDevice: BleDevice? = null
    private var scanning = false

    private val _scanResults = MutableStateFlow<List<BleDevice>>(emptyList())
    val scanResults: StateFlow<List<BleDevice>> = _scanResults

    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _rawText = MutableStateFlow("")
    val rawText: StateFlow<String> = _rawText

    private val _rawHex = MutableStateFlow("")
    val rawHex: StateFlow<String> = _rawHex

    private val _raw16Preview = MutableStateFlow("")
    val raw16Preview: StateFlow<String> = _raw16Preview

    private val _rawBytes = MutableStateFlow(0L)
    val rawBytes: StateFlow<Long> = _rawBytes

    private val _samples = MutableSharedFlow<Sample>(extraBufferCapacity = 256)
    val samples: SharedFlow<Sample> = _samples

    private val _connectionError = MutableStateFlow<String?>(null)
    val connectionError: StateFlow<String?> = _connectionError

    private val _lastRxMs = MutableStateFlow<Long?>(null)
    val lastRxMs: StateFlow<Long?> = _lastRxMs

    private var pendingByte: Int? = null
    private var lastSampleMs: Double? = null
    private val samplePeriodMs = if (sampleRateHz > 0) 1000.0 / sampleRateHz else 10.0

    fun clearPreview() {
        _rawText.value = ""
        _rawHex.value = ""
        _raw16Preview.value = ""
        _rawBytes.value = 0L
    }

    private val scanCallback = object : ScanCallback() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            val device = result.device ?: return
            val safeName = if (hasConnectPermission()) device.name else null
            val entry = BleDevice(
                name = safeName,
                address = device.address,
                rssi = result.rssi,
                device = device
            )
            deviceMap[device.address] = entry
            _scanResults.value = deviceMap.values.toList()
        }

        override fun onScanFailed(errorCode: Int) {
            _connectionError.value = "Scan failed: $errorCode"
            _connectionState.value = ConnectionState.Disconnected
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                _connectionError.value = "Connection failed: $status"
                closeGatt()
                _connectionState.value = ConnectionState.Disconnected
                return
            }

            if (newState == BluetoothProfile.STATE_CONNECTED) {
                _connectionState.value = ConnectionState.Connecting
                gatt.discoverServices()
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                closeGatt()
                _connectionState.value = ConnectionState.Disconnected
            }
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                _connectionError.value = "Service discovery failed: $status"
                gatt.disconnect()
                return
            }
            val service = gatt.getService(SERVICE_UUID)
            if (service == null) {
                _connectionError.value = "Service not found: $SERVICE_UUID"
                gatt.disconnect()
                return
            }
            val characteristic = service.getCharacteristic(DATA_CHAR_UUID)
            if (characteristic == null) {
                _connectionError.value = "Characteristic not found: $DATA_CHAR_UUID"
                gatt.disconnect()
                return
            }
            val ok = gatt.setCharacteristicNotification(characteristic, true)
            if (!ok) {
                _connectionError.value = "Failed to enable notifications"
                gatt.disconnect()
                return
            }
            val cccd = characteristic.getDescriptor(CCCD_UUID)
            if (cccd == null) {
                _connectionError.value = "Notification descriptor not found"
                gatt.disconnect()
                return
            }
            val writeOk = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                gatt.writeDescriptor(
                    cccd,
                    BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                ) == BluetoothStatusCodes.SUCCESS
            } else {
                @Suppress("DEPRECATION")
                run {
                    cccd.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                    gatt.writeDescriptor(cccd)
                }
            }
            if (!writeOk) {
                _connectionError.value = "Failed to write notification descriptor"
                gatt.disconnect()
            }
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onDescriptorWrite(
            gatt: BluetoothGatt,
            descriptor: BluetoothGattDescriptor,
            status: Int
        ) {
            if (descriptor.uuid != CCCD_UUID) return
            if (status != BluetoothGatt.GATT_SUCCESS) {
                _connectionError.value = "Notification subscription failed: $status"
                gatt.disconnect()
                return
            }
            val device = currentDevice ?: BleDevice(
                name = gatt.device.name,
                address = gatt.device.address,
                rssi = 0,
                device = gatt.device
            )
            _connectionState.value = ConnectionState.Connected(device)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic
        ) {
            handleIncoming(characteristic.value)
        }

        override fun onCharacteristicChanged(
            gatt: BluetoothGatt,
            characteristic: BluetoothGattCharacteristic,
            value: ByteArray
        ) {
            handleIncoming(value)
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    fun startScan() {
        if (!hasScanPermission()) return
        if (adapter?.isEnabled != true) return
        stopScan()
        deviceMap.clear()
        _scanResults.value = emptyList()
        _connectionState.value = ConnectionState.Scanning
        val filter = ScanFilter.Builder()
            .setServiceUuid(ParcelUuid(SERVICE_UUID))
            .build()
        val settings = ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build()
        scanner?.startScan(listOf(filter), settings, scanCallback)
        scanning = true
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    fun stopScan() {
        if (!hasScanPermission()) return
        if (scanning) {
            scanner?.stopScan(scanCallback)
            scanning = false
        }
        if (_connectionState.value is ConnectionState.Scanning) {
            _connectionState.value = ConnectionState.Disconnected
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    fun connect(device: BleDevice) {
        if (!hasConnectPermission()) return
        if (hasScanPermission()) {
            stopScan()
        }
        disconnect()
        currentDevice = device
        _connectionState.value = ConnectionState.Connecting
        _connectionError.value = null
        resetSampleClock()
        gatt = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            device.device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE)
        } else {
            device.device.connectGatt(context, false, gattCallback)
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    fun disconnect() {
        if (!hasConnectPermission()) {
            closeGatt()
            _connectionState.value = ConnectionState.Disconnected
            return
        }
        closeGatt()
        _connectionState.value = ConnectionState.Disconnected
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    private fun closeGatt() {
        gatt?.close()
        gatt = null
        currentDevice = null
        pendingByte = null
        lastSampleMs = null
    }

    @RequiresPermission(
        anyOf = [
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.ACCESS_FINE_LOCATION
        ],
        conditional = true
    )
    private fun hasScanPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            hasPermission(Manifest.permission.BLUETOOTH_SCAN)
        } else {
            hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT, conditional = true)
    private fun hasConnectPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            hasPermission(Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            true
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(context, permission) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED
    }

    private fun handleIncoming(bytes: ByteArray) {
        if (bytes.isEmpty()) return
        val preview = buildPreview(bytes, bytes.size)
        _rawText.value = appendPreview(_rawText.value, preview, 2048)
        val hex = buildHex(bytes, bytes.size)
        val line = "RX(${bytes.size}): $hex\n"
        _rawHex.value = appendPreview(_rawHex.value, line, 4096)
        Log.d(logTag, "RX(${bytes.size}): $hex")
        _rawBytes.value = _rawBytes.value + bytes.size
        _lastRxMs.value = System.currentTimeMillis()

        ensureSampleClock()
        var idx = 0
        val firstPending = pendingByte
        val leValues = ArrayList<Int>(bytes.size / 2 + 1)
        val beValues = ArrayList<Int>(bytes.size / 2 + 1)
        if (firstPending != null) {
            val lo = firstPending
            val hi = bytes[0].toInt() and 0xFF
            leValues.add((hi shl 8) or lo)
            beValues.add((lo shl 8) or hi)
            emitSample(lo, hi)
            idx = 1
            pendingByte = null
        }
        while (idx + 1 < bytes.size) {
            val lo = bytes[idx].toInt() and 0xFF
            val hi = bytes[idx + 1].toInt() and 0xFF
            leValues.add((hi shl 8) or lo)
            beValues.add((lo shl 8) or hi)
            emitSample(lo, hi)
            idx += 2
        }
        if (idx < bytes.size) {
            pendingByte = bytes[idx].toInt() and 0xFF
        }

        if (leValues.isNotEmpty()) {
            val preview16 = build16Preview(leValues, beValues, 24)
            _raw16Preview.value = appendPreview(_raw16Preview.value, preview16, 2048)
        }
    }

    private fun ensureSampleClock() {
        val now = System.currentTimeMillis().toDouble()
        val last = lastSampleMs
        if (last == null || now - last > 2000.0) {
            lastSampleMs = now
        }
    }

    private fun emitSample(lo: Int, hi: Int) {
        val raw = (hi shl 8) or lo
        val signed = raw.toShort().toInt()
        val t = lastSampleMs?.toLong() ?: System.currentTimeMillis()
        _samples.tryEmit(Sample(t, signed.toFloat()))
        lastSampleMs = (lastSampleMs ?: t.toDouble()) + samplePeriodMs
    }

    private fun resetSampleClock() {
        lastSampleMs = null
        pendingByte = null
    }

    private fun buildPreview(buf: ByteArray, len: Int): String {
        val sb = StringBuilder(len)
        for (i in 0 until len) {
            val b = buf[i].toInt() and 0xFF
            val ch = b.toChar()
            if (b == 0x0A || b == 0x0D || b == 0x09) {
                sb.append(ch)
            } else if (b in 0x20..0x7E) {
                sb.append(ch)
            } else {
                sb.append('.')
            }
        }
        return sb.toString()
    }

    private fun appendPreview(current: String, chunk: String, maxLen: Int): String {
        if (current.isEmpty()) return chunk.takeLast(maxLen)
        val combined = current + chunk
        return if (combined.length <= maxLen) combined else combined.takeLast(maxLen)
    }

    private fun buildHex(buf: ByteArray, len: Int): String {
        val sb = StringBuilder(len * 3)
        for (i in 0 until len) {
            val b = buf[i].toInt() and 0xFF
            if (sb.isNotEmpty()) sb.append(' ')
            sb.append("0123456789ABCDEF"[b ushr 4])
            sb.append("0123456789ABCDEF"[b and 0x0F])
        }
        return sb.toString()
    }

    private fun build16Preview(leValues: List<Int>, beValues: List<Int>, maxItems: Int): String {
        val count = minOf(leValues.size, maxItems)
        val sb = StringBuilder(count * 8 * 4)
        sb.append("LE u16(hex): ")
        for (i in 0 until count) {
            if (i > 0) sb.append(' ')
            sb.append(String.format("%04X", leValues[i]))
        }
        sb.append('\n')
        sb.append("LE s16(dec): ")
        for (i in 0 until count) {
            if (i > 0) sb.append(' ')
            sb.append(leValues[i].toShort().toInt())
        }
        sb.append('\n')
        sb.append("BE u16(hex): ")
        for (i in 0 until count) {
            if (i > 0) sb.append(' ')
            sb.append(String.format("%04X", beValues[i]))
        }
        sb.append('\n')
        sb.append("BE s16(dec): ")
        for (i in 0 until count) {
            if (i > 0) sb.append(' ')
            sb.append(beValues[i].toShort().toInt())
        }
        sb.append('\n')
        return sb.toString()
    }

    companion object {
        val SERVICE_UUID: UUID =
            UUID.fromString("0000abf0-0000-1000-8000-00805f9b34fb")
        val DATA_CHAR_UUID: UUID =
            UUID.fromString("0000abf2-0000-1000-8000-00805f9b34fb")
        val CCCD_UUID: UUID =
            UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")
    }
}
