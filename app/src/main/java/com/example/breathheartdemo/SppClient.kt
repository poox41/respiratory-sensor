package com.example.breathheartdemo

import android.Manifest
import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.IOException
import java.util.UUID

data class SppDevice(
    val name: String?,
    val address: String,
    val rssi: Int,
    val device: BluetoothDevice
)

sealed class ConnectionState {
    data object Disconnected : ConnectionState()
    data object Scanning : ConnectionState()
    data object Connecting : ConnectionState()
    data class Connected(val device: SppDevice) : ConnectionState()
}

class SppClient(private val context: Context) {
    private val bluetoothManager =
        context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
    private val adapter = bluetoothManager.adapter

    private val deviceMap = LinkedHashMap<String, SppDevice>()

    private var socket: BluetoothSocket? = null
    private var discoveryReceiverRegistered = false
    @Volatile private var reading = false
    private var readerThread: Thread? = null
    private var bondReceiverRegistered = false
    private var pendingDevice: SppDevice? = null

    private val _scanResults = MutableStateFlow<List<SppDevice>>(emptyList())
    val scanResults: StateFlow<List<SppDevice>> = _scanResults

    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState

    private val _rawText = MutableStateFlow("")
    val rawText: StateFlow<String> = _rawText

    private val _rawHex = MutableStateFlow("")
    val rawHex: StateFlow<String> = _rawHex

    private val _rawBytes = MutableStateFlow(0L)
    val rawBytes: StateFlow<Long> = _rawBytes

    private val _samples = MutableSharedFlow<Sample>(extraBufferCapacity = 256)
    val samples: SharedFlow<Sample> = _samples

    private val _connectionError = MutableStateFlow<String?>(null)
    val connectionError: StateFlow<String?> = _connectionError

    private val _lastRxMs = MutableStateFlow<Long?>(null)
    val lastRxMs: StateFlow<Long?> = _lastRxMs

    fun clearPreview() {
        _rawText.value = ""
        _rawHex.value = ""
        _rawBytes.value = 0L
    }

    private val discoveryReceiver = object : BroadcastReceiver() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                BluetoothDevice.ACTION_FOUND -> {
                    val device = getDeviceFromIntent(intent) ?: return
                    val safeName = if (hasConnectPermission()) device.name else null
                    val rssi = intent.getShortExtra(BluetoothDevice.EXTRA_RSSI, 0).toInt()
                    val entry = SppDevice(
                        name = safeName,
                        address = device.address,
                        rssi = rssi,
                        device = device
                    )
                    deviceMap[device.address] = entry
                    _scanResults.value = deviceMap.values.toList()
                }
                BluetoothAdapter.ACTION_DISCOVERY_FINISHED -> {
                    if (_connectionState.value is ConnectionState.Scanning) {
                        _connectionState.value = ConnectionState.Disconnected
                    }
                }
            }
        }
    }

    private val bondReceiver = object : BroadcastReceiver() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != BluetoothDevice.ACTION_BOND_STATE_CHANGED) return
            val device = getDeviceFromIntent(intent) ?: return
            val target = pendingDevice ?: return
            if (device.address != target.address) return

            val state = intent.getIntExtra(BluetoothDevice.EXTRA_BOND_STATE, BluetoothDevice.BOND_NONE)
            val prev = intent.getIntExtra(BluetoothDevice.EXTRA_PREVIOUS_BOND_STATE, BluetoothDevice.BOND_NONE)
            if (state == BluetoothDevice.BOND_BONDED) {
                unregisterBondReceiver()
                pendingDevice = null
                connect(target)
            } else if (state == BluetoothDevice.BOND_NONE && prev == BluetoothDevice.BOND_BONDING) {
                unregisterBondReceiver()
                pendingDevice = null
                _connectionState.value = ConnectionState.Disconnected
            }
        }
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    fun startScan() {
        if (!hasScanPermission()) return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && !hasConnectPermission()) return
        if (adapter?.isEnabled != true) return
        deviceMap.clear()
        _scanResults.value = emptyList()
        _connectionState.value = ConnectionState.Scanning
        registerDiscoveryReceiver()
        adapter?.startDiscovery()
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_SCAN)
    fun stopScan() {
        if (!hasScanPermission()) return
        adapter?.cancelDiscovery()
        if (_connectionState.value is ConnectionState.Scanning) {
            _connectionState.value = ConnectionState.Disconnected
        }
        unregisterDiscoveryReceiver()
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    fun connect(device: SppDevice, retries: Int = 2, retryDelayMs: Long = 800L) {
        if (!hasConnectPermission()) return
        if (hasScanPermission()) {
            stopScan()
        }
        if (device.device.bondState != BluetoothDevice.BOND_BONDED) {
            pendingDevice = device
            registerBondReceiver()
            _connectionState.value = ConnectionState.Connecting
            _connectionError.value = null
            device.device.createBond()
            return
        }
        _connectionState.value = ConnectionState.Connecting
        _connectionError.value = null
        adapter?.cancelDiscovery()
        Thread {
            val maxAttempts = maxOf(1, retries + 1)
            for (attempt in 1..maxAttempts) {
                val tmpSocket = device.device.createRfcommSocketToServiceRecord(SPP_UUID)
                try {
                    tmpSocket.connect()
                    socket = tmpSocket
                    _connectionState.value = ConnectionState.Connected(device)
                    _connectionError.value = null
                    startReader(tmpSocket)
                    return@Thread
                } catch (e: IOException) {
                    safeClose(tmpSocket)
                    val msg = e.message ?: e.javaClass.simpleName
                    _connectionError.value = "连接失败($attempt/$maxAttempts): $msg"
                    if (attempt < maxAttempts) {
                        try {
                            Thread.sleep(retryDelayMs)
                        } catch (_: InterruptedException) {
                        }
                    }
                }
            }
            _connectionState.value = ConnectionState.Disconnected
        }.start()
    }

    @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
    fun disconnect() {
        if (!hasConnectPermission()) {
            socket = null
            _connectionState.value = ConnectionState.Disconnected
            return
        }
        unregisterBondReceiver()
        pendingDevice = null
        stopReader()
        safeClose(socket)
        socket = null
        _connectionState.value = ConnectionState.Disconnected
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
            PackageManager.PERMISSION_GRANTED
    }

    private fun registerDiscoveryReceiver() {
        if (discoveryReceiverRegistered) return
        val filter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_FOUND)
            addAction(BluetoothAdapter.ACTION_DISCOVERY_FINISHED)
        }
        context.registerReceiver(discoveryReceiver, filter)
        discoveryReceiverRegistered = true
    }

    private fun unregisterDiscoveryReceiver() {
        if (!discoveryReceiverRegistered) return
        context.unregisterReceiver(discoveryReceiver)
        discoveryReceiverRegistered = false
    }

    private fun registerBondReceiver() {
        if (bondReceiverRegistered) return
        val filter = IntentFilter().apply {
            addAction(BluetoothDevice.ACTION_BOND_STATE_CHANGED)
        }
        context.registerReceiver(bondReceiver, filter)
        bondReceiverRegistered = true
    }

    private fun unregisterBondReceiver() {
        if (!bondReceiverRegistered) return
        context.unregisterReceiver(bondReceiver)
        bondReceiverRegistered = false
    }

    private fun safeClose(target: BluetoothSocket?) {
        try {
            target?.close()
        } catch (_: IOException) {
        }
    }

    private fun startReader(target: BluetoothSocket) {
        stopReader()
        reading = true
        readerThread = Thread {
            val input = try {
                target.inputStream
            } catch (_: IOException) {
                reading = false
                return@Thread
            }

            val buf = ByteArray(1024)
            val lineBuf = StringBuilder() // 累积字节，拼出一行文本
            while (reading) {
                val n = try {
                    input.read(buf)
                } catch (_: IOException) {
                    break
                }
                if (n <= 0) break
                // 仅用于预览显示，不影响解析流程。
                val preview = buildPreview(buf, n)
                _rawText.value = appendPreview(_rawText.value, preview, 2048)
                val hex = buildHex(buf, n)
                _rawHex.value = appendPreview(_rawHex.value, hex, 4096)
                _rawBytes.value = _rawBytes.value + n
                _lastRxMs.value = System.currentTimeMillis()
                // 真实解析流程：按行切分并转成浮点样本。
                parseSamples(buf, n, lineBuf)
            }
            reading = false
        }.apply { start() }
    }

    private fun stopReader() {
        reading = false
        readerThread = null
    }

    private fun buildPreview(buf: ByteArray, len: Int): String {
        val sb = StringBuilder(len)
        for (i in 0 until len) {
            val b = buf[i].toInt() and 0xFF
            val ch = b.toChar()
            if (b == 0x0A || b == 0x0D || b == 0x09) {
                sb.append(ch) // keep \n \r \t
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

    private fun parseSamples(buf: ByteArray, len: Int, lineBuf: StringBuilder) {
        for (i in 0 until len) {
            val b = buf[i].toInt() and 0xFF
            val ch = b.toChar()
            if (ch == '\n' || ch == '\r') {
                if (lineBuf.isNotEmpty()) {
                    val line = lineBuf.toString().trim()
                    lineBuf.setLength(0)
                    val parts = line.split(',')
                    if (parts.size >= 2) {
                        val tSec = parts[0].toFloatOrNull()
                        val v = parts[1].toFloatOrNull()
                        if (tSec != null && v != null) {
                            val tMs = (tSec * 1000f).toLong()
                            _samples.tryEmit(Sample(tMs, v))
                        }
                    } else {
                        // 兼容旧格式：只有数值时仍然用当前时间
                        val v = line.toFloatOrNull()
                        if (v != null) {
                            _samples.tryEmit(Sample(System.currentTimeMillis(), v))
                        }
                    }
                }
            } else {
                lineBuf.append(ch)
            }
        }
        if (lineBuf.length > 64) {
            // 防止行异常过长导致内存占用持续增长。
            lineBuf.setLength(0)
        }
    }

    @SuppressLint("MissingPermission")
    private fun getDeviceFromIntent(intent: Intent): BluetoothDevice? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE, BluetoothDevice::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(BluetoothDevice.EXTRA_DEVICE)
        }
    }

    companion object {
        private val SPP_UUID: UUID =
            UUID.fromString("00001101-0000-1000-8000-00805F9B34FB")
    }
}
