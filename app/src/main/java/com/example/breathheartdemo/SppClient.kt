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
import kotlinx.coroutines.flow.MutableStateFlow
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
    fun connect(device: SppDevice) {
        if (!hasConnectPermission()) return
        if (hasScanPermission()) {
            stopScan()
        }
        if (device.device.bondState != BluetoothDevice.BOND_BONDED) {
            pendingDevice = device
            registerBondReceiver()
            _connectionState.value = ConnectionState.Connecting
            device.device.createBond()
            return
        }
        _connectionState.value = ConnectionState.Connecting
        adapter?.cancelDiscovery()
        val tmpSocket = device.device.createRfcommSocketToServiceRecord(SPP_UUID)
        Thread {
            try {
                tmpSocket.connect()
                socket = tmpSocket
                _connectionState.value = ConnectionState.Connected(device)
                startReader(tmpSocket)
            } catch (e: IOException) {
                safeClose(tmpSocket)
                _connectionState.value = ConnectionState.Disconnected
            }
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
            while (reading) {
                val n = try {
                    input.read(buf)
                } catch (_: IOException) {
                    break
                }
                if (n <= 0) break
                val preview = buildPreview(buf, n)
                _rawText.value = appendPreview(_rawText.value, preview, 2048)
                val hex = buildHex(buf, n)
                _rawHex.value = appendPreview(_rawHex.value, hex, 4096)
                _rawBytes.value = _rawBytes.value + n
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
