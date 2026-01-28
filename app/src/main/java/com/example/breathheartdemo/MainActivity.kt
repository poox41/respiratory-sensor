package com.example.breathheartdemo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme { AppScreen() }
        }
    }
}

@Composable
@OptIn(ExperimentalMaterial3Api::class)
fun AppScreen() {
    val fsHz = 100
    val context = LocalContext.current
    val processor = remember { Processor(fsHz) }
    val rates by processor.rates.collectAsState()

    var useMock by remember { mutableStateOf(true) }
    var showBleDialog by remember { mutableStateOf(false) }

    val sppClient = remember { SppClient(context.applicationContext) }
    val devices by sppClient.scanResults.collectAsState()
    val connectionState by sppClient.connectionState.collectAsState()
    val rawText by sppClient.rawText.collectAsState()
    val rawHex by sppClient.rawHex.collectAsState()
    val rawBytes by sppClient.rawBytes.collectAsState()
    val connectionError by sppClient.connectionError.collectAsState()
    val lastRxMs by sppClient.lastRxMs.collectAsState()
    val timeFormatter = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }

    val scope = rememberCoroutineScope()
    var job by remember { mutableStateOf<Job?>(null) }

    var pendingConnect by remember { mutableStateOf(false) }
    var connectionMessage by remember { mutableStateOf<String?>(null) }
    var lastConnectionState by remember { mutableStateOf<ConnectionState>(ConnectionState.Disconnected) }

    var hasBlePermissions by remember { mutableStateOf(hasBlePermissions(context)) }
    val previewScroll = rememberScrollState()

    val permissionLauncher =
        rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            hasBlePermissions = result.values.all { it }
        }

    val requestBlePermissions = {
        permissionLauncher.launch(requiredBlePermissions())
    }

    LaunchedEffect(useMock) {
        job?.cancel()
        if (useMock) {
            job = scope.launch {
                MockDataSource(fsHz = fsHz).samples().collect { s ->
                    processor.onSample(s)
                }
            }
            if (hasBleScanPermission(context)) {
                sppClient.stopScan()
            }
        } else {
            job = scope.launch {
                sppClient.samples.collect { s ->
                    processor.onSample(s)
                }
            }
            if (!hasBlePermissions) {
                requestBlePermissions()
            }
        }
    }

    LaunchedEffect(showBleDialog, hasBlePermissions) {
        if (showBleDialog && !hasBlePermissions) {
            requestBlePermissions()
        }
    }

    LaunchedEffect(useMock, hasBlePermissions) {
        if (!useMock && hasBleScanPermission(context)) {
            sppClient.startScan()
        }
    }

    LaunchedEffect(connectionState, pendingConnect) {
        if (pendingConnect) {
            when (connectionState) {
                is ConnectionState.Connected -> {
                    connectionMessage = "连接成功"
                    showBleDialog = false
                    pendingConnect = false
                }
                ConnectionState.Disconnected -> {
                    connectionMessage = "连接失败"
                    showBleDialog = false
                    pendingConnect = false
                }
                else -> Unit
            }
        }
        if (connectionState is ConnectionState.Connected &&
            lastConnectionState !is ConnectionState.Connected
        ) {
            processor.reset()
        }
        lastConnectionState = connectionState
    }
    LaunchedEffect(rawText, rawHex, rawBytes) {
        previewScroll.scrollTo(previewScroll.maxValue)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(text = stringResource(id = R.string.title)) }
            )
        }
    ) { padding ->
        val contentPadding = PaddingValues(
            start = 16.dp,
            end = 16.dp,
            top = padding.calculateTopPadding() + 16.dp,
            bottom = padding.calculateBottomPadding() + 16.dp
        )

        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = contentPadding,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                AssistChip(
                    onClick = {},
                    label = {
                        Text(
                            if (useMock) stringResource(R.string.status_mock)
                            else stringResource(R.string.status_ble)
                        )
                    }
                )
            }

            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    MetricCard(
                        modifier = Modifier.weight(1f),
                        title = stringResource(R.string.hr),
                        value = rates.bpm?.let { "%.0f".format(it) } ?: "--",
                        unit = stringResource(R.string.bpm_unit)
                    )
                    MetricCard(
                        modifier = Modifier.weight(1f),
                        title = stringResource(R.string.rr),
                        value = rates.rpm?.let { "%.0f".format(it) } ?: "--",
                        unit = stringResource(R.string.rpm_unit)
                    )
                }
            }

            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = stringResource(R.string.mode),
                                style = MaterialTheme.typography.titleMedium
                            )
                            Text(
                                text = if (useMock) stringResource(R.string.mock) else stringResource(R.string.ble),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = {
                                    useMock = true
                                    showBleDialog = false
                                },
                                enabled = !useMock
                            ) { Text(stringResource(R.string.mock)) }

                            Button(
                                onClick = {
                                    useMock = false
                                    showBleDialog = true
                                    processor.reset()
                                },
                                enabled = !showBleDialog
                            ) { Text(stringResource(R.string.ble)) }
                        }
                    }
                }
            }

            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(text = "蓝牙接收预览", style = MaterialTheme.typography.titleMedium)
                            OutlinedButton(onClick = { sppClient.clearPreview() }) {
                                Text("清空")
                            }
                        }
                        Spacer(Modifier.height(6.dp))
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(140.dp)
                                .verticalScroll(previewScroll)
                        ) {
                        Text(
                            text = "HEX: " + if (rawHex.isBlank()) "暂无数据" else rawHex,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = "ASCII: " + if (rawText.isBlank()) "暂无数据" else rawText,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.height(6.dp))
                        Text(
                            text = "RX bytes: $rawBytes",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        val timeText = lastRxMs?.let { timeFormatter.format(Date(it)) } ?: "--:--:--"
                        Text(
                            text = "TIME: $timeText",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        if (!connectionError.isNullOrBlank()) {
                            Spacer(Modifier.height(6.dp))
                            Text(
                                text = "连接错误: $connectionError",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }
                    }
                }
            }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_raw)) {
                    Waveform(
                        buffer = processor.rawBuf,
                        color = MaterialTheme.colorScheme.primary,
                        yMin = -2f,
                        yMax = 2f,
                        showGrid = true,
                        showZeroLine = true
                    )
                }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_resp)) {
                    Waveform(
                        buffer = processor.respBuf,
                        color = MaterialTheme.colorScheme.tertiary,
                        yMin = -2f,
                        yMax = 2f,
                        showGrid = true,
                        showZeroLine = true
                    )
                }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_hr)) {
                    Waveform(
                        buffer = processor.hrBuf,
                        color = MaterialTheme.colorScheme.error,
                        yMin = -2f,
                        yMax = 2f,
                        showGrid = true,
                        showZeroLine = true,
                        zeroLineValue = 0f
                    )
                }
            }
        }
    }

    if (showBleDialog) {
        BleDialog(
            devices = devices,
            connectionState = connectionState,
            onScan = {
                if (hasBleScanPermission(context)) {
                    sppClient.startScan()
                } else {
                    requestBlePermissions()
                }
            },
            onStop = {
                if (hasBleScanPermission(context)) {
                    sppClient.stopScan()
                }
            },
            onConnect = {
                if (hasBleConnectPermission(context)) {
                    pendingConnect = true
                    sppClient.connect(it)
                } else {
                    requestBlePermissions()
                }
            },
            onDisconnect = {
                if (hasBleConnectPermission(context)) {
                    pendingConnect = false
                    sppClient.disconnect()
                }
            },
            onRequestPermissions = { requestBlePermissions() },
            onDismiss = { showBleDialog = false },
            hasPermissions = hasBlePermissions
        )
    }

    if (connectionMessage != null) {
        AlertDialog(
            onDismissRequest = { connectionMessage = null },
            title = { Text("蓝牙") },
            text = { Text(connectionMessage ?: "") },
            confirmButton = {
                Button(onClick = { connectionMessage = null }) {
                    Text("确定")
                }
            }
        )
    }
}

@Composable
private fun MetricCard(
    modifier: Modifier = Modifier,
    title: String,
    value: String,
    unit: String
) {
    Card(
        modifier = modifier,
        shape = MaterialTheme.shapes.large
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(text = title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = value,
                    style = MaterialTheme.typography.displaySmall
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    text = unit,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun ChartCard(
    title: String,
    content: @Composable () -> Unit
) {
    Card(shape = MaterialTheme.shapes.large) {
        Column(modifier = Modifier.padding(12.dp)) {
            Text(text = title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun BleDialog(
    devices: List<SppDevice>,
    connectionState: ConnectionState,
    onScan: () -> Unit,
    onStop: () -> Unit,
    onConnect: (SppDevice) -> Unit,
    onDisconnect: () -> Unit,
    onRequestPermissions: () -> Unit,
    onDismiss: () -> Unit,
    hasPermissions: Boolean
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(text = stringResource(R.string.ble_dialog_title)) },
        text = {
            Column {
                Text(
                    text = stringResource(R.string.ble_status) + ": " + when (connectionState) {
                        ConnectionState.Disconnected -> stringResource(R.string.ble_disconnected)
                        ConnectionState.Scanning -> stringResource(R.string.ble_scanning)
                        ConnectionState.Connecting -> stringResource(R.string.ble_connecting)
                        is ConnectionState.Connected -> stringResource(R.string.ble_connected) +
                                " (${connectionState.device.address})"
                    },
                    style = MaterialTheme.typography.bodyMedium
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = onScan, enabled = hasPermissions) {
                        Text(stringResource(R.string.ble_scan))
                    }
                    OutlinedButton(onClick = onStop, enabled = hasPermissions) {
                        Text(stringResource(R.string.ble_stop))
                    }
                    OutlinedButton(onClick = onDisconnect) {
                        Text(stringResource(R.string.ble_disconnect))
                    }
                }
                Spacer(Modifier.height(8.dp))
                if (!hasPermissions) {
                    Text(
                        text = stringResource(R.string.ble_permissions_missing),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = onRequestPermissions) {
                        Text("Request permission")
                    }
                }
                Spacer(Modifier.height(8.dp))
                if (devices.isEmpty()) {
                    Text(
                        text = stringResource(R.string.ble_no_devices),
                        style = MaterialTheme.typography.bodyMedium
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(320.dp)
                    ) {
                        LazyColumn(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            items(devices) { device ->
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    shape = MaterialTheme.shapes.medium
                                ) {
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(8.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Column {
                                            Text(
                                                text = device.name ?: stringResource(R.string.ble_no_name),
                                                style = MaterialTheme.typography.bodyMedium
                                            )
                                            Text(
                                                text = device.address,
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                            Text(
                                                text = "RSSI ${device.rssi}",
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                        }
                                        Button(
                                            onClick = { onConnect(device) },
                                            enabled = hasPermissions
                                        ) {
                                            Text(stringResource(R.string.ble_connect))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) { Text(stringResource(R.string.ble_close)) }
        }
    )
}

private fun requiredBlePermissions(): Array<String> {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        arrayOf(
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT
        )
    } else {
        arrayOf(
            Manifest.permission.BLUETOOTH,
            Manifest.permission.BLUETOOTH_ADMIN,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
    }
}

private fun hasBlePermissions(context: android.content.Context): Boolean {
    return requiredBlePermissions().all { perm ->
        ContextCompat.checkSelfPermission(context, perm) == PackageManager.PERMISSION_GRANTED
    }
}

private fun hasBleScanPermission(context: android.content.Context): Boolean {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BLUETOOTH_SCAN
        ) == PackageManager.PERMISSION_GRANTED
    } else {
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }
}

private fun hasBleConnectPermission(context: android.content.Context): Boolean {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BLUETOOTH_CONNECT
        ) == PackageManager.PERMISSION_GRANTED
    } else {
        true
    }
}
