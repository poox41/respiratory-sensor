package com.example.breathheartdemo

import android.annotation.SuppressLint
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.max
import kotlin.math.min
import android.util.Log

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
@SuppressLint("MissingPermission")
fun AppScreen() {
    val fsHz = 50
    val context = LocalContext.current
    val processor = remember { Processor(fsHz) }
    val sleepStateService = remember { SleepStateService(context.applicationContext, fsHz) }
    val rates by processor.rates.collectAsState()

    var useMock by remember { mutableStateOf(true) }
    var showBleDialog by remember { mutableStateOf(false) }
    var autoRawGain by remember { mutableStateOf(true) }
    var autoRawWindow by remember { mutableStateOf(true) }
    var rawGain by remember { mutableStateOf(1f) }
    var rawWindowMs by remember { mutableStateOf(6000L) }
    var autoHrGain by remember { mutableStateOf(false) }
    var hrGain by remember { mutableStateOf(1f) }
    var autoRespGain by remember { mutableStateOf(true) }
    var respGain by remember { mutableStateOf(1f) }
    val effectiveRawWindowMs = rememberAutoWindowMs(
        rates = rates,
        autoEnabled = autoRawWindow,
        manualWindowMs = rawWindowMs
    )
    val effectiveHrGain = rememberAutoGain(
        buffer = processor.hrBuf,
        windowMs = 6000L,
        yMin = -300f,
        yMax = 300f,
        autoEnabled = autoHrGain,
        manualGain = hrGain
    )

    val effectiveRawGain = rememberAutoGain(
        buffer = processor.rawBuf,
        windowMs = effectiveRawWindowMs,
        yMin = -2000f,
        yMax = 2000f,
        autoEnabled = autoRawGain,
        manualGain = rawGain
    )
    val effectiveRespGain = rememberAutoGain(
        buffer = processor.respDisplayBuf,
        windowMs = 10000L,
        yMin = -2000f,
        yMax = 2000f,
        autoEnabled = autoRespGain,
        manualGain = respGain
    )

    val bleClient = remember { BleClient(context.applicationContext, fsHz) }
    val devices by bleClient.scanResults.collectAsState()
    val connectionState by bleClient.connectionState.collectAsState()
    val rawHex by bleClient.rawHex.collectAsState()
    val rawBytes by bleClient.rawBytes.collectAsState()
    val connectionError by bleClient.connectionError.collectAsState()
    val exportSessionPath by bleClient.exportSessionPath.collectAsState()
    val exportFileName by bleClient.exportFileName.collectAsState()
    val raw16Preview by bleClient.raw16Preview.collectAsState()
    val rawPreview by processor.rawPreview.collectAsState()
    val centeredPreview by processor.centeredPreview.collectAsState()
    val peakTimes by processor.peakTimes.collectAsState()

    val scope = rememberCoroutineScope()
    var job by remember { mutableStateOf<Job?>(null) }

    var pendingConnect by remember { mutableStateOf(false) }
    var connectionMessage by remember { mutableStateOf<String?>(null) }
    var sleepStateResult by remember { mutableStateOf<SleepStateResult?>(null) }
    var sleepStateAnalyzing by remember { mutableStateOf(false) }
    var lastConnectionState by remember { mutableStateOf<ConnectionState>(ConnectionState.Disconnected) }

    var hasBlePermissions by remember { mutableStateOf(hasBlePermissions(context)) }
    val previewScroll = rememberScrollState()
    var isSensorLogging by remember { mutableStateOf(false) }
    var sensorLogPath by remember { mutableStateOf<String?>(null) }

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
                bleClient.stopScan()
            }
        } else {
            job = scope.launch {
                bleClient.samples.collect { s ->
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
            bleClient.startScan()
        }
    }

    LaunchedEffect(connectionState, pendingConnect) {
        if (pendingConnect) {
            when (connectionState) {
                is ConnectionState.Connected -> {
                    connectionMessage = "Connection successful"
                    showBleDialog = false
                    pendingConnect = false
                }
                ConnectionState.Disconnected -> {
                    connectionMessage = "Connection failed"
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
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "睡眠状态",
                                    style = MaterialTheme.typography.titleMedium
                                )
                                Spacer(Modifier.height(6.dp))
                                Text(
                                    text = sleepStateText(
                                        result = sleepStateResult,
                                        analyzing = sleepStateAnalyzing
                                    ),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Spacer(Modifier.width(12.dp))
                            Button(
                                onClick = {
                                    sleepStateAnalyzing = true
                                    sleepStateResult = null
                                    scope.launch {
                                        sleepStateResult = sleepStateService.predict(processor, rates)
                                        sleepStateAnalyzing = false
                                    }
                                },
                                enabled = !sleepStateAnalyzing
                            ) {
                                Text("判断睡眠状态")
                            }
                        }
                    }
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
                Card(shape = MaterialTheme.shapes.large) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(text = "数据值预览 (s16)", style = MaterialTheme.typography.titleMedium)
                        Text(
                            text = "原始s16: $rawPreview",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Spacer(Modifier.height(6.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            val btnLogger = processor.sensorLogger
                            if (btnLogger == null) {
                               Button(onClick = {
                                   val l = SensorDataLogger(context.applicationContext)
                                    val f = l.startLogging()
                                    sensorLogPath = f?.absolutePath
                                   processor.sensorLogger = l
                                    isSensorLogging = true
                                    Log.i("MainActivity", "日志记录已开始")
                                }) { Text("记录日志到CSV") }
                            } else {
                               Button(onClick = {
                                    val f = btnLogger.stopLogging()
                                    sensorLogPath = f?.absolutePath
                                   processor.sensorLogger = null
                                    isSensorLogging = false
                                    Log.i("MainActivity", "日志已停止")
                                }) { Text("停止记录") }
                            }
                        }
                        if (!sensorLogPath.isNullOrBlank()) {
                            Spacer(Modifier.height(4.dp))
                            Text(
                                text = "日志文件: $sensorLogPath",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Text(
                            text = "去直流: $centeredPreview",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_raw)) {
                    Column {
                        Waveform(
                            buffer = processor.rawBuf,
                            color = MaterialTheme.colorScheme.primary,
                            yMin = -2000f,
                            yMax = 2000f,
                            windowMs = effectiveRawWindowMs,
                            gain = effectiveRawGain,
                            showGrid = true,
                            showZeroLine = true
                        )
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Auto Gain",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Switch(
                                checked = autoRawGain,
                                onCheckedChange = { autoRawGain = it }
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Auto Time Window",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Switch(
                                checked = autoRawWindow,
                                onCheckedChange = { autoRawWindow = it }
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (autoRawGain) {
                                    "Gain Auto (x" + String.format("%.1f", effectiveRawGain) + ")"
                                } else {
                                    "Gain x" + String.format("%.1f", rawGain)
                                },
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(
                                    onClick = { rawGain = (rawGain / 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoRawGain
                                ) { Text("-") }
                                OutlinedButton(
                                    onClick = { rawGain = 1f },
                                    enabled = !autoRawGain
                                ) { Text("Reset") }
                                Button(
                                    onClick = { rawGain = (rawGain * 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoRawGain
                                ) { Text("+") }
                            }
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (autoRawWindow) {
                                    "Window Auto (" + String.format("%.1f", effectiveRawWindowMs / 1000f) + "s)"
                                } else {
                                    "Window " + String.format("%.1f", rawWindowMs / 1000f) + "s"
                                },
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(
                                    onClick = {
                                        rawWindowMs = (rawWindowMs / 1.5).toLong().coerceIn(1000L, 30000L)
                                    },
                                    enabled = !autoRawWindow
                                ) { Text("-") }
                                OutlinedButton(
                                    onClick = { rawWindowMs = 6000L },
                                    enabled = !autoRawWindow
                                ) { Text("Reset") }
                                Button(
                                    onClick = {
                                        rawWindowMs = (rawWindowMs * 1.5).toLong().coerceIn(1000L, 30000L)
                                    },
                                    enabled = !autoRawWindow
                                ) { Text("+") }
                            }
                        }
                    }
                }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_resp)) {
                    Column {
                        Waveform(
                            buffer = processor.respDisplayBuf,
                            color = MaterialTheme.colorScheme.tertiary,
                            yMin = -2000f,
                            yMax = 2000f,
                            windowMs = 10000L,
                            gain = effectiveRespGain,
                            showGrid = true,
                            showZeroLine = true
                        )
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(text = "Auto Gain", style = MaterialTheme.typography.bodyMedium)
                            Switch(checked = autoRespGain, onCheckedChange = { autoRespGain = it })
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (autoRespGain) {
                                    "Gain Auto (x" + String.format("%.1f", effectiveRespGain) + ")"
                                } else {
                                    "Gain x" + String.format("%.1f", respGain)
                                },
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(
                                    onClick = { respGain = (respGain / 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoRespGain
                                ) { Text("-") }
                                OutlinedButton(
                                    onClick = { respGain = 1f },
                                    enabled = !autoRespGain
                                ) { Text("Reset") }
                                Button(
                                    onClick = { respGain = (respGain * 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoRespGain
                                ) { Text("+ ") }
                            }
                        }
                    }
                }
            }

            item {
                ChartCard(title = stringResource(R.string.chart_hr)) {
                    Column {
                        Waveform(
                            buffer = processor.hrBuf,
                            color = MaterialTheme.colorScheme.error,
                            yMin = -300f,
                        yMax = 300f,
                        windowMs = 10000L,
                        gain = effectiveHrGain,
                            showGrid = true,
                            showZeroLine = true,
                            zeroLineValue = 0f,
                        peakTimes = peakTimes
                    )
                    Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Auto Gain",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Switch(
                                checked = autoHrGain,
                                onCheckedChange = { autoHrGain = it }
                            )
                        }
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (autoHrGain) {
                                    "Gain Auto (x" + String.format("%.1f", effectiveHrGain) + ")"
                                } else {
                                    "Gain x" + String.format("%.1f", hrGain)
                                },
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(
                                    onClick = { hrGain = (hrGain / 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoHrGain
                                ) { Text("-") }
                                OutlinedButton(
                                    onClick = { hrGain = 1f },
                                    enabled = !autoHrGain
                                ) { Text("Reset") }
                                Button(
                                    onClick = { hrGain = (hrGain * 1.2f).coerceIn(0.1f, 200f) },
                                    enabled = !autoHrGain
                                ) { Text("+ ") }
                            }
                        }
                    }
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
                    bleClient.startScan()
                } else {
                    requestBlePermissions()
                }
            },
            onStop = {
                if (hasBleScanPermission(context)) {
                    bleClient.stopScan()
                }
            },
            onConnect = {
                if (hasBleConnectPermission(context)) {
                    pendingConnect = true
                    bleClient.connect(it)
                } else {
                    requestBlePermissions()
                }
            },
            onDisconnect = {
                if (hasBleConnectPermission(context)) {
                    pendingConnect = false
                    bleClient.disconnect()
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
            title = { Text("Bluetooth") },
            text = { Text(connectionMessage ?: "") },
            confirmButton = {
                Button(onClick = { connectionMessage = null }) {
                    Text("OK")
                }
            }
        )
    }
}

@Composable
private fun rememberAutoWindowMs(
    rates: Rates,
    autoEnabled: Boolean,
    manualWindowMs: Long
): Long {
    var autoWindowMs by remember { mutableLongStateOf(manualWindowMs) }

    LaunchedEffect(autoEnabled, manualWindowMs, rates.bpm, rates.rpm) {
        if (!autoEnabled) {
            autoWindowMs = manualWindowMs
            return@LaunchedEffect
        }

        val bpm = rates.bpm
        val rpm = rates.rpm
        val targetMs = when {
            bpm != null && bpm > 1f -> ((8f * 60_000f) / bpm).toLong()
            rpm != null && rpm > 0.2f -> ((2.5f * 60_000f) / rpm).toLong()
            else -> 6000L
        }.coerceIn(2000L, 15000L)

        autoWindowMs = (autoWindowMs * 0.7f + targetMs * 0.3f).toLong()
    }

    return if (autoEnabled) autoWindowMs else manualWindowMs
}

@Composable
private fun rememberAutoGain(
    buffer: RingBuffer,
    windowMs: Long,
    yMin: Float,
    yMax: Float,
    autoEnabled: Boolean,
    manualGain: Float
): Float {
    var autoGain by remember { mutableFloatStateOf(manualGain) }

    LaunchedEffect(buffer, windowMs, yMin, yMax, autoEnabled, manualGain) {
        if (!autoEnabled) {
            autoGain = manualGain
            return@LaunchedEffect
        }

        while (true) {
            val (ts, vs) = buffer.snapshot()
            if (vs.isNotEmpty()) {
                val tMax = ts[vs.lastIndex]
                val tMin = tMax - windowMs
                var minV = Float.POSITIVE_INFINITY
                var maxV = Float.NEGATIVE_INFINITY

                for (i in vs.indices) {
                    if (ts[i] < tMin) continue
                    minV = min(minV, vs[i])
                    maxV = max(maxV, vs[i])
                }

                if (minV != Float.POSITIVE_INFINITY && maxV != Float.NEGATIVE_INFINITY) {
                    val p2p = max(1f, maxV - minV)
                    val maxExtent = maxOf(kotlin.math.abs(minV), kotlin.math.abs(maxV), 1f)
                    val range = max(1f, yMax - yMin)
                    val halfRange = (yMax - yMin) / 2f
                    // 基于最大幅值计算增益，避免DC偏置导致削顶
                    // targetGain = 半量程 * 0.72 / maxExtent
                    val targetGain = (halfRange * 0.72f / maxExtent).coerceIn(0.1f, 200f)
                    autoGain = autoGain * 0.75f + targetGain * 0.25f
                }
            }
            delay(120L)
        }
    }

    return if (autoEnabled) autoGain else manualGain
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
    devices: List<BleDevice>,
    connectionState: ConnectionState,
    onScan: () -> Unit,
    onStop: () -> Unit,
    onConnect: (BleDevice) -> Unit,
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

private fun sleepStateText(result: SleepStateResult?, analyzing: Boolean): String {
    if (analyzing) return "正在分析..."
    if (result == null) return "等待判断"

    return when (result.code) {
        0 -> "当前状态：${result.stateName}，置信度 ${"%.0f".format(result.confidence * 100f)}%"
        1001 -> "当前采集数据不足，请继续采集后重试（${result.message}）"
        1002 -> "当前数据暂无法判断：信号质量差"
        else -> "状态识别失败：${result.message}"
    }
}








