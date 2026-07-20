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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
    val waveformFrame = rememberWaveformFrame(refreshMs = 50L)

    var useMock by remember { mutableStateOf(true) }
    var showBleDialog by remember { mutableStateOf(false) }
    var autoRawGain by remember { mutableStateOf(true) }
    var autoRawWindow by remember { mutableStateOf(true) }
    var rawGain by remember { mutableStateOf(1f) }
    var rawWindowMs by remember { mutableStateOf(6000L) }
    var autoHrGain by remember { mutableStateOf(true) }
    var hrGain by remember { mutableStateOf(1f) }
    var autoRespGain by remember { mutableStateOf(true) }
    var respGain by remember { mutableStateOf(1f) }
    val effectiveRawWindowMs = rememberAutoWindowMs(
        processor = processor,
        autoEnabled = autoRawWindow,
        manualWindowMs = rawWindowMs
    )
    val effectiveHrGain = rememberAutoGain(
        buffer = processor.cleanHeartBuf,
        windowMs = 6000L,
        yMin = -300f,
        yMax = 300f,
        autoEnabled = autoHrGain,
        manualGain = hrGain
    )

    val effectiveRawGain = rememberAutoGain(
        buffer = processor.rawBuf,
        windowMs = 6000L,
        windowMsState = effectiveRawWindowMs,
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

    val scope = rememberCoroutineScope()
    var job by remember { mutableStateOf<Job?>(null) }

    var pendingConnect by remember { mutableStateOf(false) }
    var connectionMessage by remember { mutableStateOf<String?>(null) }
    var sleepStateResult by remember { mutableStateOf<SleepStateResult?>(null) }
    var sleepStateAnalyzing by remember { mutableStateOf(false) }
    var sleepModelHealth by remember { mutableStateOf<SleepModelHealth?>(null) }
    val sleepInputProgress = remember {
        MutableStateFlow(sleepStateService.inputProgress(processor))
    }
    var lastSleepAnalysisSampleMs by remember { mutableStateOf<Long?>(null) }
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
        job?.cancelAndJoin()
        sleepStateService.resetFeatureCache()
        processor.reset()
        sleepStateResult = null
        sleepStateAnalyzing = false
        lastSleepAnalysisSampleMs = null
        sleepInputProgress.value = sleepStateService.inputProgress(processor)
        if (useMock) {
            job = scope.launch(Dispatchers.Default) {
                MockDataSource(fsHz = fsHz).samples().collect { s ->
                    processor.onSample(s, bypassMotionDetection = true)
                }
            }
            if (hasBleScanPermission(context)) {
                bleClient.stopScan()
            }
        } else {
            job = scope.launch(Dispatchers.Default) {
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
            sleepStateService.resetFeatureCache()
            processor.reset()
            sleepStateResult = null
            lastSleepAnalysisSampleMs = null
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

            item { VitalSignsRow(processor) }

            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = MaterialTheme.shapes.large
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text(
                            text = "Sleep Diagnosis",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(Modifier.height(12.dp))
                        Text(
                            text = "Sleeping",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = sleepStateValue(
                                result = sleepStateResult,
                                analyzing = sleepStateAnalyzing
                            ),
                            style = MaterialTheme.typography.headlineMedium
                        )
                        Spacer(Modifier.height(12.dp))
                        SleepCollectionProgress(sleepInputProgress)
                        Spacer(Modifier.height(12.dp))
                        Button(
                            onClick = {
                                sleepStateAnalyzing = true
                                sleepStateResult = null
                                scope.launch {
                                    val result = sleepStateService.predict(processor, processor.rates.value)
                                    sleepStateResult = result
                                    processor.updateSleepPrediction(result)
                                    if (result.code == 0) {
                                        lastSleepAnalysisSampleMs = result.timestampMs
                                    }
                                    sleepStateAnalyzing = false
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !sleepStateAnalyzing && sleepModelHealth?.ready != false
                        ) {
                            Text(if (sleepStateAnalyzing) "Analyzing..." else "Analyze")
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
                        Text(text = "Sensor Data Preview (s16)", style = MaterialTheme.typography.titleMedium)
                        RawSensorPreview(processor)
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
                                    Log.i("MainActivity", "CSV logging started")
                                }) { Text("Start CSV Logging") }
                            } else {
                               Button(onClick = {
                                    val f = btnLogger.stopLogging()
                                    sensorLogPath = f?.absolutePath
                                   processor.sensorLogger = null
                                    isSensorLogging = false
                                    Log.i("MainActivity", "CSV logging stopped")
                                }) { Text("Stop Logging") }
                            }
                        }
                        if (!sensorLogPath.isNullOrBlank()) {
                            Spacer(Modifier.height(4.dp))
                            Text(
                                text = "Log File: $sensorLogPath",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        CenteredSensorPreview(processor)
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
                            windowMs = 6000L,
                            windowMsProvider = { effectiveRawWindowMs.value },
                            gain = 1f,
                            gainProvider = { effectiveRawGain.value },
                            showGrid = true,
                            showZeroLine = true,
                            frameTick = waveformFrame
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
                            AutoGainLabel(autoRawGain, effectiveRawGain, rawGain)
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
                            AutoWindowLabel(autoRawWindow, effectiveRawWindowMs, rawWindowMs)
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
                            gain = 1f,
                            gainProvider = { effectiveRespGain.value },
                            showGrid = true,
                            showZeroLine = true,
                            frameTick = waveformFrame
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
                            AutoGainLabel(autoRespGain, effectiveRespGain, respGain)
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
                            buffer = processor.cleanHeartBuf,
                            color = MaterialTheme.colorScheme.error,
                            yMin = -300f,
                        yMax = 300f,
                        windowMs = 10000L,
                        gain = 1f,
                        gainProvider = { effectiveHrGain.value },
                            showGrid = true,
                            showZeroLine = true,
                            zeroLineValue = 0f,
                        peakTimesProvider = { processor.cleanPeakTimes.value },
                        frameTick = waveformFrame
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
                            AutoGainLabel(autoHrGain, effectiveHrGain, hrGain)
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

            item {
                ChartCard(title = "Heartbeat Enhancement") {
                    Waveform(
                        buffer = processor.morphologyHeartBuf,
                        color = MaterialTheme.colorScheme.tertiary,
                        yMin = -0.8f,
                        yMax = 1.15f,
                        windowMs = 7_000L,
                        windowMsProvider = {
                            val snapshot = processor.heartMorphologySnapshot.value
                            if (snapshot.ready) snapshot.durationMs.coerceIn(4_000L, 12_000L)
                            else 7_000L
                        },
                        gain = 1f,
                        showGrid = true,
                        showZeroLine = true,
                        zeroLineValue = 0f,
                        peakTimesProvider = { processor.morphologyPlaybackBoundaryTimes.value },
                        frameTick = waveformFrame
                    )
                }
            }
        }
    }

    LaunchedEffect(sleepStateService) {
        sleepModelHealth = sleepStateService.checkModel()
    }

    LaunchedEffect(sleepModelHealth?.ready) {
        while (true) {
            delay(1_000L)
            val progress = withContext(Dispatchers.Default) {
                sleepStateService.inputProgress(processor)
            }
            sleepInputProgress.value = progress
            val lastAnalyzed = lastSleepAnalysisSampleMs
            val newWindowAvailable = progress.timestampMs != null &&
                (lastAnalyzed == null || progress.timestampMs - lastAnalyzed >= 30_000L)
            if (sleepModelHealth?.ready == true &&
                progress.ready &&
                newWindowAvailable &&
                !sleepStateAnalyzing
            ) {
                sleepStateAnalyzing = true
                val result = sleepStateService.predict(processor, processor.rates.value)
                sleepStateResult = result
                processor.updateSleepPrediction(result)
                lastSleepAnalysisSampleMs = result.timestampMs ?: progress.timestampMs
                sleepStateAnalyzing = false
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

/**
 * Keep a paper-style ten-second multi-cycle window in the enhanced-display chart.
 * This changes only the x-axis window; it does not resample the waveform or affect BPM.
 */
internal fun calculateEnhancedHeartWindowMs(): Long = 10_000L

@Composable
private fun rememberAutoWindowMs(
    processor: Processor,
    autoEnabled: Boolean,
    manualWindowMs: Long
): State<Long> {
    val autoWindowMs = remember { mutableLongStateOf(manualWindowMs) }

    LaunchedEffect(processor, autoEnabled, manualWindowMs) {
        if (!autoEnabled) {
            autoWindowMs.longValue = manualWindowMs
            return@LaunchedEffect
        }

        while (true) {
            val rates = processor.rates.value
            val bpm = rates.bpm
            val rpm = rates.rpm
            val targetMs = when {
                bpm != null && bpm > 1f -> ((8f * 60_000f) / bpm).toLong()
                rpm != null && rpm > 0.2f -> ((2.5f * 60_000f) / rpm).toLong()
                else -> 6000L
            }.coerceIn(2000L, 15000L)

            autoWindowMs.longValue =
                (autoWindowMs.longValue * 0.7f + targetMs * 0.3f).toLong()
            delay(1_000L)
        }
    }

    return autoWindowMs
}

@Composable
private fun rememberAutoGain(
    buffer: RingBuffer,
    windowMs: Long,
    windowMsState: State<Long>? = null,
    yMin: Float,
    yMax: Float,
    autoEnabled: Boolean,
    manualGain: Float
): State<Float> {
    val autoGain = remember { mutableFloatStateOf(manualGain) }

    LaunchedEffect(buffer, windowMs, windowMsState, yMin, yMax, autoEnabled, manualGain) {
        if (!autoEnabled) {
            autoGain.floatValue = manualGain
            return@LaunchedEffect
        }

        while (true) {
            val recentRange = buffer.valueRange(windowMsState?.value ?: windowMs)
            if (recentRange != null) {
                val minV = recentRange.minimum
                val maxV = recentRange.maximum
                if (recentRange.sampleCount > 1) {
                    val maxExtent = maxOf(kotlin.math.abs(minV), kotlin.math.abs(maxV), 1f)
                    val range = max(1f, yMax - yMin)
                    val halfRange = (yMax - yMin) / 2f
                    // 基于最大幅值计算增益，避免DC偏置导致削顶
                    // targetGain = 半量程 * 0.72 / maxExtent
                    val targetGain = (halfRange * 0.72f / maxExtent).coerceIn(0.1f, 200f)
                    autoGain.floatValue = autoGain.floatValue * 0.75f + targetGain * 0.25f
                }
            }
            delay(250L)
        }
    }

    return autoGain
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

private fun sleepStateValue(result: SleepStateResult?, analyzing: Boolean): String {
    if (analyzing) return "..."
    if (result == null) return "--"
    return when (result.code) {
        0 -> if (result.state == "sleep") "Yes" else "No"
        1001 -> "Collecting"
        1002 -> "Unknown"
        1004 -> "Unavailable"
        else -> "Failed"
    }
}

@Composable
private fun RawSensorPreview(processor: Processor) {
    val value by processor.rawPreview.collectAsState()
    Text(
        text = "Raw s16: $value",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

@Composable
private fun CenteredSensorPreview(processor: Processor) {
    val value by processor.centeredPreview.collectAsState()
    Text(
        text = "DC Removed: $value",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.primary
    )
}

private fun sleepCollectionProgressValue(progress: SleepModelInputProgress): String {
    if (progress.ready) return "${progress.requiredSeconds} / ${progress.requiredSeconds} s · Ready"
    if (progress.collectedSamples >= progress.requiredSamples) {
        return "${progress.requiredSeconds} / ${progress.requiredSeconds} s · Signal poor"
    }
    val collectedSeconds = progress.collectedSeconds
        .coerceAtMost((progress.requiredSeconds - 1).coerceAtLeast(0).toFloat())
        .toInt()
    return "$collectedSeconds / ${progress.requiredSeconds} s"
}

@Composable
private fun SleepCollectionProgress(progressFlow: StateFlow<SleepModelInputProgress>) {
    val progress by progressFlow.collectAsState()
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = "Collection Progress",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = sleepCollectionProgressValue(progress),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
    Spacer(Modifier.height(6.dp))
    LinearProgressIndicator(
        progress = { progress.fraction },
        modifier = Modifier.fillMaxWidth()
    )
}

@Composable
private fun VitalSignsRow(processor: Processor) {
    val rates by processor.rates.collectAsState()
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

@Composable
private fun AutoGainLabel(
    autoEnabled: Boolean,
    autoGain: State<Float>,
    manualGain: Float
) {
    val gain = if (autoEnabled) autoGain.value else manualGain
    Text(
        text = if (autoEnabled) {
            "Gain Auto (x${String.format("%.1f", gain)})"
        } else {
            "Gain x${String.format("%.1f", gain)}"
        },
        style = MaterialTheme.typography.bodyMedium
    )
}

@Composable
private fun AutoWindowLabel(
    autoEnabled: Boolean,
    autoWindowMs: State<Long>,
    manualWindowMs: Long
) {
    val windowMs = if (autoEnabled) autoWindowMs.value else manualWindowMs
    Text(
        text = if (autoEnabled) {
            "Window Auto (${String.format("%.1f", windowMs / 1000f)}s)"
        } else {
            "Window ${String.format("%.1f", windowMs / 1000f)}s"
        },
        style = MaterialTheme.typography.bodyMedium
    )
}








