package com.example.quant_repeatability_benchmark

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.view.WindowManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            "quant_benchmark/platform",
        ).setMethodCallHandler { call, result ->
            when (call.method) {
                "externalFilesPath" -> result.success(getExternalFilesDir(null)?.absolutePath)
                "deviceContext" -> result.success(contextMap())
                else -> result.notImplemented()
            }
        }
    }

    private fun contextMap(): Map<String, Any?> {
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val status = battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val temperature = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        val memory = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
        return mapOf(
            "device_id" to "${Build.MANUFACTURER}_${Build.MODEL}_api${Build.VERSION.SDK_INT}",
            "process_pss_mb" to memory.totalPss / 1024.0,
            "process_rss_mb" to readProcStatusMb("VmRSS"),
            "battery_percent" to if (level >= 0 && scale > 0) level * 100.0 / scale else null,
            "charging_state" to when (status) {
                BatteryManager.BATTERY_STATUS_CHARGING -> "charging"
                BatteryManager.BATTERY_STATUS_FULL -> "full"
                BatteryManager.BATTERY_STATUS_DISCHARGING -> "discharging"
                BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "not_charging"
                else -> "unknown"
            },
            "battery_saver" to power.isPowerSaveMode,
            "thermal_status" to if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1,
            "battery_temperature_c" to if (temperature >= 0) temperature / 10.0 else null,
        )
    }

    private fun readProcStatusMb(field: String): Double? = try {
        File("/proc/self/status").useLines { lines ->
            lines.firstOrNull { it.startsWith("$field:") }
                ?.substringAfter(':')
                ?.trim()
                ?.substringBefore(' ')
                ?.toDoubleOrNull()
                ?.div(1024.0)
        }
    } catch (_: Exception) {
        null
    }
}
