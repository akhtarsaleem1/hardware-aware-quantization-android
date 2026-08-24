import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

typedef ProgressCallback = void Function(String message);

Future<String> runBenchmarkInBackground({
  required ProgressCallback onProgress,
}) async {
  final token = RootIsolateToken.instance;
  if (token == null) {
    throw StateError('Flutter root isolate token is unavailable');
  }
  final manifestJson = await rootBundle.loadString(
    'assets/benchmark_manifest.json',
  );
  final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
  final transferredModels = <String, TransferableTypedData>{};
  for (final value in (manifest['models'] as List).cast<Map>()) {
    final model = Map<String, dynamic>.from(value);
    final data = await rootBundle.load(model['asset'] as String);
    transferredModels[model['id'] as String] = TransferableTypedData.fromList([
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    ]);
  }
  final receive = ReceivePort();
  await Isolate.spawn<List<Object?>>(_benchmarkIsolateMain, <Object?>[
    token,
    receive.sendPort,
    manifestJson,
    transferredModels,
  ], debugName: 'litert-benchmark-worker');
  try {
    await for (final message in receive) {
      final event = Map<String, dynamic>.from(message as Map);
      switch (event['type']) {
        case 'progress':
          onProgress(event['message'] as String);
          break;
        case 'complete':
          return event['path'] as String;
        case 'error':
          throw StateError(
            'Background benchmark failed: ${event['error']}\n${event['stack']}',
          );
        default:
          throw StateError('Unknown background benchmark event: $event');
      }
    }
    throw StateError('Background benchmark isolate exited without a result');
  } finally {
    receive.close();
  }
}

@pragma('vm:entry-point')
void _benchmarkIsolateMain(List<Object?> message) async {
  final token = message[0] as RootIsolateToken;
  final sendPort = message[1] as SendPort;
  BackgroundIsolateBinaryMessenger.ensureInitialized(token);
  final manifestJson = message[2] as String;
  final transferred = Map<String, TransferableTypedData>.from(
    message[3] as Map,
  );
  final modelBuffers = <String, Uint8List>{
    for (final entry in transferred.entries)
      entry.key: entry.value.materialize().asUint8List(),
  };
  final workerPlatform = MethodChannel(
    BenchmarkRunner.platformChannelName,
    const StandardMethodCodec(),
    BackgroundIsolateBinaryMessenger.instance,
  );
  try {
    final path = await BenchmarkRunner(
      onProgress: (value) =>
          sendPort.send({'type': 'progress', 'message': value}),
      platform: workerPlatform,
      manifestJson: manifestJson,
      modelBuffers: modelBuffers,
    ).run();
    Isolate.exit(sendPort, {'type': 'complete', 'path': path});
  } catch (error, stack) {
    Isolate.exit(sendPort, {
      'type': 'error',
      'error': '$error',
      'stack': '$stack',
    });
  }
}

class ModelSpec {
  ModelSpec.fromJson(Map<String, dynamic> value)
    : id = value['id'],
      asset = value['asset'],
      sha256 = value['sha256'],
      architecture = value['architecture'],
      quantization = value['quantization'],
      inputDtype = value['input_dtype'],
      outputDtype = value['output_dtype'];

  final String id;
  final String asset;
  final String sha256;
  final String architecture;
  final String quantization;
  final String inputDtype;
  final String outputDtype;
}

class Configuration {
  Configuration(this.model, this.runtime, this.threads);

  final ModelSpec model;
  final String runtime;
  final int threads;
}

class BenchmarkRunner {
  BenchmarkRunner({
    required this.onProgress,
    required this.platform,
    required this.manifestJson,
    required this.modelBuffers,
  });

  final ProgressCallback onProgress;
  final MethodChannel platform;
  final String manifestJson;
  final Map<String, Uint8List> modelBuffers;
  static const platformChannelName = 'quant_benchmark/platform';
  static const header = [
    'timestamp_utc',
    'protocol_version',
    'device_id',
    'app_version',
    'build_mode',
    'model_id',
    'model_sha256',
    'architecture',
    'quantization',
    'input_dtype',
    'output_dtype',
    'input_shape',
    'runtime',
    'requested_delegate',
    'effective_delegate',
    'delegate_error',
    'threads',
    'trial_id',
    'randomized_order_index',
    'phase',
    'run_index',
    'latency_ms',
    'model_load_ms',
    'process_pss_mb',
    'process_rss_mb',
    'battery_percent',
    'charging_state',
    'battery_saver',
    'thermal_status',
    'soc_temperature_c',
    'gpu_temperature_c',
    'battery_temperature_c',
    'screen_policy',
    'background_load_policy',
    'error',
  ];

  Future<String> run() async {
    final preflight = Map<String, dynamic>.from(
      await platform.invokeMapMethod<String, dynamic>('deviceContext') ?? {},
    );
    if (preflight['battery_saver'] == true) {
      throw StateError(
        'Frozen protocol requires Android battery saver to be off',
      );
    }
    final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
    final models = (manifest['models'] as List)
        .map((value) => ModelSpec.fromJson(Map<String, dynamic>.from(value)))
        .toList();
    final configurations = <Configuration>[
      for (final model in models)
        for (final runtime in (manifest['runtimes'] as List).cast<String>())
          for (final threads in (manifest['threads'] as List).cast<int>())
            Configuration(model, runtime, threads),
    ];
    final root = await platform.invokeMethod<String>('externalFilesPath');
    if (root == null) throw StateError('No external files directory');
    final stamp = DateTime.now().toUtc().toIso8601String().replaceAll(':', '-');
    final file = File('$root/benchmark_raw_$stamp.csv');
    final sink = file.openWrite()..writeln(header.join(','));
    final random = Random(manifest['random_seed']);
    for (var trial = 1; trial <= manifest['complete_trials']; trial++) {
      final order = List<Configuration>.from(configurations)..shuffle(random);
      for (var index = 0; index < order.length; index++) {
        final configuration = order[index];
        onProgress(
          'trial $trial ${index + 1}/${order.length}: '
          '${configuration.model.id} ${configuration.runtime} '
          '${configuration.threads}t',
        );
        await _configuration(
          configuration,
          trial,
          index,
          manifest['warmup_runs'],
          manifest['measured_runs'],
          sink,
        );
        await sink.flush();
      }
    }
    await sink.close();
    await File('$root/latest_benchmark_status.json').writeAsString(
      jsonEncode({
        'status': 'COMPLETE',
        'raw_csv': file.path,
        'finished_utc': DateTime.now().toUtc().toIso8601String(),
      }),
      flush: true,
    );
    return file.path;
  }

  Future<void> _configuration(
    Configuration configuration,
    int trial,
    int orderIndex,
    int warmups,
    int measured,
    IOSink sink,
  ) async {
    Interpreter? interpreter;
    XNNPackDelegate? delegate;
    var effective = 'none_builtin';
    var delegateError = '';
    var loadMs = 0.0;
    try {
      final options = InterpreterOptions()..threads = configuration.threads;
      if (configuration.runtime == 'xnnpack_cpu') {
        delegate = XNNPackDelegate();
        options.addDelegate(delegate);
        effective = 'xnnpack_initialized_partition_unverified';
      }
      final modelBuffer = modelBuffers[configuration.model.id];
      if (modelBuffer == null) {
        throw StateError('Missing transferred model buffer');
      }
      final loadTimer = Stopwatch()..start();
      interpreter = Interpreter.fromBuffer(modelBuffer, options: options);
      loadTimer.stop();
      loadMs = loadTimer.elapsedMicroseconds / 1000;
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      if (_type(inputTensor.type) != configuration.model.inputDtype ||
          _type(outputTensor.type) != configuration.model.outputDtype) {
        throw StateError('Manifest tensor type mismatch');
      }
      final input = _input(inputTensor);
      final output = _output(outputTensor);
      for (var index = 0; index < warmups; index++) {
        interpreter.run(input, output);
      }
      for (var index = 0; index < measured; index++) {
        final timer = Stopwatch()..start();
        interpreter.run(input, output);
        timer.stop();
        sink.writeln(
          await _row(
            configuration,
            trial,
            orderIndex,
            index,
            timer.elapsedMicroseconds / 1000,
            loadMs,
            effective,
            '',
            '',
          ),
        );
      }
    } catch (error) {
      delegateError = '$error';
      sink.writeln(
        await _row(
          configuration,
          trial,
          orderIndex,
          -1,
          0,
          loadMs,
          effective,
          delegateError,
          delegateError,
        ),
      );
    } finally {
      interpreter?.close();
      delegate?.delete();
    }
  }

  Object _input(Tensor tensor) {
    final count = tensor.shape.reduce((a, b) => a * b);
    if (tensor.type == TensorType.float32) {
      return Float32List.fromList(
        List.generate(count, (index) => (index % 256).toDouble()),
      ).reshape(tensor.shape);
    }
    if (tensor.params.scale <= 0) {
      throw StateError('Invalid integer input scale');
    }
    final values = List<int>.generate(
      count,
      (index) => ((index % 256) / tensor.params.scale + tensor.params.zeroPoint)
          .round(),
    );
    return tensor.type == TensorType.int8
        ? Int8List.fromList(
            values.map((value) => value.clamp(-128, 127)).toList(),
          ).reshape(tensor.shape)
        : Uint8List.fromList(
            values.map((value) => value.clamp(0, 255)).toList(),
          ).reshape(tensor.shape);
  }

  Object _output(Tensor tensor) {
    final count = tensor.shape.reduce((a, b) => a * b);
    if (tensor.type == TensorType.float32) {
      return Float32List(count).reshape(tensor.shape);
    }
    return tensor.type == TensorType.int8
        ? Int8List(count).reshape(tensor.shape)
        : Uint8List(count).reshape(tensor.shape);
  }

  Future<String> _row(
    Configuration configuration,
    int trial,
    int orderIndex,
    int runIndex,
    double latency,
    double loadMs,
    String effective,
    String delegateError,
    String error,
  ) async {
    final device = Map<String, dynamic>.from(
      await platform.invokeMapMethod<String, dynamic>('deviceContext') ?? {},
    );
    final values = <String, Object?>{
      'timestamp_utc': DateTime.now().toUtc().toIso8601String(),
      'protocol_version': '1.2.0',
      'device_id': device['device_id'] ?? 'unknown',
      'app_version': '1.2.0+3',
      'build_mode': 'release',
      'model_id': configuration.model.id,
      'model_sha256': configuration.model.sha256,
      'architecture': configuration.model.architecture,
      'quantization': configuration.model.quantization,
      'input_dtype': configuration.model.inputDtype,
      'output_dtype': configuration.model.outputDtype,
      'input_shape': '1x224x224x3',
      'runtime': configuration.runtime,
      'requested_delegate': configuration.runtime == 'xnnpack_cpu'
          ? 'xnnpack'
          : 'none',
      'effective_delegate': effective,
      'delegate_error': delegateError,
      'threads': configuration.threads,
      'trial_id': trial,
      'randomized_order_index': orderIndex,
      'phase': runIndex < 0 ? 'configuration_error' : 'measured',
      'run_index': runIndex,
      'latency_ms': latency == 0 ? '' : latency.toStringAsFixed(6),
      'model_load_ms': loadMs.toStringAsFixed(6),
      'process_pss_mb': device['process_pss_mb'] ?? '',
      'process_rss_mb': device['process_rss_mb'] ?? '',
      'battery_percent': device['battery_percent'] ?? '',
      'charging_state': device['charging_state'] ?? '',
      'battery_saver': device['battery_saver'] ?? '',
      'thermal_status': device['thermal_status'] ?? '',
      'soc_temperature_c': '',
      'gpu_temperature_c': '',
      'battery_temperature_c': device['battery_temperature_c'] ?? '',
      'screen_policy': 'FLAG_KEEP_SCREEN_ON',
      'background_load_policy': 'no_load_injected_background_apps_unverified',
      'error': error,
    };
    return header.map((name) => _escape('${values[name] ?? ''}')).join(',');
  }

  String _escape(String value) => value.contains(RegExp('[,\\n"]'))
      ? '"${value.replaceAll('"', '""')}"'
      : value;

  String _type(TensorType type) => type == TensorType.float32
      ? 'float32'
      : type == TensorType.int8
      ? 'int8'
      : type == TensorType.uint8
      ? 'uint8'
      : '$type';
}
