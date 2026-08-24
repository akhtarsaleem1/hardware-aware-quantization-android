import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class DemoModel {
  DemoModel.fromJson(Map<String, dynamic> value)
    : id = value['id'],
      asset = value['asset'],
      architecture = value['architecture'],
      quantization = value['quantization'];

  final String id;
  final String asset;
  final String architecture;
  final String quantization;
}

class InferenceDemo extends StatefulWidget {
  const InferenceDemo({super.key});

  @override
  State<InferenceDemo> createState() => _InferenceDemoState();
}

class _InferenceDemoState extends State<InferenceDemo> {
  static const labels = [
    'Chinee Apple',
    'Lantana',
    'Parkinsonia',
    'Parthenium',
    'Prickly Acacia',
    'Rubber Vine',
    'Siam Weed',
    'Snake Weed',
    'Negative',
  ];
  final picker = ImagePicker();
  List<DemoModel> models = [];
  DemoModel? selected;
  Uint8List? imageBytes;
  String status = 'Choose a model and an image.';
  String prediction = '';
  double confidence = 0;
  double inferenceMs = 0;
  double modelSizeMb = 0;
  bool busy = false;

  @override
  void initState() {
    super.initState();
    _loadManifest();
  }

  Future<void> _loadManifest() async {
    final manifest =
        jsonDecode(
              await rootBundle.loadString('assets/benchmark_manifest.json'),
            )
            as Map<String, dynamic>;
    final loaded = (manifest['models'] as List)
        .map((value) => DemoModel.fromJson(Map<String, dynamic>.from(value)))
        .toList();
    if (!mounted) return;
    setState(() {
      models = loaded;
      selected = loaded.isEmpty ? null : loaded.first;
    });
  }

  Future<void> _choose(ImageSource source) async {
    final file = await picker.pickImage(source: source, imageQuality: 100);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    if (!mounted) return;
    setState(() {
      imageBytes = bytes;
      prediction = '';
      status = 'Image selected. Run local inference when ready.';
    });
  }

  Future<void> _infer() async {
    final model = selected;
    final encoded = imageBytes;
    if (model == null || encoded == null || busy) return;
    setState(() {
      busy = true;
      status = 'Loading ${model.id}…';
    });
    Interpreter? interpreter;
    try {
      final decoded = img.decodeImage(encoded);
      if (decoded == null) throw StateError('Unsupported image format');
      final side = decoded.width < decoded.height
          ? decoded.width
          : decoded.height;
      final cropped = img.copyCrop(
        decoded,
        x: (decoded.width - side) ~/ 2,
        y: (decoded.height - side) ~/ 2,
        width: side,
        height: side,
      );
      final resized = img.copyResize(cropped, width: 224, height: 224);
      interpreter = await Interpreter.fromAsset(model.asset);
      final inputTensor = interpreter.getInputTensor(0);
      final outputTensor = interpreter.getOutputTensor(0);
      final input = _input(resized, inputTensor);
      final output = _output(outputTensor);
      interpreter.run(input, output);
      final timer = Stopwatch()..start();
      interpreter.run(input, output);
      timer.stop();
      final probabilities = _probabilities(output, outputTensor);
      var best = 0;
      for (var index = 1; index < probabilities.length; index++) {
        if (probabilities[index] > probabilities[best]) best = index;
      }
      final asset = await rootBundle.load(model.asset);
      if (!mounted) return;
      setState(() {
        prediction = labels[best];
        confidence = probabilities[best];
        inferenceMs = timer.elapsedMicroseconds / 1000;
        modelSizeMb = asset.lengthInBytes / (1024 * 1024);
        status = 'Inference completed locally; no image left the device.';
      });
    } catch (error) {
      if (mounted) setState(() => status = 'Inference error: $error');
    } finally {
      interpreter?.close();
      if (mounted) setState(() => busy = false);
    }
  }

  Object _input(img.Image image, Tensor tensor) {
    final values = List<double>.generate(224 * 224 * 3, (index) {
      final pixelIndex = index ~/ 3;
      final pixel = image.getPixel(pixelIndex % 224, pixelIndex ~/ 224);
      return switch (index % 3) {
        0 => pixel.r,
        1 => pixel.g,
        _ => pixel.b,
      }.toDouble();
    });
    if (tensor.type == TensorType.float32) {
      return Float32List.fromList(values).reshape(tensor.shape);
    }
    if (tensor.params.scale <= 0) {
      throw StateError('Invalid input quantization scale');
    }
    final quantized = values.map(
      (value) =>
          (value / tensor.params.scale + tensor.params.zeroPoint).round(),
    );
    if (tensor.type == TensorType.int8) {
      return Int8List.fromList(
        quantized.map((value) => value.clamp(-128, 127)).toList(),
      ).reshape(tensor.shape);
    }
    return Uint8List.fromList(
      quantized.map((value) => value.clamp(0, 255)).toList(),
    ).reshape(tensor.shape);
  }

  Object _output(Tensor tensor) {
    final count = tensor.shape.reduce((a, b) => a * b);
    if (tensor.type == TensorType.float32) {
      return Float32List(count).reshape(tensor.shape);
    }
    if (tensor.type == TensorType.int8) {
      return Int8List(count).reshape(tensor.shape);
    }
    return Uint8List(count).reshape(tensor.shape);
  }

  List<double> _probabilities(Object output, Tensor tensor) {
    final raw = (output as List).first as List;
    if (tensor.type == TensorType.float32) {
      return raw.map((value) => (value as num).toDouble()).toList();
    }
    return raw
        .map(
          (value) =>
              ((value as num).toDouble() - tensor.params.zeroPoint) *
              tensor.params.scale,
        )
        .toList();
  }

  @override
  Widget build(BuildContext context) => ListView(
    padding: const EdgeInsets.all(16),
    children: [
      const Card(
        child: Padding(
          padding: EdgeInsets.all(12),
          child: Text(
            'Research prototype: all inference is local. Predictions are '
            'experimental and must not be used for agronomic decisions.',
          ),
        ),
      ),
      DropdownButtonFormField<DemoModel>(
        initialValue: selected,
        decoration: const InputDecoration(labelText: 'Model configuration'),
        items: models
            .map(
              (model) => DropdownMenuItem(
                value: model,
                child: Text('${model.architecture} · ${model.quantization}'),
              ),
            )
            .toList(),
        onChanged: busy ? null : (value) => setState(() => selected = value),
      ),
      const SizedBox(height: 12),
      if (imageBytes != null)
        ClipRRect(
          borderRadius: BorderRadius.circular(12),
          child: Image.memory(imageBytes!, height: 260, fit: BoxFit.cover),
        ),
      const SizedBox(height: 12),
      Row(
        children: [
          Expanded(
            child: OutlinedButton.icon(
              onPressed: busy ? null : () => _choose(ImageSource.gallery),
              icon: const Icon(Icons.photo_library_outlined),
              label: const Text('Gallery'),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: OutlinedButton.icon(
              onPressed: busy ? null : () => _choose(ImageSource.camera),
              icon: const Icon(Icons.camera_alt_outlined),
              label: const Text('Camera'),
            ),
          ),
        ],
      ),
      FilledButton.icon(
        onPressed: busy || imageBytes == null ? null : _infer,
        icon: const Icon(Icons.memory),
        label: Text(busy ? 'Running…' : 'Run local inference'),
      ),
      if (prediction.isNotEmpty)
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  prediction,
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                Text('Confidence: ${(confidence * 100).toStringAsFixed(2)}%'),
                Text('Inference: ${inferenceMs.toStringAsFixed(3)} ms'),
                Text('Model size: ${modelSizeMb.toStringAsFixed(2)} MB'),
                Text('Input: 224 × 224 RGB'),
              ],
            ),
          ),
        ),
      Text(status),
    ],
  );
}
