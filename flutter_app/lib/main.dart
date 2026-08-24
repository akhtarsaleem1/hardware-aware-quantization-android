import 'dart:async';

import 'package:flutter/material.dart';

import 'benchmark_runner.dart';
import 'inference_demo.dart';

const autoBenchmark = bool.fromEnvironment(
  'AUTO_BENCHMARK',
  defaultValue: false,
);

void main() => runApp(const ResearchApp());

class ResearchApp extends StatelessWidget {
  const ResearchApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'LiteRT Quantization Research',
    theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
    home: const ResearchHome(),
  );
}

class ResearchHome extends StatefulWidget {
  const ResearchHome({super.key});

  @override
  State<ResearchHome> createState() => _ResearchHomeState();
}

class _ResearchHomeState extends State<ResearchHome> {
  int index = autoBenchmark ? 1 : 0;

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      title: Text(
        index == 0 ? 'Local Inference Prototype' : 'LiteRT Experiment Mode',
      ),
    ),
    body: IndexedStack(
      index: index,
      children: const [
        InferenceDemo(),
        BenchmarkScreen(autoStart: autoBenchmark),
      ],
    ),
    bottomNavigationBar: NavigationBar(
      selectedIndex: index,
      onDestinationSelected: (value) => setState(() => index = value),
      destinations: const [
        NavigationDestination(
          icon: Icon(Icons.image_search),
          label: 'Inference',
        ),
        NavigationDestination(
          icon: Icon(Icons.science_outlined),
          label: 'Experiment',
        ),
      ],
    ),
  );
}

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key, required this.autoStart});

  final bool autoStart;

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  final messages = <String>[];
  bool running = false;
  String result = '';

  @override
  void initState() {
    super.initState();
    if (widget.autoStart) scheduleMicrotask(runBenchmark);
  }

  Future<void> runBenchmark() async {
    if (running) return;
    setState(() {
      running = true;
      messages.clear();
      result = '';
    });
    try {
      result = await runBenchmarkInBackground(
        onProgress: (message) {
          if (!mounted) return;
          setState(() {
            messages.add(message);
            if (messages.length > 24) messages.removeAt(0);
          });
        },
      );
    } catch (error, stack) {
      messages.add('FATAL: $error\n$stack');
    } finally {
      if (mounted) setState(() => running = false);
    }
  }

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.all(16),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Card(
          child: Padding(
            padding: EdgeInsets.all(12),
            child: Text(
              'Frozen release-mode benchmark: randomized configurations, 20 '
              'warm-ups, 100 measured runs, three complete trials. Raw '
              'per-inference rows are preserved in app-specific storage.',
            ),
          ),
        ),
        LinearProgressIndicator(value: running ? null : 1),
        if (result.isNotEmpty) SelectableText('COMPLETE\n$result'),
        Expanded(child: ListView(children: messages.map(Text.new).toList())),
        FilledButton(
          onPressed: running ? null : runBenchmark,
          child: const Text('Run frozen protocol'),
        ),
      ],
    ),
  );
}
