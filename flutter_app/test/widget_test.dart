import 'package:flutter_test/flutter_test.dart';
import 'package:quant_repeatability_benchmark/main.dart';

void main() {
  testWidgets('renders the local inference research interface', (tester) async {
    await tester.pumpWidget(const ResearchApp());
    await tester.pump();

    expect(find.text('Local Inference Prototype'), findsOneWidget);
    expect(find.text('Inference'), findsOneWidget);
    expect(find.text('Experiment'), findsOneWidget);
  });
}
