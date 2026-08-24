Small reference models vendored from https://github.com/sdatkinson/NeuralAmpModelerCore (MIT license, commit e49c93e) example_models/ for format-parity tests:

- wavenet.nam            tiny 2-array WaveNet, 131 weights, file version 0.5.4
- lstm.nam               1-layer LSTM (hidden 3), 70 weights, file version 0.5.4
- slimmable_wavenet.nam  slimmable WaveNet (457 weights, version 0.7.0) — used to test the explicit unsupported-feature error
- slimmable_container.nam   SlimmableContainer (3 submodels, version 0.7.0) — container-loading test
